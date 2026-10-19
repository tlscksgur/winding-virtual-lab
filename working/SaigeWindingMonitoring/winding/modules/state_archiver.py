"""
State-based frame archiving module for winding monitoring system.

This module provides functionality to automatically archive frame data during specific
state transitions, particularly during the swing operation phases.
"""

from __future__ import annotations

import os
import pickle
import zipfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal

from ..modules import logging as winding_logging
from ..engine.config import CURRENT_SITE


ProcessState = Literal[
    "winding",
    "waiting_tightening",
    "tightening_start",
    "tightening",
    "tightening_end",
    "protrusion_detection_start",
    "protrusion_detection",
    "protrusion_detection_end",
    "waiting_for_first_swing",
    "first_swing_start",
    "first_swing",
    "first_swing_end",
    "swing_term",
    "second_swing_start",
    "second_swing",
    "second_swing_end",
    "auto_swing_skip_reset",
]


class FrameMetadata:
    """Metadata for a single frame."""

    def __init__(
        self,
        file_path: str,
        state: str | None,
        process_state: ProcessState | None,
        state_index: int | None,
        frame_counter: int,
    ):
        self.file_path = file_path
        self.state = state
        self.process_state = process_state
        self.state_index = state_index
        self.frame_counter = frame_counter

    def __repr__(self):
        return (
            f"FrameMetadata(frame={self.frame_counter}, "
            f"state={self.state}, process_state={self.process_state}, "
            f"state_index={self.state_index})"
        )


class StateArchiver:
    """
    Manages automatic archiving of frames during specific state transitions.

    This class monitors the state transitions of the winding monitoring system
    and automatically creates compressed archives of frame data during swing operations.
    """

    # Target states to record
    TARGET_PROCESS_STATES = {
        "waiting_tightening",
        "tightening_start",
        "tightening",
        "tightening_end",
        "protrusion_detection_start",
        "protrusion_detection",
        "protrusion_detection_end",
        "waiting_for_first_swing",
        "first_swing_start",
        "first_swing",
        "first_swing_end",
        "swing_term",
        "second_swing_start",
        "second_swing",
    }

    # Process states that map to simplified state names
    PROCESS_STATE_MAPPING = {
        "waiting_tightening": "waiting_tightening",
        "tightening_start": "tightening",
        "tightening": "tightening",
        "tightening_end": "tightening",
        "protrusion_detection_start": "protrusion_detection",
        "protrusion_detection": "protrusion_detection",
        "protrusion_detection_end": "protrusion_detection",
        "waiting_for_first_swing": "first_swing",
        "first_swing_start": "first_swing",
        "first_swing": "first_swing",
        "first_swing_end": "first_swing",
        "swing_term": "swing_term",
        "second_swing_start": "second_swing",
        "second_swing": "second_swing",
    }

    def __init__(self, buffer_size: int = 15, max_workers: int = 2):
        """
        Initialize the StateArchiver.

        Args:
            buffer_size: Number of frames to keep before and after the target segment
            max_workers: Maximum number of concurrent compression threads
        """
        self._buffer_size = buffer_size
        self._buffer: deque[FrameMetadata] = deque(maxlen=buffer_size)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="StateArchiver")

        # Recording state
        self._recording = False
        self._recorded_frames: list[FrameMetadata] = []

        # Tracking last state_index=2 frame during target phases (second_swing or auto_swing_skip_reset)
        self._last_shift_frame: FrameMetadata | None = None
        self._frames_after_last_shift = 0
        self._in_target_phase = False

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._archive_count = 0

    def on_frame_saved(
        self,
        file_path: str,
        state: str | None,
        process_state: ProcessState | None,
        state_index: int | None,
        frame_counter: int,
    ) -> None:
        """
        Called when a frame is saved. Updates internal state and triggers recording/compression.

        Args:
            file_path: Path to the saved pkl file
            state: Current state name
            process_state: Current process state
            state_index: Current state index (1-5)
            frame_counter: Frame counter number
        """
        with self._lock:
            metadata = FrameMetadata(file_path, state, process_state, state_index, frame_counter)

            # Always add to buffer for potential use
            self._buffer.append(metadata)

            # Check for recording start
            if not self._recording and process_state == "waiting_tightening":
                self._start_recording()

            # If recording, collect frames
            if self._recording:
                self._recorded_frames.append(metadata)

                # Track target phases (second_swing or auto_swing_skip_reset)
                if process_state in ["second_swing_start", "second_swing", "auto_swing_skip_reset"]:
                    self._in_target_phase = True

                    # Track last frame with state_index=2
                    if state_index == 2:
                        self._last_shift_frame = metadata
                        self._frames_after_last_shift = 0
                    elif self._last_shift_frame is not None:
                        self._frames_after_last_shift += 1

                # Check for recording end
                if self._should_stop_recording():
                    self._trigger_compression()

    def _start_recording(self) -> None:
        """Start recording frames. Includes the last buffer_size frames."""
        self._recording = True
        self._recorded_frames = list(self._buffer)  # Include previous frames
        self._last_shift_frame = None
        self._frames_after_last_shift = 0
        self._in_second_swing = False

        winding_logging.log(">>> StateArchiver: Recording started (waiting_tightening detected)")

    def _should_stop_recording(self) -> bool:
        """
        Check if recording should stop.

        Recording stops when:
        - We are in target phase (second_swing or auto_swing_skip_reset)
        - We have detected at least one state_index=2 frame
        - buffer_size frames have passed since the last state_index=2 frame
        """
        if not self._in_target_phase:
            return False

        if self._last_shift_frame is None:
            return False

        return self._frames_after_last_shift >= self._buffer_size

    def _trigger_compression(self) -> None:
        """Trigger asynchronous compression of recorded frames."""
        if not self._recorded_frames:
            winding_logging.log(">>> StateArchiver: No frames to compress")
            self._reset_recording_state()
            return

        # Prepare data for compression
        file_paths = [frame.file_path for frame in self._recorded_frames]
        first_frame = self._recorded_frames[0]
        last_frame = self._recorded_frames[-1]

        # Generate output filename using first frame's timestamp
        first_pkl_name = os.path.basename(first_frame.file_path)
        timestamp_part = first_pkl_name.split("_", 1)[1].replace(".pkl", "")  # YYMMDD_HHMMSS

        # Remove seconds from timestamp: YYMMDD_HHMMSS -> YYMMDD_HHMM
        date_part, time_part = timestamp_part.split("_")  # ['YYMMDD', 'HHMMSS']
        time_without_seconds = time_part[:-2]  # HHMMSS -> HHMM
        timestamp_without_seconds = f"{date_part}_{time_without_seconds}"

        output_dir = os.path.dirname(first_frame.file_path)
        output_filename = f"{timestamp_without_seconds}.zip"
        output_path = os.path.join(output_dir, output_filename)

        frame_count = len(file_paths)
        winding_logging.log(
            f">>> StateArchiver: Triggering compression of {frame_count} frames "
            f"(frames {first_frame.frame_counter} to {last_frame.frame_counter})"
        )

        # Submit compression task (non-blocking)
        self._executor.submit(
            self._compress_pkl_files,
            file_paths,
            output_path,
            frame_count,
        )

        # Reset state
        self._reset_recording_state()
        self._archive_count += 1

    def _reset_recording_state(self) -> None:
        """Reset recording state."""
        self._recording = False
        self._recorded_frames = []
        self._last_shift_frame = None
        self._frames_after_last_shift = 0
        self._in_target_phase = False

    def _compress_pkl_files(
        self, file_paths: list[str], output_path: str, frame_count: int
    ) -> None:
        """
        Compress pkl files into a zip archive (runs in separate thread).

        Args:
            file_paths: List of pkl file paths to compress
            output_path: Output zip file path
            frame_count: Number of frames being compressed (for logging)
        """
        try:
            winding_logging.log(
                f">>> StateArchiver: Starting compression to {os.path.basename(output_path)}"
            )

            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname=arcname)

            # Get compressed size
            size_mb = os.path.getsize(output_path) / (1024 * 1024)

            winding_logging.log(
                f">>> StateArchiver: Compression completed successfully "
                f"({frame_count} frames, {size_mb:.2f} MB) -> {os.path.basename(output_path)}"
            )


        except Exception as e:
            winding_logging.log(f">>> StateArchiver: Compression failed: {e}")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the archiver and cleanup resources.

        Args:
            wait: If True, wait for all pending compression tasks to complete
        """
        winding_logging.log(
            f">>> StateArchiver: Shutting down (archives created: {self._archive_count})"
        )
        self._executor.shutdown(wait=wait)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

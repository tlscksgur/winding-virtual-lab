"""Extract frames from videos for dataset creation."""

from pathlib import Path

import cv2
from SaigeToolkit.log.log_handler import LogHandler
from SaigeToolkit.util.reproducibility import get_function_args


def extract_frame(
    source_dir: str,
    target_dir: str,
    target_fps: float = 0.01,
    distance_threshold: float = 30000.0,
    log_handler: dict | LogHandler | None = None,
    **kwargs,
):
    """Extract sparse frames from videos in a directory.

    This function processes all video files in the source directory and extracts
    frames based on the specified FPS and frame distance threshold.

    Args:
        source_dir: Directory containing input video files
        target_dir: Base directory for extracted frames
        target_fps: Target frame extraction rate (frames per second)
        distance_threshold: Minimum L2 distance required to save a frame
        log_handler: Optional logger for status messages
        **kwargs: Additional arguments (ignored)
    """
    args_dict = get_function_args()

    if log_handler is None:
        log_handler = LogHandler(stdout=True)
    elif isinstance(log_handler, dict):
        log_handler = LogHandler(**log_handler)

    log_handler.save_yaml(args_dict, "extract_frame.yaml")

    # Constants
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MOV", ".AVI")
    FRAME_FILENAME_PATTERN = "out{:05d}.jpg"
    DEFAULT_INITIAL_DISTANCE = 1e10

    source_path = Path(source_dir)

    if not source_path.exists():
        log_handler.logger.info(f"Source directory does not exist: {source_dir}")
        return

    # Find all video files
    video_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix in VIDEO_EXTENSIONS]

    if not video_files:
        log_handler.logger.info(f"No video files found in {source_dir}")
        return

    log_handler.logger.info(f"Found {len(video_files)} video file(s) to process")

    total_frames = 0
    for video_idx, video_file in enumerate(video_files, 1):
        log_handler.logger.info(f"[{video_idx}/{len(video_files)}] Processing: {video_file.name}")

        # Create output directory for this video
        output_subdir = f"{video_file.stem}_jpg_fps{target_fps}"
        output_path = Path(target_dir) / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract frames from video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            log_handler.logger.info(f"Cannot open video file: {video_file}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        log_handler.logger.info(f"Video FPS: {video_fps}")

        if video_fps is None or video_fps <= 0:
            log_handler.logger.info(f"Invalid FPS value for video: {video_file}")
            cap.release()
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(round(video_fps / target_fps)) if target_fps > 0 else 1

        log_handler.logger.info(f"Extracting every {frame_interval} frames (target FPS: {target_fps})")

        saved_count = 0
        prev_frame = None

        for frame_idx in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Calculate frame distance
            if prev_frame is None:
                frame_distance = DEFAULT_INITIAL_DISTANCE
            else:
                frame_distance = cv2.norm(prev_frame, frame, cv2.NORM_L2)

            # Only save frames that differ significantly
            if frame_distance > distance_threshold:
                frame_filename = output_path / FRAME_FILENAME_PATTERN.format(saved_count)
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1

            prev_frame = frame

        cap.release()
        log_handler.logger.info(f"Saved {saved_count} frames from {video_file.name}")
        total_frames += saved_count

    log_handler.logger.info(f"Total frames extracted: {total_frames}")

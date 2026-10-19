import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Iterable, Iterator, Union
from bisect import bisect_right

import numpy as np
from numpy.typing import NDArray

import av
from av.video.stream import VideoStream
from av.container.input import InputContainer


def _frame_repr(self):
    # Utility to replace __repr__ method of dataclasses below. This prints the
    # shape of the .data tensor rather than printing the (potentially very long)
    # data tensor itself.
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    with np.printoptions(precision=4, suppress=True):
        for field in dataclasses.fields(self):
            field_name = field.name
            field_val = getattr(self, field_name)
            if field_name == "data":
                field_name = "data (shape)"
                field_val = field_val.shape
            s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class Frame(Iterable):
    """A single video frame with associated metadata."""

    data: NDArray
    """The frame data as (3-D ``NDArray``)."""
    pts_seconds: float
    """The :term:`pts` of the frame, in seconds (float)."""

    def __post_init__(self):
        # This is called after __init__() when a Frame is created. We can run
        # input validation checks here.
        if not self.data.ndim == 3:
            raise ValueError(f"data must be 3-dimensional, got {self.data.shape = }")
        self.pts_seconds = float(self.pts_seconds)

    def __iter__(self) -> Iterator[Union[NDArray, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __repr__(self):
        return _frame_repr(self)


@dataclass
class FrameBatch(Iterable):
    """Multiple video frames with associated metadata.

    The ``data`` tensor is typically 4D for sequences of frames (NHWC or NCHW),
    or 5D for sequences of clips, as returned by the :ref:`samplers
    <sphx_glr_generated_examples_decoding_sampling.py>`. When ``data`` is 4D (resp.  5D)
    the ``pts_seconds`` and ``duration_seconds`` tensors are 1D (resp. 2D).

    .. note::
        The ``pts_seconds`` and ``duration_seconds`` Tensors are always returned
        on CPU, even if ``data`` is on GPU.
    """

    data: NDArray
    """The frames data (``NDArray`` of uint8)."""
    pts_seconds: NDArray
    """The :term:`pts` of the frame, in seconds (``NDArray`` of floats)."""

    def __post_init__(self):
        # This is called after __init__() when a FrameBatch is created. We can
        # run input validation checks here.
        if self.data.ndim < 3:
            raise ValueError(f"data must be at least 3-dimensional, got {self.data.shape = }")

        leading_dims = self.data.shape[:-3]
        if not (leading_dims == self.pts_seconds.shape):
            raise ValueError(
                "Tried to create a FrameBatch but the leading dimensions of the inputs do not match. "
                f"Got {self.data.shape = } so we expected the shape of pts_seconds "
                f" to be {leading_dims = }, but got {self.pts_seconds.shape = }."
            )

    def __iter__(self) -> Iterator["FrameBatch"]:
        for data, pts_seconds in zip(self.data, self.pts_seconds):
            yield FrameBatch(
                data=data,
                pts_seconds=pts_seconds,
            )

    def __getitem__(self, key) -> "FrameBatch":
        return FrameBatch(
            data=self.data[key],
            pts_seconds=self.pts_seconds[key],
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _frame_repr(self)


def _is_likely_cfr(
    container: InputContainer,
    stream: VideoStream,
    *,
    max_samples: int = 500,
    threshold: float = 0.01,
):
    container.seek(0, stream=stream)

    time_diff_list = []
    num_diffs = 0
    prev_pts = None

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if prev_pts is not None:
            diff = frame.pts - prev_pts
            if diff > 0:
                time_diff_list.append(diff)
                num_diffs += 1
        prev_pts = frame.pts

        if num_diffs > max_samples:
            break

    if len(time_diff_list) < 2:
        # not enough samples
        return True

    time_diffs = np.asarray(time_diff_list, dtype=np.float64)

    avg_diff = np.mean(time_diffs)
    rel_err = np.abs(time_diffs - avg_diff) / avg_diff
    p95_err = np.percentile(rel_err, 95).item()

    return p95_err <= threshold, p95_err


def _sanity_checks(
    vd: "VideoDecoder",
    file: Union[str, Path],
    *,
    max_frames: int = 500,
    num_samples: int = 10,
) -> bool:
    import cv2

    rand_indices = np.arange(min(max_frames, len(vd)))
    np.random.shuffle(rand_indices)
    rand_indices = rand_indices[:num_samples]
    indices_set = set(rand_indices.tolist())

    frame_batch = vd.get_frames_at(rand_indices.tolist())
    frames = {idx: frame_batch.data[i] for i, idx in enumerate(rand_indices.tolist())}

    cap = cv2.VideoCapture(str(file))
    equal_count, not_equal_count = 0, 0

    for idx in range(max_frames):
        if idx not in indices_set:
            ret = cap.grab()
            if not ret:
                break
            else:
                continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not vd._channel_last:
            frame = frame.transpose(2, 0, 1)

        if np.allclose(frames[idx], frame):
            equal_count += 1
            # print(f"Frame @ {idx}: equal to OpenCV")
        else:
            not_equal_count += 1
            # print(f"Frame @ {idx}: not equal to OpenCV")

    cap.release()

    total_count = equal_count + not_equal_count
    if not_equal_count > 1:
        print(f"Diff frames: {not_equal_count}/{total_count} ({not_equal_count/total_count*100:.2f}%)")

    return not_equal_count <= 1


class VideoDecoder:
    def __init__(
        self,
        file: Union[str, Path],
        *,
        stream_index: Optional[int] = None,
        dimension_order: Literal["NCHW", "NHWC"] = "NHWC",
        check: bool = True,
    ):
        # XXX: current implementation only supports CFR videos
        self.stream_index = stream_index if stream_index is not None else 0
        self.container = av.open(file, mode="r")
        self.vstream = self.container.streams.video[self.stream_index]
        self.vstream.thread_type = "AUTO"

        assert self.vstream.average_rate is not None
        assert self.vstream.time_base is not None
        assert self.vstream.start_time is not None

        if not _is_likely_cfr(self.container, self.vstream):
            raise NotImplementedError("Currently only supports CFR videos (VFR videos not supported)")

        if self.vstream.average_rate is None:
            raise NotImplementedError("Currently only supports videos with average rate")

        # Metadata
        self._average_rate = self.vstream.average_rate
        self._time_base = self.vstream.time_base
        self._time_base_float = float(self._time_base)
        self._num_frames = self.vstream.frames
        self._keyframe_index = self._build_keyframe_index()

        # XXX: this is a hack. Implement better indexing logic.
        self._keyframe_offset = self._keyframe_index[0]

        # Times
        # FIXME: what is start_time?
        self._start_time = self.vstream.start_time
        self._begin_stream_seconds = float(self._start_time * self._time_base)
        self._end_stream_seconds = float(self._start_time + self._num_frames / self._average_rate)

        # Transform
        if dimension_order not in ("NCHW", "NHWC"):
            raise ValueError(
                f"Invalid dimension order ({dimension_order}). " "Supported values are NCHW, NHWC."
            )

        self._channel_last = dimension_order == "NHWC"

        if check:
            if not _sanity_checks(self, file):
                raise ValueError("Sanity checks failed")

    def __len__(self) -> int:
        return self._num_frames

    def _seek(self, offset: int) -> None:
        self.container.seek(offset, stream=self.vstream, any_frame=False, backward=True)

    def _to_ndarray(self, frame: Frame) -> NDArray:
        array = frame.to_ndarray(format="rgb24")
        if not self._channel_last:
            array = array.transpose(2, 0, 1)
        return array

    def _build_keyframe_index(self) -> list[int]:
        self._seek(0)
        keyframe_index = [
            packet.pts
            for packet in self.container.demux(self.vstream)
            if packet.is_keyframe and packet.pts is not None
        ]
        return keyframe_index

    def _key_pts_from_pts(self, pts: int) -> int:
        key_pts_idx = bisect_right(self._keyframe_index, pts)
        key_pts = self._keyframe_index[key_pts_idx - 1]
        key_pts += self._keyframe_offset
        return key_pts

    def _batch_key_pts_from_pts(self, pts_list: list[int]) -> list[tuple[int, list[int]]]:
        def _key_pts_bound(pts: int):
            key_pts_idx = bisect_right(self._keyframe_index, pts)
            key_pts_left = self._keyframe_index[key_pts_idx - 1]
            if key_pts_idx < len(self._keyframe_index):
                key_pts_right = self._keyframe_index[key_pts_idx]
            else:
                key_pts_right = 100000000000000
            return key_pts_left, key_pts_right

        sorted_pts = sorted(set(pts_list))
        key_pts_left, key_pts_right = _key_pts_bound(sorted_pts[0])

        current_pts_list = []
        batch_pts = []

        for pts in sorted_pts:
            if key_pts_left <= pts < key_pts_right:
                current_pts_list.append(pts)
            else:
                batch_pts.append((key_pts_left, current_pts_list))
                key_pts_left, key_pts_right = _key_pts_bound(pts)
                current_pts_list = [pts]

        if current_pts_list:
            batch_pts.append((key_pts_left, current_pts_list))

        return batch_pts

    def _pts_from_frame_index(self, frame_index: int) -> int:
        if frame_index < 0:
            frame_index += self._num_frames
        if frame_index >= self._num_frames or frame_index < 0:
            raise IndexError(f"Index {frame_index} is out of bounds; length is {self._num_frames}")
        pts = round((self._start_time + frame_index / self._average_rate) / self._time_base)
        pts += self._keyframe_offset
        return pts

    def _getframe(self, pts: int) -> tuple[NDArray, float]:
        key_pts = self._key_pts_from_pts(pts)
        self._seek(key_pts)

        for packet in self.container.demux(self.vstream):
            for frame in packet.decode():
                if frame.pts >= pts:
                    return (
                        self._to_ndarray(frame),  # type: ignore
                        frame.pts * self._time_base_float,
                    )

        raise RuntimeError(f"Frame at pts {pts} not found")

    def _getframes(self, pts_list: list[int]) -> list[tuple[NDArray, float]]:
        batch_pts = self._batch_key_pts_from_pts(pts_list)
        frame_dict = {}

        for key_pts, _pts_list in batch_pts:
            self._seek(key_pts)
            target_pts_iter = iter(_pts_list)
            target_pts = next(target_pts_iter)

            for packet in self.container.demux(self.vstream):
                for frame in packet.decode():
                    if frame.pts >= target_pts:
                        frame_dict[target_pts] = (
                            self._to_ndarray(frame),  # type: ignore
                            frame.pts * self._time_base_float,
                        )
                        # TODO 같은 것들 같이 묶기
                        try:
                            target_pts = next(target_pts_iter)
                        except StopIteration:
                            target_pts = -1
                            break
                if target_pts == -1:
                    break

        # TODO: add (near) infinite loop guard for edge cases

        return [frame_dict[pts] for pts in pts_list]

    def get_frame_at(self, index: int) -> Frame:
        """Return a single frame at the given index.

        If you need to decode multiple frames, we recommend using the batch
        methods instead, since they are faster:
        `VideoDecoder.get_frames_at`,
        `VideoDecoder.get_frames_in_range`,
        `VideoDecoder.get_frames_played_at`,
        `VideoDecoder.get_frames_played_in_range`.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            Frame: The frame at the given index.
        """
        pts = self._pts_from_frame_index(index)
        data, pts_seconds = self._getframe(pts)

        return Frame(
            data=data,
            pts_seconds=pts_seconds,
        )

    def get_frames_at(self, indices: list[int]) -> FrameBatch:
        """Return frames at the given indices.

        Args:
            indices (list of int): The indices of the frames to retrieve.

        Returns:
            FrameBatch: The frames at the given indices.
        """
        pts_list = [self._pts_from_frame_index(index) for index in indices]
        data_pts_list = self._getframes(pts_list)

        return FrameBatch(
            data=np.stack([data for data, _ in data_pts_list]),
            pts_seconds=np.asarray([pts for _, pts in data_pts_list]),
        )

    @property
    def fps(self) -> int:
        return round(float(self._average_rate))

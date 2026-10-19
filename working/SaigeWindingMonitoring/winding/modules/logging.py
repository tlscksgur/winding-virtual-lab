import os
import pickle
import zipfile
from functools import cache, lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Literal, TypeAlias, overload

import cv2
import zstandard as zstd

import numpy as np
from numpy.typing import NDArray

from ..engine.config import LoggingSettings
from ..data.data_class import SegmentationResults, SideSegmentationResults
from .state_archiver import StateArchiver


_KST = timezone(timedelta(seconds=32400))
_LOGGER = None


def _compress(data):
    return zstd.compress(pickle.dumps(data))


def _decompress(data):
    return pickle.loads(zstd.decompress(data))


State: TypeAlias = Literal[
    "winding",
    "tightening",
    "winding_first_wrap",
    "first_swing",
    "swing_term",
    "second_swing",
    "auto_swing_skip_reset",
]


@dataclass
class Log:
    timestamp: datetime
    results: dict
    metrics: dict
    frame: NDArray[np.uint8]
    side_view_frame: NDArray[np.uint8]
    segmentation_results: SegmentationResults
    side_segmentation_results: SideSegmentationResults

    def __repr__(self):
        return _log_repr(self)


@dataclass
class PartialLog:
    timestamp: datetime
    results: dict
    metrics: dict
    frame: None = None
    side_view_frame: None = None
    segmentation_results: None = None
    side_segmentation_results: None = None

    def __repr__(self):
        return _log_repr(self)


def _log_repr(log: Log | PartialLog) -> str:
    s = log.__class__.__name__ + ":\n"
    spaces = "  "
    s += f"{spaces}timestamp: {log.timestamp}\n"
    s += f"{spaces}frame: "
    if log.frame is None:
        s += "None\n"
    else:
        s += f"NDArray[np.uint8] (shape={log.frame.shape})\n"

    s += f"{spaces}side_view_frame: "
    if log.side_view_frame is None:
        s += "None\n"
    else:
        s += f"NDArray[np.uint8] (shape={log.side_view_frame.shape})\n"

    s += f"{spaces}segmentation_results: "
    if log.segmentation_results is None:
        s += "None\n"
    else:
        s += "SegmentationResults(...)\n"

    s += f"{spaces}side_segmentation_results: "
    if log.side_segmentation_results is None:
        s += "None\n"
    else:
        s += "SideSegmentationResults(...)\n"

    s += f"{spaces}results: {{\n"
    for k, v in log.results.items():
        if k == "segmentation_contours":
            s += f"{spaces}{spaces}{k}: ...\n"
        else:
            s += f"{spaces}{spaces}{k}: {v!r}\n"
    s += f"{spaces}}}\n"

    s += f"{spaces}metrics: {{\n"
    for k, v in log.metrics.items():
        s += f"{spaces}{spaces}{k}: {v!r}\n"
    s += f"{spaces}}}\n"
    return s


def get_logger():
    global _LOGGER
    if _LOGGER is None:
        if LoggingSettings.log_dir is None:
            _LOGGER = NoOpLogger()
        else:
            try:
                _LOGGER = Logger(LoggingSettings())
            except Exception:
                _LOGGER = NoOpLogger()
    return _LOGGER


def supress_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(f"Exception occurred in {func.__name__}")
            pass

    return wrapper


@supress_exceptions
def call_start():
    get_logger().call_start()


@supress_exceptions
def call_end():
    get_logger().call_end()


@supress_exceptions
def reset():
    get_logger().reset()


@supress_exceptions
def dump(
    frame: "NDArray[np.uint8]",
    side_view_frame: "NDArray[np.uint8]",
    segmentation_results: "SegmentationResults",
    side_segmentation_results: "SideSegmentationResults",
    results: dict,
):
    get_logger().dump(frame, side_view_frame, segmentation_results, side_segmentation_results, results)


@supress_exceptions
def log_metrics(**kwargs):
    get_logger().log_metrics(**kwargs)


@supress_exceptions
def log(*values, sep=" ", end="\n", flush=False):
    print(*values, sep=sep, end=end, flush=flush)
    get_logger().log(*values, sep=sep, end=end, flush=flush)


class NoOpLogger:
    def reset(self): ...

    def call_start(self): ...

    def dump(self, *args, **kwargs): ...

    def log_metrics(self, **kwargs): ...

    def call_end(self): ...

    def log(self, *args, **kwargs): ...


class Logger:
    def __init__(self, logging_settings: LoggingSettings):
        log_dir = logging_settings.log_dir
        assert log_dir is not None, "log_dir must be set"

        self._log_dir = os.path.join(log_dir, datetime.now().strftime("%y%m%d_%H%M%S"))
        os.makedirs(self._log_dir, exist_ok=True)

        self._log_file = open(os.path.join(self._log_dir, "log.txt"), "wt")
        self._image_resolution = logging_settings.image_resolution
        self._image_quality = logging_settings.image_quality

        # Initialize state archiver if enabled
        self._state_archiver = None
        if logging_settings.enable_state_archiving:
            self._state_archiver = StateArchiver(
                buffer_size=logging_settings.archive_buffer_size,
            )

        self._counter = -1
        self._last_serialized_path = None
        self.reset()

    def reset(self):
        self._timestamp = None

        self._frame = None
        self._side_view_frame = None

        self._segmentation_results = None
        self._side_segmentation_results = None

        self._results = None
        self._metrics = {}

    def call_start(self):
        if self._timestamp is not None:
            # Do not raise error, just print a warning message
            print("call_start() must be called only once")

        self._timestamp = datetime.now(tz=_KST)
        self._counter += 1

    def dump(
        self,
        frame: NDArray[np.uint8],
        side_view_frame: NDArray[np.uint8],
        segmentation_results: SegmentationResults,
        side_segmentation_results: SideSegmentationResults,
        results: dict,
    ):
        self._frame = frame
        self._side_view_frame = side_view_frame
        self._segmentation_results = segmentation_results
        self._side_segmentation_results = side_segmentation_results
        self._results = results

    def log_metrics(self, **kwargs):
        self._metrics.update(kwargs)

    def _serialize(self):
        assert self._timestamp is not None, "call_start() must be called before _serialize()"

        file_name = f"{self._counter:06d}_{self._timestamp.strftime('%y%m%d_%H%M%S')}.pkl"
        file_path = os.path.join(self._log_dir, file_name)

        # Frames
        encode_options = [cv2.IMWRITE_JPEG_QUALITY, self._image_quality]

        if self._frame is not None:
            frame = cv2.cvtColor(self._frame, cv2.COLOR_RGB2BGR)
            frame_size = (frame.shape[1], frame.shape[0])  # W, H
            frame = cv2.resize(frame, self._image_resolution)
            ret, frame_array = cv2.imencode(".jpg", frame, encode_options)
            frame_bytes = frame_array.tobytes() if ret else None
        else:
            frame_bytes, frame_size = None, None

        if self._side_view_frame is not None:
            side_frame = cv2.cvtColor(self._side_view_frame, cv2.COLOR_RGB2BGR)
            side_size = (side_frame.shape[1], side_frame.shape[0])  # W, H
            side_frame = cv2.resize(side_frame, self._image_resolution)
            ret, side_array = cv2.imencode(".jpg", side_frame, encode_options)
            side_bytes = side_array.tobytes() if ret else None
        else:
            side_bytes, side_size = None, None

        # Segmentation results
        if self._segmentation_results is not None:
            segmentation_results = {
                "segmentation_masks": _compress(self._segmentation_results.segmentation_masks),
                "segmentation_contours": self._segmentation_results.segmentation_contours,
                "bounding_boxes": self._segmentation_results.bounding_boxes,
            }
        else:
            segmentation_results = None

        if self._side_segmentation_results is not None:
            side_segmentation_results = {
                "segmentation_masks": _compress(self._side_segmentation_results.segmentation_masks),
                "segmentation_contours": self._side_segmentation_results.segmentation_contours,
                "bounding_boxes": self._side_segmentation_results.bounding_boxes,
            }
        else:
            side_segmentation_results = None

        # Results
        if self._results is not None:
            results = {k: v for k, v in self._results.items() if k != "segmentation_contours"}
        else:
            results = None

        # Serialize
        serialize_dict = {
            "timestamp": self._timestamp.isoformat(),
            "frame": (frame_size, frame_bytes),
            "side_view_frame": (side_size, side_bytes),
            "segmentation_results": segmentation_results,
            "side_segmentation_results": side_segmentation_results,
            "results": results,
            "metrics": self._metrics,
        }
        with open(file_path, "wb") as f:
            pickle.dump(serialize_dict, f)

        # Store the file path for state archiver
        self._last_serialized_path = file_path

    def call_end(self):
        if self._timestamp is None:
            raise ValueError("call_start() must be called before call_end()")

        self._serialize()

        # Notify state archiver about the saved frame
        if self._state_archiver is not None and self._results is not None:
            state = self._results.get("state")
            process_state = self._results.get("process_state")
            state_index = self._results.get("state_index")

            if self._last_serialized_path is not None:
                self._state_archiver.on_frame_saved(
                    file_path=self._last_serialized_path,
                    state=state,
                    process_state=process_state,
                    state_index=state_index,
                    frame_counter=self._counter,
                )

        self.reset()

    def log(self, *values, sep=" ", end="\n", flush=False):
        prefix = f"[{datetime.now().strftime('%y%m%d_%H%M%S')}][{self._counter:06d}]"
        print(prefix, *values, sep=sep, end=end, flush=flush, file=self._log_file)

    def __del__(self):
        if hasattr(self, "_state_archiver") and self._state_archiver is not None:
            self._state_archiver.shutdown(wait=True)
        if hasattr(self, "_log_file"):
            self._log_file.close()


class LogReplayer:
    def __init__(self, log_path: str):
        self._log_path = log_path

        if zipfile.is_zipfile(log_path):
            self._is_zip = True
            self._zip_file = zipfile.ZipFile(log_path)
            file_list = self._zip_file.namelist()
        elif os.path.isdir(log_path):
            self._is_zip = False
            file_list = [os.path.join(log_path, f) for f in os.listdir(log_path)]
        else:
            raise ValueError(f"Invalid log path: {log_path}")

        file_list = sorted(
            [f for f in file_list if f.endswith(".pkl") and not f.startswith("__MACOSX/")]
        )

        self._file_map = {int(n.split("/")[-1][:6]): n for n in file_list}
        self._offset = min(self._file_map.keys())
        self._length = max(self._file_map.keys()) - self._offset + 1

        # XXX: assuming that log files are continuous
        assert len(self._file_map) == self._length, "Log files are not continuous"

    def __len__(self):
        return self._length

    @overload
    def get(self, index: int, skip_decode: Literal[False] = False) -> Log: ...

    @overload
    def get(self, index: int, skip_decode: Literal[True]) -> PartialLog: ...

    @lru_cache(maxsize=512)
    def get(self, index: int, skip_decode: bool = False) -> Log | PartialLog:
        _index = index + self._offset
        file_path = self._file_map.get(_index, None)

        if file_path is None:
            raise IndexError(f"Index out of range: {index}")

        if self._is_zip:
            with self._zip_file.open(file_path) as f:
                data = pickle.load(f)
        else:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

        if skip_decode:
            deserialize_dict = {
                "timestamp": datetime.fromisoformat(data["timestamp"]),
                "frame": None,
                "side_view_frame": None,
                "segmentation_results": None,
                "side_segmentation_results": None,
                "results": data.get("results", {}),
                "metrics": data.get("metrics", {}),
            }
            return PartialLog(**deserialize_dict)

        # XXX: assuming that all required data is present in the log file

        # Frames
        frame_size, frame_bytes = data["frame"]
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        side_view_frame_size, side_view_frame_bytes = data["side_view_frame"]
        side_view_frame = np.frombuffer(side_view_frame_bytes, dtype=np.uint8)
        side_view_frame = cv2.imdecode(side_view_frame, cv2.IMREAD_COLOR)
        side_view_frame = cv2.resize(side_view_frame, side_view_frame_size)
        side_view_frame = cv2.cvtColor(side_view_frame, cv2.COLOR_BGR2RGB)

        # Segmentation results
        _segmentation_results = data["segmentation_results"]
        segmentation_results = SegmentationResults(
            segmentation_masks=_decompress(_segmentation_results["segmentation_masks"]),
            segmentation_contours=_segmentation_results["segmentation_contours"],
            bounding_boxes=_segmentation_results["bounding_boxes"],
        )

        _side_segmentation_results = data["side_segmentation_results"]
        side_segmentation_results = SideSegmentationResults(
            segmentation_masks=_decompress(_side_segmentation_results["segmentation_masks"]),
            segmentation_contours=_side_segmentation_results["segmentation_contours"],
            bounding_boxes=_side_segmentation_results["bounding_boxes"],
        )

        # Results
        results = data.get("results", {})
        results["segmentation_contours"] = segmentation_results.segmentation_contours

        deserialize_dict = {
            "timestamp": datetime.fromisoformat(data["timestamp"]),
            "frame": frame,
            "side_view_frame": side_view_frame,
            "segmentation_results": segmentation_results,
            "side_segmentation_results": side_segmentation_results,
            "results": results,
            "metrics": data.get("metrics", {}),
        }
        return Log(**deserialize_dict)

    def __getitem__(self, index: int):
        return self.get(index)

    @cache
    def _load_timeseries(self):
        keys = set()

        raw_series = []
        for i in range(len(self)):
            log = self.get(i, skip_decode=True)
            assert log is not None, "log is None"

            r = {**log.results, **log.metrics}
            r.pop("time", None)
            raw_series.append(r)
            keys.update(r.keys())

        timeseries = {k: [] for k in keys}
        for shot in raw_series:
            for k in timeseries.keys():
                if k == "state":
                    if k in shot:
                        v = shot[k] if shot[k] is not None else "stop"
                    else:
                        v = None
                else:
                    v = shot.get(k, None)

                timeseries[k].append(v)

        return timeseries

    @cache
    def _load_state_sections(self):
        timeseries = self._load_timeseries()
        state_series = timeseries["state"]

        state_sections: dict[int, str] = {}
        state_sections_with_stop: dict[int, str] = {}
        current_state = ""
        current_state_with_stop = ""

        for i, state in enumerate(state_series):
            if state != "stop" and state != current_state:
                state_sections[i] = state
                current_state = state
            if state != current_state_with_stop:
                state_sections_with_stop[i] = state
                current_state_with_stop = state

        state_sections[len(state_series)] = "stop"
        state_sections_with_stop[len(state_series)] = "stop"
        return state_sections, state_sections_with_stop

    def get_state_range(
        self,
        start_state: State,
        start_offset: int = 0,
        stop_state: State | None = None,
        stop_offset: int = 0,
    ):
        """State에 대한 범위를 반환합니다.

        Args:
            start_state: 시작 시점을 가져올 상태
            start_offset: 시작 시점 전 오프셋
            stop_state: 종료 시점을 가져올 상태
            stop_offset: 종료 시점 후 오프셋

        Returns:
            start_idx: 시작 인덱스
            end_idx: 종료 인덱스
        """

        if stop_state is None:
            stop_state = start_state

        sections, sections_with_stop = self._load_state_sections()
        idxs, states = list(sections.keys()), list(sections.values())

        for idx, state in zip(idxs, states):
            if state == start_state:
                start_idx = idx
                break
        else:
            raise ValueError(f"Start state not found in the log: {start_state}")

        for idx, state in zip(idxs[1:], states[:-1]):
            if state == stop_state:
                end_idx = idx
                break
        else:
            raise ValueError(f"Stop state not found in the log: {stop_state}")

        start_idx = min(max(0, start_idx + start_offset), len(self))
        end_idx = min(max(0, end_idx + stop_offset), len(self))

        return start_idx, end_idx

    @overload
    def get_timeseries(
        self,
        start: int,
        stop: int,
        /,
        *,
        keys: list[str],
        exclude_stop: bool = False,
    ) -> tuple[list[int], dict[str, list]]: ...

    @overload
    def get_timeseries(
        self,
        stop: int,
        /,
        *,
        keys: list[str],
        exclude_stop: bool = False,
    ) -> tuple[list[int], dict[str, list]]: ...

    def get_timeseries(
        self,
        start: int,
        stop: int | None = None,
        /,
        *,
        keys: list[str],
        exclude_stop: bool = False,
    ) -> tuple[list[int], dict[str, list]]:
        """주어진 인덱스 범위에 대한 타임시리즈를 반환합니다.

        Args:
            start: 시작 인덱스
            stop: 종료 인덱스
            keys: 추출할 키
            exclude_stop: 종료 상태를 제외할지 여부

        Returns:
            idxs: 인덱스 리스트
            timeseries: 타임시리즈 딕셔너리
        """

        if stop is None:
            stop = len(self)
        elif stop < 0:
            stop += len(self)

        raw_timeseries = self._load_timeseries()

        state_series = raw_timeseries["state"]
        if exclude_stop:
            idxs = [i for i, s in enumerate(state_series[start:stop], start=start) if s != "stop"]
        else:
            idxs = list(range(start, stop))

        timeseries = {}

        for key in keys:
            if key == "frame":
                data = [self.get(i).frame for i in idxs]
            elif key == "side_view_frame":
                data = [self.get(i).side_view_frame for i in idxs]
            elif key == "segmentation_results":
                data = [self.get(i).segmentation_results for i in idxs]
            elif key == "side_segmentation_results":
                data = [self.get(i).side_segmentation_results for i in idxs]
            elif key in raw_timeseries:
                data = [raw_timeseries[key][i] for i in idxs]
            else:
                raise ValueError(f"Invalid key: {key}")

            timeseries[key] = data

        return idxs, timeseries

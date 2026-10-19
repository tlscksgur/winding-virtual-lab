import os
import shutil
import warnings
from typing import Callable

import pandas as pd

from SaigeToolkit.log.log_handler import LogHandler
from SaigeToolkit.util.reproducibility import get_function_args

from ..engine.monitoring import WindingMonitoringEngine
from ..modules.direction import Direction
from ..modules.shifting import ShiftMonitor
from ..modules.shift_state import WindingState, FirstWrappingState, TighteningState, ProcessState
from ..utils.cache import SegmentationCacheManager
from ..utils.label import LabelStore
from ..utils.video_decoder import VideoDecoder


VIDEO_PATH_PREFIX = "./assets/dataset/"
LABEL_PATH_PREFIX = "./assets/label/"


def _validate_video_path(main_video_path: str, side_video_path: str | None = None):
    # Main
    main_video_path = os.path.realpath(main_video_path)

    if not main_video_path.startswith(VIDEO_PATH_PREFIX):
        raise ValueError(f"Main video path is not in the expected prefix: {VIDEO_PATH_PREFIX}")
    if not os.path.exists(main_video_path):
        raise FileNotFoundError(f"Main video path does not exist: {main_video_path}")

    if side_video_path is None:
        return main_video_path, None

    # Side
    side_video_path = os.path.realpath(side_video_path)

    if not side_video_path.startswith(VIDEO_PATH_PREFIX):
        raise ValueError(f"Side video path is not in the expected prefix: {VIDEO_PATH_PREFIX}")
    if not os.path.exists(side_video_path):
        raise FileNotFoundError(f"Side video path does not exist: {side_video_path}")

    return main_video_path, side_video_path


def _validate_label_path(label_path: str):
    label_path = os.path.realpath(label_path)
    if not label_path.endswith(".json"):
        raise ValueError(f"Label must be a JSON file: {label_path}")

    if not label_path.startswith(LABEL_PATH_PREFIX):
        raise ValueError(f"Label path is not in the expected prefix: {LABEL_PATH_PREFIX}")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label path does not exist: {label_path}")

    return label_path


def _initialize_cache(
    main_video_path: str,
    checkpoint_name: str,
    inference_handler: WindingMonitoringEngine,
    side_video_path: str | None = None,
    cache_dir: str | None = None,
):
    # Main
    if cache_dir is None:
        main_cache_path = None
    else:
        main_video_rel_path = os.path.relpath(main_video_path, VIDEO_PATH_PREFIX)
        main_video_name = os.path.splitext(main_video_rel_path)[0] + ".main.pkl"
        main_cache_path = os.path.join(cache_dir, checkpoint_name, main_video_name)

    main_seg_cache_manager = SegmentationCacheManager(
        inference_handler.segmentation_engine.main_inference_handler,
        main_video_path,
        cache_path=main_cache_path,
        side=False,
    )

    if side_video_path is None:
        return main_seg_cache_manager, None

    # Side
    if cache_dir is None:
        side_cache_path = None
    else:
        side_video_rel_path = os.path.relpath(side_video_path, VIDEO_PATH_PREFIX)
        side_video_name = os.path.splitext(side_video_rel_path)[0] + ".side.pkl"
        side_cache_path = os.path.join(cache_dir, checkpoint_name, side_video_name)

    side_seg_cache_manager = SegmentationCacheManager(
        inference_handler.segmentation_engine.side_inference_handler,
        side_video_path,
        cache_path=side_cache_path,
        side=True,
    )

    # if len(main_seg_cache_manager) != len(side_seg_cache_manager):
    #     warnings.warn(
    #         "Main and side videos have different lengths: "
    #         f"{len(main_seg_cache_manager)} != {len(side_seg_cache_manager)}"
    #     )

    return main_seg_cache_manager, side_seg_cache_manager


def _get_protrusion_range_and_edges(
    label_store: LabelStore,
    start_idx_padding: int = 300,
    end_idx_padding: int = 900,
):
    # NOTE: direction label이 먼저 있고, 그 이후 protrusion label이 있다고 가정

    range_and_edges = []
    idx = -1

    for _ in range(100):  # some big number
        protrusion_start_idx, protrusion_end_idx, protrusion = label_store.next_protrusion(idx)
        if protrusion is None:
            break

        _, direction_end_idx, direction = label_store.prev_direction(protrusion_start_idx)
        if direction is None:
            continue

        # start_idx = min(max(0, protrusion_start_idx - start_idx_padding), direction_end_idx)
        start_idx = direction_end_idx - start_idx_padding
        end_idx = protrusion_end_idx + end_idx_padding

        edge_str = "left" if protrusion["left"] else "right"

        range_and_edges.append((start_idx, protrusion_start_idx, protrusion_end_idx, end_idx, edge_str))
        idx = protrusion_end_idx

    return range_and_edges


def _print_result_summary(print_fn: Callable, results: list[tuple[str, int, int, str, int]]):
    if not results:
        print_fn("No results")
        return

    df = pd.DataFrame(
        results,
        columns=["video_name", "edge", "start_idx", "end_idx", "protr_idx"],
    )

    def get_protrusion_status(row):
        if row["start_idx"] <= row["protr_idx"] <= row["end_idx"]:
            return "success"
        elif 0 <= row["protr_idx"] < row["start_idx"]:
            return "early"
        elif row["end_idx"] < row["protr_idx"]:
            return "late"
        else:
            return "failure"

    def get_protrusion_diff(row):
        if row["start_idx"] <= row["protr_idx"] <= row["end_idx"]:
            return 0
        elif 0 <= row["protr_idx"] < row["start_idx"]:
            return row["protr_idx"] - row["start_idx"]
        elif row["end_idx"] < row["protr_idx"]:
            return row["protr_idx"] - row["end_idx"]
        else:
            return -99999

    df["protr_diff"] = df.apply(get_protrusion_diff, axis=1)
    df["protr_status"] = df.apply(get_protrusion_status, axis=1)

    # Print individual results
    print_fn("")
    print_fn("INDIVIDUAL RESULTS")
    print_fn("")
    for line in df.to_string(index=False).split("\n"):
        print_fn(line)

    # Calculate overall statistics
    print_fn("")
    print_fn("OVERALL STATISTICS")
    print_fn("")

    success_count = (df["protr_status"] == "success").sum()
    early_count = (df["protr_status"] == "early").sum()
    late_count = (df["protr_status"] == "late").sum()
    failure_count = (df["protr_status"] == "failure").sum()

    print_fn(
        f"Protrusion (<-/[O]/->|X): {early_count} / [{success_count}] / {late_count} | {failure_count}"
    )

    # Calculate statistics per edge
    print_fn("")
    print_fn("STATISTICS BY EDGE")
    print_fn("")

    for edge in sorted(df["edge"].unique()):
        edge_df = df[df["edge"] == edge]
        if len(edge_df) == 0:
            continue

        print_fn(f"Edge: {edge}")

        success_count = (edge_df["protr_status"] == "success").sum()
        early_count = (edge_df["protr_status"] == "early").sum()
        late_count = (edge_df["protr_status"] == "late").sum()
        failure_count = (edge_df["protr_status"] == "failure").sum()

        print_fn(
            f"Protrusion (<-/[O]/->|X): {early_count} / [{success_count}] / {late_count} | {failure_count}"
        )
        print_fn("")


def _get_handle_state_transition(monitor: ShiftMonitor):
    def handle_state_transition(context: dict | None = None):
        label_state: str = monitor.label_state

        if label_state in {"winding", "stop"} and monitor.state.name == "winding":
            return {}
        elif label_state == "tightening" and monitor.state.name == "winding":
            monitor.reset()
            monitor.set_state(TighteningState(monitor.shift_alarm_interval))
            monitor.set_process_state(ProcessState.TIGHTENING)
            return {}
        elif label_state == "tightening" and monitor.state.name == "tightening":
            return {}
        elif label_state == "winding_first_wrap" and monitor.state.name == "tightening":
            monitor.tightening_score_calculator.reset()
            monitor.set_state(FirstWrappingState(monitor.shift_alarm_interval))
            monitor.set_process_state(ProcessState.PROTRUSION_DETECTION)
            return {"require_direction_change": True}
        else:
            return monitor.state.handle_transition(monitor, context)

    return handle_state_transition


def _winding_monitoring_engine_monitor(
    self: WindingMonitoringEngine,
    segmentation_results,
    side_segmentation_results,
    side_view_frame,
    label_state: str,
):
    self.shift_monitor.label_state = label_state  # XXX: hack

    rotation_angle = self.angle_monitor.process_frame(
        side_view_frame, side_segmentation_results
    )

    direction_monitoring_results = self.direction_monitor.process_frame(
        segmentation_results,
        self.previous_state,
        rotation_angle,
    )

    shift_monitoring_results = self.shift_monitor.process_frame(
        segmentation_results,
        direction_monitoring_results,
        side_segmentation_results,
        rotation_angle,
    )

    if shift_monitoring_results.get("require_direction_change", False):
        self.direction_monitor.change_direction()

    state = shift_monitoring_results.pop("current_state", None)
    if state is not None:
        self.previous_state = state

    return shift_monitoring_results


def _test_protrusion(
    logger: LogHandler,
    inference_handler: WindingMonitoringEngine,
    main_video_path: str,
    side_video_path: str | None,
    main_label_path: str,
    checkpoint_name: str,
    start_padding_seconds: int,
    end_padding_seconds: int,
    inference_interval: int,
    segmentation_cache_dir: str | None = None,
):
    assert side_video_path is not None, "Not implemented yet"

    # XXX: state transition hack
    shift_monitor = inference_handler.shift_monitor
    shift_monitor.handle_state_transition = _get_handle_state_transition(shift_monitor)

    main_video_path, side_video_path = _validate_video_path(main_video_path, side_video_path)
    label_path = _validate_label_path(main_label_path)

    video_name = os.path.splitext(os.path.basename(main_video_path))[0]

    if side_video_path is None:
        side_video_decoder = None
    else:
        side_video_decoder = VideoDecoder(side_video_path, check=False)

    label_store = LabelStore(label_path)
    main_seg_cache_manager, side_seg_cache_manager = _initialize_cache(
        main_video_path=main_video_path,
        side_video_path=side_video_path,
        checkpoint_name=checkpoint_name,
        inference_handler=inference_handler,
        cache_dir=segmentation_cache_dir,
    )
    video_fps = main_seg_cache_manager.fps

    range_and_directions = _get_protrusion_range_and_edges(
        label_store,
        start_idx_padding=start_padding_seconds * video_fps,
        end_idx_padding=end_padding_seconds * video_fps,
    )

    # Test
    logger.logger.info(f"[{video_name}] Start testing protrusion")

    results = []

    for start_idx, protrusion_start_idx, protrusion_end_idx, end_idx, edge_str in range_and_directions:
        start_idx = round(start_idx / inference_interval) * inference_interval
        end_idx = min(end_idx, len(main_seg_cache_manager))

        inference_handler.reset()
        protrusion_idx = -1

        for frame_idx in range(start_idx, end_idx, inference_interval):
            label = label_store.get_label(frame_idx)
            label_state = "-" if label["state"] is None else label["state"]["state"]

            if label_state == "first_swing":
                break

            side_view_frame = side_video_decoder.get_frame_at(frame_idx).data

            with inference_handler.timers["segmentation"]:
                main_seg_results = main_seg_cache_manager.get(frame_idx)
                side_seg_results = side_seg_cache_manager.get(frame_idx)

            prediction = _winding_monitoring_engine_monitor(
                inference_handler,
                main_seg_results,
                side_seg_results,
                side_view_frame,
                label_state,
            )
            process_state = prediction["process_state"]

            if inference_handler.direction_monitor.cable_moving_direction.is_unknown():
                if label["direction"] is not None:
                    if label["direction"]["left_to_right"]:
                        inference_handler.direction_monitor.cable_moving_direction = Direction.LEFT_TO_RIGHT
                    else:
                        inference_handler.direction_monitor.cable_moving_direction = Direction.RIGHT_TO_LEFT

            # Log
            if label["direction"] is None:
                direction_str = "- "
            else:
                direction_str = "->" if label["direction"]["left_to_right"] else "<-"

            if label["protrusion"] is None:
                protrusion_str = "  -  "
            else:
                protrusion_str = "left" if label["protrusion"]["left"] else "right"

            log_str = (
                f"[{video_name}][{frame_idx:07d}] "
                f"direction: {direction_str} | "
                f"protrusion: {protrusion_str} | "
                f"label_state: {label_state} | "
                f"process_state: {process_state}"
            )

            logger.logger.info(log_str)

            if process_state == "protrusion_detection_end":
                protrusion_idx = frame_idx
                break

        results.append((video_name, edge_str, protrusion_start_idx, protrusion_end_idx, protrusion_idx))

    main_seg_cache_manager.save()
    if side_seg_cache_manager is not None:
        side_seg_cache_manager.save()

    return results


def test_protrusion(
    videos: list[dict],
    segmentation: dict,
    monitoring_options: dict | None = None,
    segmentation_cache_dir: str | None = None,
    inference_interval: int = 30,
    start_padding_seconds: int = 60,
    end_padding_seconds: int = 60,
    logging: bool = True,
    log_handler: dict | None = None,
    **kwargs,
):
    args_dict = get_function_args()

    inference_handler = WindingMonitoringEngine.build(segmentation, monitoring_options)

    checkpoint_name = os.path.splitext(os.path.basename(segmentation["checkpoint_path"]))[0]

    if logging:
        if isinstance(log_handler, dict):
            logger = LogHandler(**log_handler)
        else:
            logger = LogHandler(stdout=True)

        logger.save_yaml(args_dict, "test.yaml")
    else:
        logger = None

    # Hook print builtin to redirect to logger
    if logging and logger is not None:
        import builtins
        def print_to_logger(*args, **kwargs):
            logger.logger.info(" ".join([str(arg) for arg in args]))
        builtins.print = print_to_logger

    # Iterate videos
    protrusion_results = []

    for video_info in videos:
        if "main_video" not in video_info or "main_label" not in video_info:
            continue

        _protrusion_results = _test_protrusion(
            logger,
            inference_handler,
            main_video_path=video_info["main_video"],
            side_video_path=video_info.get("side_video", None),
            main_label_path=video_info["main_label"],
            checkpoint_name=checkpoint_name,
            start_padding_seconds=start_padding_seconds,
            end_padding_seconds=end_padding_seconds,
            inference_interval=inference_interval,
            segmentation_cache_dir=segmentation_cache_dir,
        )
        protrusion_results.extend(_protrusion_results)

    if logger is not None:
        print_fn = lambda msg: logger.logger.info(msg)
    else:
        print_fn = print

    _print_result_summary(print_fn, protrusion_results)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/protrusion.yml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    assert isinstance(config, dict), "config must be a dictionary"

    test_protrusion(**config)

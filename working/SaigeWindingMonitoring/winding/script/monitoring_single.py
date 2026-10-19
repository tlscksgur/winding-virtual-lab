import os
import shutil
import warnings

from PIL import Image

from SaigeToolkit.log.log_handler import LogHandler
from SaigeToolkit.util.reproducibility import get_function_args

from ..engine.config import CommonSettings
from ..engine.monitoring import WindingMonitoringEngine
from ..utils.cache import SegmentationCacheManager
from ..utils.video_decoder import VideoDecoder
from ..utils.visualization import ResultVisualizer
from ..utils.visualization_tools import image_dir_to_mp4


VIDEO_PATH_PREFIX = "./assets/dataset/"
# CACHE_PATH_PREFIX = "./assets/segmentation_cache/"


def time_str_to_seconds(t: str) -> int:
    # "MM:SS" or "HH:MM:SS" → 초로 변환
    parts = list(map(int, t.strip().split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Use MM:SS or HH:MM:SS")


def _validate_path(main_video_path: str, side_video_path: str):
    main_video_path = os.path.realpath(main_video_path)

    if not main_video_path.startswith(VIDEO_PATH_PREFIX):
        raise ValueError(f"Main video path is not in the expected prefix: {VIDEO_PATH_PREFIX}")
    if not os.path.exists(main_video_path):
        raise FileNotFoundError(f"Main video path does not exist: {main_video_path}")

    side_video_path = os.path.realpath(side_video_path)

    if not side_video_path.startswith(VIDEO_PATH_PREFIX):
        raise ValueError(f"Side video path is not in the expected prefix: {VIDEO_PATH_PREFIX}")
    if not os.path.exists(side_video_path):
        raise FileNotFoundError(f"Side video path does not exist: {side_video_path}")

    return main_video_path, side_video_path


def _initialize_cache(
    main_video_path: str,
    side_video_path: str,
    checkpoint_name: str,
    inference_handler: WindingMonitoringEngine,
    video_fps: int,
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

    if main_seg_cache_manager.fps != video_fps:
        raise ValueError(
            f"Main segmentation cache have different fps: {main_seg_cache_manager.fps} != {video_fps}"
        )

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

    if side_seg_cache_manager.fps != video_fps:
        raise ValueError(
            f"Side segmentation cache have different fps: {side_seg_cache_manager.fps} != {video_fps}"
        )

    if len(main_seg_cache_manager) != len(side_seg_cache_manager):
        warnings.warn(
            "Main and side videos have different lengths: "
            f"{len(main_seg_cache_manager)} != {len(side_seg_cache_manager)}"
        )

    return main_seg_cache_manager, side_seg_cache_manager


def _calculate_target_frames(
    max_frame_idx: int,
    video_fps: int,
    frame_range: list[int] | None = None,
    time_range: list[str] | None = None,
    inference_interval: int = 30,
):
    if frame_range is not None and time_range is not None:
        raise ValueError("Both frame_range and time_range cannot be provided")

    elif frame_range is None and time_range is None:
        start_idx, end_idx = 0, max_frame_idx

    elif frame_range is not None:
        if len(frame_range) != 2:
            raise ValueError(f"Frame range must be a list of two integers: {frame_range}")

        start_idx, end_idx = frame_range

    elif time_range is not None:
        if len(time_range) != 2:
            raise ValueError(f"Time range must be a list of two strings: {time_range}")

        start_time, end_time = time_range
        start_idx = None if start_time is None else time_str_to_seconds(start_time) * video_fps
        end_idx = None if end_time is None else time_str_to_seconds(end_time) * video_fps

    else:
        raise RuntimeError  # for typing. logically should not reach here

    start_idx = 0 if start_idx is None else max(0, start_idx)
    end_idx = max_frame_idx if end_idx is None else min(max_frame_idx, end_idx + 1)

    if start_idx >= end_idx:
        raise ValueError(f"Start index must be less than end index: {start_idx} >= {end_idx}")

    if len(range(start_idx, end_idx, inference_interval)) == 0:
        raise ValueError(f"No frames to process in the given time range: {start_idx} - {end_idx}")

    return start_idx, end_idx


def monitoring(
    main_video_path: str,
    side_video_path: str,
    segmentation: dict,
    monitoring_options: dict | None = None,
    visualizer: dict | None = None,
    frame_range: list[int] | None = None,
    time_range: list[str] | None = None,
    segmentation_cache_dir: str | None = None,
    inference_interval: int = 30,
    save_image: bool = False,
    save_video: bool = False,
    logging: bool = True,
    log_handler: dict | None = None,
    log_interval: int = 20,
    **kwargs,
):
    args_dict = get_function_args()

    # Core components
    main_video_path, side_video_path = _validate_path(main_video_path, side_video_path)

    video_name = os.path.splitext(os.path.basename(main_video_path))[0]
    image_save_dir = os.path.join("frame", video_name)

    inference_handler = WindingMonitoringEngine.build(segmentation, monitoring_options)
    main_video_decoder = VideoDecoder(main_video_path, check=False)
    side_video_decoder = VideoDecoder(side_video_path, check=False)
    result_visualizer = ResultVisualizer(**(visualizer or {}))

    if main_video_decoder.fps != side_video_decoder.fps:
        raise ValueError(
            "Main video and side video have different fps: "
            f"{main_video_decoder.fps} != {side_video_decoder.fps}"
        )

    if logging:
        if isinstance(log_handler, dict):
            logger = LogHandler(**log_handler)
        else:
            logger = LogHandler(stdout=True)

        logger.save_yaml(args_dict, "test.yaml")
    else:
        logger = None

    # Cache
    main_seg_cache_manager, side_seg_cache_manager = _initialize_cache(
        main_video_path=main_video_path,
        side_video_path=side_video_path,
        checkpoint_name=os.path.splitext(os.path.basename(segmentation["checkpoint_path"]))[0],
        inference_handler=inference_handler,
        video_fps=side_video_decoder.fps,
        cache_dir=segmentation_cache_dir,
    )

    if len(main_video_decoder) != len(main_seg_cache_manager):
        raise ValueError(
            "Main video and main segmentation cache have different lengths: "
            f"{len(main_video_decoder)} != {len(main_seg_cache_manager)}"
        )

    if len(side_video_decoder) != len(side_seg_cache_manager):
        raise ValueError(
            "Side video and side segmentation cache have different lengths: "
            f"{len(side_video_decoder)} != {len(side_seg_cache_manager)}"
        )

    # Target frames
    max_frame_idx = min(len(main_video_decoder), len(side_video_decoder))

    start_idx, end_idx = _calculate_target_frames(
        video_fps=main_video_decoder.fps,
        max_frame_idx=max_frame_idx,
        frame_range=frame_range,
        time_range=time_range,
        inference_interval=inference_interval,
    )

    # Inference
    for frame_idx in range(start_idx, end_idx, inference_interval):
        main_view_frame = main_video_decoder.get_frame_at(frame_idx).data
        side_view_frame = side_video_decoder.get_frame_at(frame_idx).data

        # Skip inference segmentation inference if the frame is already in the cache
        with inference_handler.timers["segmentation"]:
            main_seg_results = main_seg_cache_manager.get(frame_idx)
            side_seg_results = side_seg_cache_manager.get(frame_idx)

        # Directly feed the segmentation results to the monitoring engine
        prediction = inference_handler.monitor(main_seg_results, side_seg_results, side_view_frame)
        assert (
            prediction["state_index"] != CommonSettings.state_index["error"]
        ), "An error occurred during inference."

        if (save_image or save_video) and logger is not None:
            visualization = result_visualizer(
                main_view_frame.copy(),
                side_view_frame.copy(),
                prediction=prediction,
                side_segmentation_results=side_seg_results,
                direction_monitor=inference_handler.direction_monitor,
                shift_monitor=inference_handler.shift_monitor,
            )

            logger.save_image(
                image=Image.fromarray(visualization),
                file_path=os.path.join(image_save_dir, f"{frame_idx:07d}.jpg"),
            )

        if frame_idx % log_interval == 0 and logger is not None:
            time_result = prediction["time"]

            log = f"[Monitoring: {frame_idx:5}/{max_frame_idx:5}]"
            log += f" | segmentation: {time_result['segmentation_time']:8.4f} ms"
            log += f" | direction: {time_result['direction_monitoring_time']:8.4f} ms"
            log += f" | anomaly: {time_result['anomaly_monitoring_time']:8.4f} ms"
            log += f" | shift: {time_result['shift_monitoring_time']:8.4f} ms"
            log += f" | total: {time_result['total_time']:8.4f} ms"
            log += f" | FPS: {1 / time_result['total_time'] * 1000:6.2f}"
            log += f" | current_state: {inference_handler.previous_state}"
            log += f" | process_state: {prediction['process_state']}"
            if prediction["rotation_angle"] is not None:
                log += f" | angle: {prediction['rotation_angle']:6.2f} / {prediction['reference_angle']}"

            logger.logger.info(log)

    main_seg_cache_manager.save()
    side_seg_cache_manager.save()

    if save_video and logger is not None and logger.logdir is not None:
        image_dir = os.path.join(logger.logdir, image_save_dir)
        output_video_path = os.path.join(logger.logdir, f"{video_name}.mp4")
        image_dir_to_mp4(image_dir, output_video_path, fps=10)

        if not save_image:
            shutil.rmtree(image_dir)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/monitoring_single.yml")
    parser.add_argument("-v", "--video_number", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    assert isinstance(config, dict), "config must be a dictionary"

    video_number = config.pop("video_number", 0)
    videos = config.pop("videos")

    if args.video_number is not None:
        video_number = args.video_number

    video_info = videos[video_number]

    config["main_video_path"] = video_info["main_video"]
    config["side_video_path"] = video_info["side_video"]

    if "frame_range" in video_info:
        frame_range = video_info["frame_range"]
        if isinstance(frame_range, list) or frame_range is None:
            config["frame_range"] = frame_range
        else:
            config["frame_range"] = [frame_range, None]

    if "time_range" in video_info:
        time_range = video_info["time_range"]
        if isinstance(time_range, list) or time_range is None:
            config["time_range"] = time_range
        else:
            config["time_range"] = [time_range, None]

    monitoring(**config)

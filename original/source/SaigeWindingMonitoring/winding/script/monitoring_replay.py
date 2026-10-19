import os
import shutil
import warnings

from PIL import Image

from SaigeToolkit.log.log_handler import LogHandler
from SaigeToolkit.util.reproducibility import get_function_args

from ..engine.config import CommonSettings, MonitoringConfig
from ..engine.monitoring import WindingMonitoringEngine
from ..modules import AngleMonitor, DirectionMonitor, AnomalyMonitor, ShiftMonitor, SpongeMonitor
from ..modules.logging import LogReplayer
from ..utils.visualization import ResultVisualizer
from ..utils.visualization_tools import image_dir_to_mp4


VIDEO_PATH_PREFIX = "./assets/dataset/"


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


def _calculate_target_frames(
    max_frame_idx: int,
    frame_range: list[int] | None = None,
    time_range: list[str] | None = None,
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
        raise NotImplementedError("Time range is not supported for replay")

        if len(time_range) != 2:
            raise ValueError(f"Time range must be a list of two strings: {time_range}")

        start_time, end_time = time_range
        start_idx = None if start_time is None else time_str_to_seconds(start_time)
        end_idx = None if end_time is None else time_str_to_seconds(end_time)

    else:
        raise RuntimeError  # for typing. logically should not reach here

    start_idx = 0 if start_idx is None else max(0, start_idx)
    end_idx = max_frame_idx if end_idx is None else min(max_frame_idx, end_idx + 1)

    if start_idx >= end_idx:
        raise ValueError(f"Start index must be less than end index: {start_idx} >= {end_idx}")

    if start_idx >= end_idx:
        raise ValueError(f"No frames to process in the given time range: {start_idx} - {end_idx}")

    return start_idx, end_idx


def _monitoring(
    replay_path: str,
    monitoring_options: dict | None = None,
    visualizer: dict | None = None,
    frame_range: list[int] | None = None,
    time_range: list[str] | None = None,
    save_image: bool = False,
    save_video: bool = False,
    logging: bool = True,
    log_handler: dict | None = None,
    log_interval: int = 1,
    **kwargs,
):
    args_dict = get_function_args()

    # Build engine
    monitoring_config = MonitoringConfig()
    inference_handler = WindingMonitoringEngine(
        monitoring_config=monitoring_config,
        segmentation_engine=None,
        angle_monitor=AngleMonitor(monitoring_config),
        direction_monitor=DirectionMonitor(monitoring_config),
        sponge_monitor=SpongeMonitor(monitoring_config),
        anomaly_monitor=AnomalyMonitor(monitoring_config),
        shift_monitor=ShiftMonitor(monitoring_config),
    )

    if monitoring_options is not None:
        for key, value in monitoring_options.items():
            inference_handler.set_monitoring_option(key, value)

    # Core components
    video_name = os.path.splitext(os.path.basename(replay_path))[0]
    image_save_dir = os.path.join("frame", video_name)

    replayer = LogReplayer(replay_path)
    result_visualizer = ResultVisualizer(**(visualizer or {}))

    if logging:
        if isinstance(log_handler, dict):
            logger = LogHandler(**log_handler)
        else:
            logger = LogHandler(stdout=True)

        logger.save_yaml(args_dict, "test.yaml")
    else:
        logger = None

    max_frame_idx = len(replayer)
    start_idx, end_idx = _calculate_target_frames(
        max_frame_idx=max_frame_idx,
        frame_range=frame_range,
        time_range=time_range,
    )

    # Inference
    for frame_idx in range(start_idx, end_idx):
        with inference_handler.timers["segmentation"]:  # Just for avoiding timer error
            data = replayer.get(frame_idx)

        if data is None:
            logger.logger.warning(f"No data found for frame {frame_idx}")
            continue

        main_seg_results = data.segmentation_results
        side_seg_results = data.side_segmentation_results
        main_view_frame = data.frame
        side_view_frame = data.side_view_frame

        # Directly feed the segmentation results to the monitoring engine
        prediction = inference_handler.monitor(main_seg_results, side_seg_results, side_view_frame)
        assert (
            prediction["state_index"] != CommonSettings.state_index["error"]
        ), "An error occurred during inference."

        if (save_image or save_video) and logger is not None:
            visualization = result_visualizer(
                frame=main_view_frame.copy(),
                side_view_frame=side_view_frame,
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
            log = f"[Monitoring: {frame_idx:5}/{max_frame_idx:5}]"
            log += f" | state: [log: {data.metrics['state']}"
            log += f" / pred: {inference_handler.previous_state}]"
            log += f" | process_state: [log: {data.results['process_state']}"
            log += f" / pred: {prediction['process_state']}]"
            if prediction["rotation_angle"] is not None:
                log += f" | angle: {prediction['rotation_angle']:6.2f} / {prediction['reference_angle']}"

            logger.logger.info(log)

    if save_video and logger is not None and logger.logdir is not None:
        image_dir = os.path.join(logger.logdir, image_save_dir)
        output_video_path = os.path.join(logger.logdir, f"{video_name}.mp4")
        image_dir_to_mp4(image_dir, output_video_path, fps=10)

        if not save_image:
            shutil.rmtree(image_dir)


def monitoring(
    replays: list[dict],
    replay_number: int = 0,
    monitoring_options: dict | None = None,
    visualizer: dict | None = None,
    frame_range: list[int] | None = None,
    time_range: list[str] | None = None,
    save_image: bool = False,
    save_video: bool = False,
    logging: bool = True,
    log_handler: dict | None = None,
    log_interval: int = 1,
    **kwargs,
):
    replay_info = replays[replay_number]

    replay_path = replay_info["path"]

    if "frame_range" in replay_info:
        _frame_range = replay_info["frame_range"]
        if isinstance(_frame_range, list) or _frame_range is None:
            frame_range = _frame_range
        else:
            frame_range = [_frame_range, None]

    if "time_range" in replay_info:
        _time_range = replay_info["time_range"]
        if isinstance(_time_range, list) or _time_range is None:
            time_range = _time_range
        else:
            time_range = [_time_range, None]

    _monitoring(
        replay_path=replay_path,
        monitoring_options=monitoring_options,
        visualizer=visualizer,
        frame_range=frame_range,
        time_range=time_range,
        save_image=save_image,
        save_video=save_video,
        logging=logging,
        log_handler=log_handler,
        log_interval=log_interval,
    )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/monitoring_replay.yml")
    parser.add_argument("-r", "--replay_number", type=int, default=0)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    assert isinstance(config, dict), "config must be a dictionary"

    monitoring(**config)

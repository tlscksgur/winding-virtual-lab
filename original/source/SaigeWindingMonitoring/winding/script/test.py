import os
import shutil
from typing import Optional

import cv2
from PIL import Image
from SaigeToolkit.log.log_handler import LogHandler
from SaigeToolkit.util.reproducibility import get_function_args

from ..engine.monitoring import WindingMonitoringEngine
from ..engine.config import CommonSettings
from ..utils.visualization_tools import image_dir_to_mp4
from ..utils.visualization import ResultVisualizer

VIDEO_EXTENSIONS = (".mp4", ".MOV", ".avi")


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


def get_video_path_list(video_file_paths: dict) -> tuple:

    def get_video_path_from_dir(path) -> list:
        video_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(VIDEO_EXTENSIONS):
                    video_paths.append(os.path.join(root, file))
        return video_paths

    def process_path_and_range(path_dict, range_value):
        """경로와 범위를 처리하는 공통 함수 - dict 형태만 지원"""
        if isinstance(range_value, (str, int)):
            range_value = [range_value, None]

        # path_dict는 딕셔너리여야 함 (main, side 키를 가진 경우)
        if not isinstance(path_dict, dict):
            raise ValueError(
                f"Path must be a dictionary with 'main' and/or 'side' keys, got: {type(path_dict)}"
            )

        processed_path_dict = {}

        for key, path in path_dict.items():
            if os.path.isdir(path):
                _video_paths = get_video_path_from_dir(path)
                if len(_video_paths) > 1:
                    # 디렉토리에 여러 비디오가 있는 경우, 첫 번째만 사용
                    processed_path_dict[key] = _video_paths[0]
                elif len(_video_paths) == 1:
                    processed_path_dict[key] = _video_paths[0]
                else:
                    raise ValueError(f"No video files found in directory: {path}")
            elif os.path.isfile(path) and path.endswith(VIDEO_EXTENSIONS):
                processed_path_dict[key] = path
            else:
                raise ValueError(f"Invalid video file path: {path}")

        return processed_path_dict, range_value

    video_path_list = []
    time_range_list = []

    paths = video_file_paths.get("path")
    time_ranges = video_file_paths.get("time_range", None)
    frame_ranges = video_file_paths.get("frame_range", None)

    # 범위가 있는 경우 (time_range 또는 frame_range)
    if time_ranges is not None or frame_ranges is not None:
        ranges = time_ranges if time_ranges is not None else frame_ranges
        range_type = "time_range" if time_ranges is not None else "frame_range"

        if len(paths) != len(ranges):
            raise ValueError(f"The lengths of 'path' and '{range_type}' lists must be the same.")

        for path, range_value in zip(paths, ranges):
            paths_to_add, ranges_to_add = process_path_and_range(path, range_value)
            video_path_list.append(paths_to_add)
            time_range_list.append(ranges_to_add)

    # 범위가 없는 경우
    else:
        for path in paths:
            paths_to_add, ranges_to_add = process_path_and_range(path, None)
            video_path_list.append(paths_to_add)
            time_range_list.append(ranges_to_add)

    return video_path_list, time_range_list


def test_monitoring(
    segmentation: dict,
    video_file_paths: dict,
    monitoring_options: Optional[dict] = None,
    log_handler: dict | LogHandler | None = None,
    inference_interval: int = 100,
    save_image: bool = False,
    save_video: bool = False,
    logging: bool = True,
    visualizer: dict = {},
    log_interval: int = 20,
    segmentation_cache_dir: str = None,
    **kwargs,
):

    args_dict = get_function_args()

    if logging:
        if log_handler is None:
            log_handler = LogHandler(stdout=True)
        elif isinstance(log_handler, dict):
            log_handler = LogHandler(**log_handler)
        log_handler.save_yaml(args_dict, "test.yaml")

    if save_image or save_video:
        frame_path = os.path.join(log_handler.logdir, "frame")
        os.makedirs(frame_path, exist_ok=True)

    inference_handler = WindingMonitoringEngine.build(
        segmentation=segmentation,
        monitoring_options=monitoring_options,
    )

    input_video_paths, time_range_list = get_video_path_list(video_file_paths)

    for video_path_dict, time_range in zip(input_video_paths, time_range_list):
        # 딕셔너리에서 main과 side 경로 추출
        input_video_path = video_path_dict.get("main")
        input_side_video_path = video_path_dict.get("side")

        print(
            f"Processing Video - main: {input_video_path}, side: {input_side_video_path}, time_range: {time_range}"
        )
        input_video_obj = cv2.VideoCapture(input_video_path)
        fps = int(round(input_video_obj.get(cv2.CAP_PROP_FPS)))

        input_side_video_obj = cv2.VideoCapture(input_side_video_path)
        side_fps = int(round(input_side_video_obj.get(cv2.CAP_PROP_FPS)))
        assert fps == side_fps, "Main and side video FPS must match."

        total_frame_count = int(input_video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        if time_range is None:
            frame_count = total_frame_count
            start_frame_index = 0
        else:
            if isinstance(time_range[0], str) and isinstance(time_range[1], str | None):
                start_time_str = time_range[0]
                end_time_str = time_range[1]
                start_frame_index = time_str_to_seconds(start_time_str) * fps
                frame_count = (
                    time_str_to_seconds(end_time_str) * fps if end_time_str else total_frame_count
                )

            elif isinstance(time_range[0], int) and isinstance(time_range[1], int | None):
                start_frame_index = time_range[0]
                frame_count = time_range[1] if time_range[1] is not None else total_frame_count

        input_video_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        input_side_video_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        num_target_frames = int((frame_count - start_frame_index) / inference_interval)
        assert num_target_frames > 0, "No frames to process in the given time range."

        result_visualizer = ResultVisualizer(**visualizer)

        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        frame_save_path = os.path.join("frame", video_name)

        current_frame = start_frame_index

        for i in range(num_target_frames):
            if inference_interval is None or inference_interval == 1:
                frame_idx = start_frame_index + i
            else:
                frame_idx = start_frame_index + i * inference_interval

            # 순차적으로 프레임 건너뛰기
            while current_frame < frame_idx:
                main_success = input_video_obj.grab()
                side_success = input_side_video_obj.grab()
                if not main_success or not side_success:
                    break
                current_frame += 1

            success, image = input_video_obj.read()
            if not success:
                break
            success, side_image = input_side_video_obj.read()
            if not success:
                break
            current_frame += 1

            input_image = image.copy()
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_side_image = side_image.copy()
            input_side_image = cv2.cvtColor(input_side_image, cv2.COLOR_BGR2RGB)

            if segmentation_cache_dir is not None:
                main_segmentation_cache_path = os.path.join(
                    segmentation_cache_dir, video_name, f"{frame_idx:06d}.pkl"
                )
                side_segmentation_cache_path = os.path.join(
                    segmentation_cache_dir, video_name + "_side", f"{frame_idx:06d}.pkl"
                )
            else:
                main_segmentation_cache_path = None
                side_segmentation_cache_path = None

            prediction, side_segmentation_results = inference_handler.process_frame(
                frame=input_image,
                side_view_frame=input_side_image,
                segmentation_cache_path=main_segmentation_cache_path,
                side_segmentation_cache_path=side_segmentation_cache_path,
            )
            assert (
                prediction["state_index"] != CommonSettings.state_index["error"]
            ), "An error occurred during inference."

            visualization = result_visualizer(
                frame=image.copy(),
                side_view_frame=side_image.copy(),
                prediction=prediction,
                side_segmentation_results=side_segmentation_results,
                direction_monitor=inference_handler.direction_monitor,
                shift_monitor=inference_handler.shift_monitor,
            )

            if save_image or save_video:
                log_handler.save_image(
                    image=Image.fromarray(visualization),
                    file_path=os.path.join(
                        frame_save_path, f"{os.path.basename(input_video_path)}_{frame_idx:07d}.jpg"
                    ),
                )

            if frame_idx % log_interval == 0:
                time_result = prediction["time"]

                log = f"[Monitoring: {frame_idx:5}/{frame_count:5}]"
                log += f" | segmentation: {time_result['segmentation_time']:8.4f} ms"
                log += f" | direction: {time_result['direction_monitoring_time']:8.4f} ms"
                log += f" | anomaly: {time_result['anomaly_monitoring_time']:8.4f} ms"
                log += f" | shift: {time_result['shift_monitoring_time']:8.4f} ms"
                log += f" | total: {time_result['total_time']:8.4f} ms"
                log += f" | FPS: {1 / time_result['total_time'] * 1000:6.2f}"
                log_handler.logger.info(log)

        inference_handler.reset()
        input_video_obj.release()
        input_side_video_obj.release()

        if save_video:
            output_video_path = os.path.join(log_handler.logdir, os.path.basename(input_video_path))
            image_dir_to_mp4(
                os.path.join(log_handler.logdir, frame_save_path), output_video_path, fps=fps
            )

            if not save_image:
                # Remove the frame directory if only video is saved
                shutil.rmtree(os.path.join(log_handler.logdir, frame_save_path))

    print(f"Result saved at {log_handler.logdir}")

    return log_handler.logdir if log_handler is not None else None

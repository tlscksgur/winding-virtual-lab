# 사이드 뷰 영상에 대해, 회전 각도 추정하는 스크립트

import os
import shutil

import cv2
from PIL import Image
from SaigeToolkit.log.log_handler import LogHandler
from SaigeToolkit.util.reproducibility import get_function_args
from SaigeToolkit.util.timer import Timer

from ..modules.segmentation import SegmentationEngine
from ..modules.line_tracker import MultiLineTracker
from ..engine.config import RotationSettings
from ..utils.visualization_tools import image_dir_to_mp4
from ..utils.visualization import SideViewResultVisualizer

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

    def process_range(range_value):
        if range_value is None:
            return None
        if isinstance(range_value, (str, int)):
            return [range_value, None]
        return range_value

    video_path_list = []
    time_range_list = []

    paths = video_file_paths.get("path", [])
    time_ranges = video_file_paths.get("time_range", None)
    frame_ranges = video_file_paths.get("frame_range", None)

    # If ranges provided
    if time_ranges is not None or frame_ranges is not None:
        ranges = time_ranges if time_ranges is not None else frame_ranges
        if len(paths) != len(ranges):
            raise ValueError(f"The lengths of 'path' and range lists must be the same.")

        for path, range_value in zip(paths, ranges):
            if os.path.isdir(path):
                _video_paths = get_video_path_from_dir(path)
                if len(_video_paths) > 0:
                    video_path = _video_paths[0]
                else:
                    raise ValueError(f"No video files found in directory: {path}")
            elif os.path.isfile(path) and path.endswith(VIDEO_EXTENSIONS):
                video_path = path
            else:
                raise ValueError(f"Invalid video file path: {path}")

            processed_range = process_range(range_value)
            video_path_list.append(video_path)
            time_range_list.append(processed_range)

    # No ranges
    else:
        for path in paths:
            if os.path.isdir(path):
                _video_paths = get_video_path_from_dir(path)
                if len(_video_paths) > 0:
                    video_path = _video_paths[0]
                else:
                    raise ValueError(f"No video files found in directory: {path}")
            elif os.path.isfile(path) and path.endswith(VIDEO_EXTENSIONS):
                video_path = path
            else:
                raise ValueError(f"Invalid video file path: {path}")

            video_path_list.append(video_path)
            time_range_list.append(None)

    return video_path_list, time_range_list


def add_time_to_result(timers, predictions: dict) -> dict:
    inference_time_dict = {f"{k}_time": v.time_stack[-1] for k, v in timers.items()}
    predictions["time"] = {k: round(v, 2) for k, v in inference_time_dict.items()}
    predictions["time"]["total_time"] = sum(inference_time_dict.values())

    return predictions


def test_side_view(
    segmentation: dict,
    video_file_paths: dict,
    log_handler: dict | LogHandler | None = None,
    inference_interval: int = 100,
    save_image: bool = False,
    save_video: bool = False,
    logging: bool = True,
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

    # Build SegmentationEngine
    segmentation_engine = SegmentationEngine.build(**segmentation)

    # Build LineTracker
    line_tracker = MultiLineTracker(RotationSettings)

    input_video_paths, time_range_list = get_video_path_list(video_file_paths)

    for video_path, time_range in zip(input_video_paths, time_range_list):
        # INIT
        predictions = {}
        previous_frame = None
        timers: dict[str, Timer] = {
            "segmentation": Timer(is_enabled=True),
            "line_tracking": Timer(is_enabled=True),
        }

        # 딕셔너리에서 main과 side 경로 추출
        input_video_path = video_path

        print(f"Processing Video - side: {input_video_path}, time_range: {time_range}")
        input_video_obj = cv2.VideoCapture(input_video_path)
        fps = int(round(input_video_obj.get(cv2.CAP_PROP_FPS)))

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
        num_target_frames = int((frame_count - start_frame_index) / inference_interval)
        assert num_target_frames > 0, "No frames to process in the given time range."

        result_visualizer = SideViewResultVisualizer(plot_graph=True)

        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        frame_save_path = os.path.join("frame", video_name)

        current_frame = start_frame_index
        for i in range(num_target_frames):
            frame_idx = start_frame_index + i * inference_interval

            # 순차적으로 프레임 건너뛰기
            while current_frame < frame_idx:
                success = input_video_obj.grab()
                if not success:
                    break
                current_frame += 1

            success, image = input_video_obj.read()
            current_frame += 1

            input_image = image.copy()
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            if segmentation_cache_dir is not None:
                segmentation_cache_path = os.path.join(
                    segmentation_cache_dir, video_name, f"{frame_idx:06d}.pkl"
                )
            else:
                segmentation_cache_path = None

            with timers["segmentation"]:
                side_segmentation_results = segmentation_engine.inference_side_frame(
                    frame=input_image, segmentation_cache_path=segmentation_cache_path
                )

            # Line tracking
            with timers["line_tracking"]:
                line_tracker.update(input_image, side_segmentation_results=side_segmentation_results)
                line_rotation_angle = line_tracker.get_result()

            # Skip first frame for visualization and time recording
            if previous_frame is None:
                previous_frame = input_image
                continue

            predictions = add_time_to_result(timers, predictions)

            visualization = result_visualizer(
                image=image,
                side_segmentation_results=side_segmentation_results,
                line_tracker=line_tracker,
                rotation_angle=line_rotation_angle,
            )

            previous_frame = input_image

            if save_image or save_video:
                log_handler.save_image(
                    image=Image.fromarray(visualization),
                    file_path=os.path.join(
                        frame_save_path, f"{os.path.basename(input_video_path)}_{frame_idx:07d}.jpg"
                    ),
                )

            if frame_idx % log_interval == 0:
                time_result = predictions["time"]
                log = f"[Monitoring: {frame_idx:5}/{frame_count:5}]"
                log += f" | segmentation: {time_result['segmentation_time']:8.4f} ms"
                if line_tracker is not None and line_rotation_angle is not None:
                    log += f" | line_tracking: {time_result['line_tracking_time']:8.4f} ms"
                    log += f" | line_angle: {line_rotation_angle:7.2f}°"
                log += f" | total: {time_result['total_time']:8.4f} ms"
                log += f" | FPS: {1 / time_result['total_time'] * 1000:6.2f}"
                log_handler.logger.info(log)

        input_video_obj.release()
        line_tracker.reset()

        if save_video:
            output_video_path = os.path.join(log_handler.logdir, os.path.basename(input_video_path))
            # Adjust fps based on inference_interval to maintain proper playback speed
            output_fps = max(1, int(fps / inference_interval))
            print(
                f"Creating output video with fps={output_fps} (original fps={fps}, inference_interval={inference_interval})"
            )
            image_dir_to_mp4(
                os.path.join(log_handler.logdir, frame_save_path), output_video_path, fps=output_fps
            )
            print(f"Output video saved: {output_video_path}")

            if not save_image:
                # Remove the frame directory if only video is saved
                shutil.rmtree(os.path.join(log_handler.logdir, frame_save_path))

    print(f"Result saved at {log_handler.logdir}")

    return log_handler.logdir if log_handler is not None else None

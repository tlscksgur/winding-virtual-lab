import os
import json
import datetime

import cv2
import tqdm

from ..engine.monitoring import WindingMonitoringEngine
from ..utils.metric import compute_metrics


def benchmark(
    video_number: int,
    dataset_dir: str,
    segmentation: dict = None,
    monitoring_options: dict = None,
    save_dir: str = None,
    segmentation_cache_dir: str = None,
    benchmark_video_range: list[float] = [0.3, 0.7],
    **kwargs,
):
    inference_handler = WindingMonitoringEngine.build(
        segmentation=segmentation,
        monitoring_options=monitoring_options,
    )
    video_path = os.path.join(dataset_dir, "videos", f"encoded_drum-{video_number:02d}_summary.mp4")
    label_path = video_path.replace("videos", "labels").replace(".mp4", ".json")

    with open(label_path, "r") as f:
        labels = json.load(f)

    labels, predictions = _get_predictions(
        labels,
        inference_handler,
        video_path,
        segmentation_cache_dir,
        benchmark_video_range,
    )

    filtered_label = []
    filtered_prediction = []
    for list_, filtered_list in [
        (labels["shiftings"], filtered_label),
        (predictions["shiftings"], filtered_prediction),
    ]:
        for item in list_:
            filtered_list.append(item)

    metrics = compute_metrics(filtered_label, filtered_prediction, iou_threshold=0.1)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    predictions.update(metrics)
    timestamp = predictions["timestamp"]
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/encoded_drum-{video_number:02d}_summary_{timestamp}.json", "w") as f:
        json.dump(predictions, f, indent=4)


def _get_predictions(
    labels,
    inference_handler,
    video_path,
    segmentation_cache_dir=None,
    benchmark_video_range=None,
):
    input_video_obj = cv2.VideoCapture(video_path)
    start_frame_idx = 0
    end_frame_idx = int(input_video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

    if benchmark_video_range is not None:
        progress_range_start, progress_range_end = benchmark_video_range
        start_frame_idx = int(end_frame_idx * progress_range_start)
        end_frame_idx = int(end_frame_idx * progress_range_end)
        input_video_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    shifting_start_threshold = inference_handler.shift_monitor.shifting_start_threshold
    shifting_end_threshold = inference_handler.shift_monitor.shifting_end_threshold

    shifting_is_in_progress = False
    shifting_start_idx = None
    shifting_start_progress = None
    shiftings = []
    scores = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for frame_idx in tqdm.tqdm(range(start_frame_idx, end_frame_idx)):
        success, input_image = input_video_obj.read()
        if not success:
            break

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        segmentation_cache_path = None
        if segmentation_cache_dir is not None:
            segmentation_cache_path = os.path.join(
                segmentation_cache_dir, video_name, f"{frame_idx:06d}.pkl"
            )
        prediction = inference_handler.process_frame(input_image, segmentation_cache_path)
        shifting_score = prediction["drum_shifting_score"]
        direction = prediction["drum_moving_direction"]
        progress_rate = prediction["progress_rate"]
        state_index = prediction["state_index"]

        scores.append(shifting_score)
        if shifting_score > shifting_start_threshold and state_index == 2:
            if not shifting_is_in_progress:
                shifting_is_in_progress = True
                shifting_start_idx = frame_idx
                shifting_start_progress = progress_rate

        elif shifting_score < shifting_end_threshold and state_index != 2:
            if shifting_is_in_progress:
                shifting_is_in_progress = False
                shiftings.append(
                    {
                        "start_progress_rate": shifting_start_progress,
                        "end_progress_rate": prediction["progress_rate"],
                        "direction": direction,
                        "start": shifting_start_idx,
                        "end": frame_idx,
                    }
                )

    inference_handler.reset()
    input_video_obj.release()

    labels["shiftings"] = [
        shifting
        for shifting in labels["shiftings"]
        if start_frame_idx <= shifting["start"] and shifting["end"] < end_frame_idx
    ]
    final_data = {
        "file_name": os.path.basename(video_path),
        "timestamp": datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
        "shiftings": shiftings,
        "scores": scores,
    }
    return labels, final_data

from typing import Optional
import time

from SaigeToolkit.util.test import load_numpy_image

from ..engine.monitoring import WindingMonitoringEngine


INFERENCE_IMAGE_PATH = "./assets/sample_frame.jpg"

THIN_DIVIDER = "-" * 40


def get_tact_time(
    segmentation: dict = None,
    monitoring_options: Optional[dict] = None,
    **kwargs,
):
    image = load_numpy_image(INFERENCE_IMAGE_PATH)

    inference_build_config = {
        "segmentation": segmentation,
        "monitoring_options": monitoring_options,
    }

    inference_handler = WindingMonitoringEngine.build(**inference_build_config)

    frame_idx = 0

    fps = 0
    fps_count = 0
    module_time_dict = {}
    start_time = time.time()
    last_fps_print_time = start_time
    while True:

        current_time = time.time()

        result = inference_handler.process_frame(image)
        for k, v in result["time"].items():
            if k not in module_time_dict:
                module_time_dict[k] = []
            module_time_dict[k].append(v)

        fps_count += 1
        if current_time - last_fps_print_time >= 5.0:  # 5초마다 FPS 출력
            module_time_summary = {k: sum(v) / len(v) for k, v in module_time_dict.items()}
            fps = fps_count / (current_time - last_fps_print_time)
            print(f"FPS: {fps:.2f} (총 {fps_count}프레임 / {current_time - last_fps_print_time:.2f}초)")
            for k, v in module_time_summary.items():
                print(f"{k}: {v:.2f} ms")
            print(THIN_DIVIDER)
            last_fps_print_time = current_time
            fps_count = 0

        frame_idx += 1

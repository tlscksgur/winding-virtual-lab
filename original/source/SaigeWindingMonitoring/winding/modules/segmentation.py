import os
import cv2
import numpy as np

from ..SaigeSegmentation2.segmentation import InferenceHandler
from ..SaigeSegmentation2.segmentation.engine.inference import InferenceHandler as SegInferenceHandler
from ..data.data_class import SegmentationResults, SideSegmentationResults
from ..error import SegmentationModelBuildError, SegmentationModelInferenceError
from ..utils.cache import save_data_pkl, load_data_pkl

from .checkpoint_handler import MergedCheckpointHandler

MIN_OBJECT_AREA_THRESHOLD = 1000

# Main view frame segmentation
WINDING_CABLE_INDEX = 1
DRUM_LEFT_CLASS_IDX = 2
DRUM_RIGHT_CLASS_IDX = 3
DRUM_BODY_CLASS_IDX = 4

CLASS_INDEX_TO_NAME: dict[int, str] = {
    DRUM_LEFT_CLASS_IDX: "left_drum",
    DRUM_RIGHT_CLASS_IDX: "right_drum",
    DRUM_BODY_CLASS_IDX: "drum_body",
    WINDING_CABLE_INDEX: "winding_cable",
}

# Side view frame segmentation
SIDE_DRUM_CLASS_IDX = 1
SIDE_DRUM_CENTER_CLASS_IDX = 2
CLASS_INDEX_TO_NAME_SIDE: dict[int, str] = {
    SIDE_DRUM_CLASS_IDX: "side_drum",
    SIDE_DRUM_CENTER_CLASS_IDX: "side_drum_center",
}


class SegmentationEngine:

    def __init__(
        self, main_inference_handler: InferenceHandler, side_inference_handler: InferenceHandler
    ):
        self.main_inference_handler = main_inference_handler
        self.side_inference_handler = side_inference_handler

    @classmethod
    def build(cls, checkpoint_path: str, password: str | None = None, device=0, **kwargs):
        """
        Build a segmentation engine for the SaigeSegmentation2 model.

        Parameters:
            checkpoint_path (str): Path to the model checkpoint.
            password (str): Password for the model checkpoint.
            device (DeviceType): Device to use for inference.

        Returns:
            InferenceHandler: The segmentation engine.
        """

        # XXX: 기본 옵션 - 필요한 것만 활성화
        inference_options = {
            "outputs.scoremap": False,
            "outputs.objects": True,
            "params.calc_object_area_and_apply_threshold": True,
        }

        try:
            main_ckpt, side_ckpt = MergedCheckpointHandler.load_merged(checkpoint_path, password)
            _main_inference_handler = SegInferenceHandler.build_from_checkpoint(
                checkpoint=main_ckpt, inference_options=inference_options, device=device
            )
            _side_inference_handler = SegInferenceHandler.build_from_checkpoint(
                checkpoint=side_ckpt, inference_options=inference_options, device=device
            )

            main_inference_handler = InferenceHandler(_main_inference_handler)
            side_inference_handler = InferenceHandler(_side_inference_handler)

        except:
            raise SegmentationModelBuildError("Failed to load merged checkpoint.")

        main_inference_handler.set_inference_option(
            "params.object_area_threshold",
            [0] + [MIN_OBJECT_AREA_THRESHOLD] * (main_inference_handler.handler.model.n_classes - 2),
        )
        side_inference_handler.set_inference_option(
            "params.object_area_threshold", [0] + [MIN_OBJECT_AREA_THRESHOLD]
        )

        handler = cls(main_inference_handler, side_inference_handler)

        return handler

    def inference(
        self,
        frame: np.ndarray,
        side_view_frame: np.ndarray,
        segmentation_cache_path: str | None = None,
        side_segmentation_cache_path: str | None = None,
    ) -> tuple[SegmentationResults, SideSegmentationResults]:
        main_segmentation_results = self.inference_main_frame(frame, segmentation_cache_path)
        # TODO: 측면 프레임 추론을 필요에 따라 수행하도록 수정 (회전 각도 측정 필요할 때만)
        side_segmentation_results = self.inference_side_frame(
            side_view_frame, side_segmentation_cache_path
        )

        return main_segmentation_results, side_segmentation_results

    def load_cached_segmentation(self, segmentation_cache_path: str | None) -> dict | None:
        # 연구소 내부 실험 가속을 위하여 캐싱된 세그멘테이션 결과를 불러옵니다.
        segmentation_results_dict = None
        if segmentation_cache_path and os.path.isfile(segmentation_cache_path):
            try:
                segmentation_results_dict = load_data_pkl(segmentation_cache_path)
            except:
                print(
                    f"Failed to load segmentation results from {segmentation_cache_path}. Running inference..."
                )

        return segmentation_results_dict

    def inference_main_frame(
        self, frame: np.ndarray, segmentation_cache_path: str | None = None
    ) -> SegmentationResults:
        segmentation_results_dict = self.load_cached_segmentation(segmentation_cache_path)

        if segmentation_results_dict is None:
            error_code, message, segmentation_results = (
                self.main_inference_handler.infer_and_postprocess(images=[frame])
            )
            if error_code != 0:
                raise SegmentationModelInferenceError(f"Error code {error_code} : {message}")
            segmentation_results_dict = self.parse_segmentation_results(
                segmentation_results, CLASS_INDEX_TO_NAME
            )
            if segmentation_cache_path:
                save_data_pkl(file_path=segmentation_cache_path, data=segmentation_results_dict)

        return SegmentationResults.from_dict(segmentation_results_dict)

    def inference_side_frame(
        self, frame: np.ndarray, segmentation_cache_path: str | None = None
    ) -> SideSegmentationResults:
        segmentation_results_dict = self.load_cached_segmentation(segmentation_cache_path)

        if segmentation_results_dict is None:
            error_code, message, segmentation_results = (
                self.side_inference_handler.infer_and_postprocess(images=[frame])
            )
            if error_code != 0:
                raise SegmentationModelInferenceError(f"Error code {error_code} : {message}")
            segmentation_results_dict = self.parse_segmentation_results(
                segmentation_results, CLASS_INDEX_TO_NAME_SIDE
            )
            if segmentation_cache_path:
                save_data_pkl(file_path=segmentation_cache_path, data=segmentation_results_dict)

        return SideSegmentationResults.from_dict(segmentation_results_dict)

    @staticmethod
    def parse_segmentation_results(
        segmentation_results: list, class_index_to_name: dict[int, str]
    ) -> dict:
        """
        기존 SEG의 출력을 현재 레포의 요구사항에 맞게 변환하여 리턴합니다.
        SEG 출력의 "objects"를 {CLASS_INDEX : OBJECT_INFO} 딕셔너리 형태로 변환합니다.
        단, 입력 이미지의 batch size는 1이어야 합니다.

        Returns:
            list[dict]: image의 postprocess 결과들이 들어있는 list 입니다.
                ```python
                    {
                        "segmentation_masks": {
                            "left_drum": (None, np.ndarray)  # 왼쪽 드럼에 대한 [H, W] shape의 np.ndarray(uint8) mask 입니다.
                            "right_drum": (None, np.ndarray)  # 오른쪽 드럼에 대한 [H, W] shape의 np.ndarray(uint8) mask 입니다.
                            "winding_cable": (None, np.ndarray)  # 권취 중인 전선에 대한 [H, W] shape의 np.ndarray(uint8) mask 입니다.
                            "drum_body": (None, np.ndarray)  # 권취가 완료된 전선에 대한 [H, W] shape의 np.ndarray(uint8) mask 입니다.
                        },
                        "segmentation_contours": {
                            "left_drum": (None, np.ndarray)  # 왼쪽 드럼에 대한 contour (N, 2) 입니다.
                            "right_drum": (None, np.ndarray)  # 오른쪽 드럼에 대한 contour (N, 2) 입니다.
                            "winding_cable": (None, np.ndarray)  # 권취 중인 전선에 대한 contour (N, 2) 입니다.
                            "drum_body": (None, np.ndarray)  # 권취가 완료된 전선에 대한 contour (N, 2) 입니다.
                        },
                        "bounding_boxes": {
                            "left_drum": (None, np.ndarray)  # 왼쪽 드럼을 둘러싸는 박스 [left, top, width, height] 입니다.
                            "right_drum": (None, np.ndarray)  # 오른쪽 드럼을 둘러싸는 박스 [left, top, width, height] 입니다.
                            "winding_cable": (None, np.ndarray)  # 권취 중인 전선을 둘러싸는 박스 [left, top, width, height] 입니다.
                            "drum_body": (None, np.ndarray)  # 권취가 완료된 전선을 둘러싸는 박스 [left, top, width, height] 입니다.
                        }
                    }
                ```
        """
        assert len(segmentation_results) == 1, "Batch size must be 1 for segmentation inference."

        segmentation_result = segmentation_results[0]
        objects = segmentation_result["objects"]
        mask_shape = segmentation_result["mask"].shape

        # Initialize
        predictions = {
            "segmentation_masks": {class_name: None for class_name in class_index_to_name.values()},
            "segmentation_contours": {class_name: None for class_name in class_index_to_name.values()},
            "bounding_boxes": {class_name: None for class_name in class_index_to_name.values()},
            "areas": {class_name: -1.0 for class_name in class_index_to_name.values()},
        }

        # Convert objects to the required format
        for obj in objects:
            class_index = obj.pop("class_index")
            class_name = class_index_to_name.get(class_index, None)

            if class_name is None:
                continue

            if obj["area"] > predictions["areas"][class_name]:
                # 가장 큰 contour를 선택합니다.
                contours = obj["contours"]
                largest_mask = np.zeros(mask_shape, dtype=np.uint8)

                if class_name == "side_drum":
                    contour = contours
                    cv2.drawContours(largest_mask, contour, -1, 1, thickness=cv2.FILLED)
                else:
                    contour = contours[0]  # 첫번째 contour가 outer (나머지는 inner)
                    contour = contour[:, 0, :]  # (N, 1, 2) -> (N, 2)
                    cv2.drawContours(largest_mask, [contour], -1, 1, thickness=cv2.FILLED)

                predictions["segmentation_masks"][class_name] = largest_mask
                predictions["segmentation_contours"][class_name] = contour
                predictions["bounding_boxes"][class_name] = obj["bounding_box"]
                predictions["areas"][class_name] = obj["area"]

        predictions.pop("areas")  # areas는 사용하지 않으므로 제거합니다.

        return predictions

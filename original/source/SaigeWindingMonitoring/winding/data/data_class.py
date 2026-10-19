import numpy as np
from dataclasses import dataclass


@dataclass
class BaseSegmentationResults:
    # 원본 데이터
    segmentation_masks: dict[str, np.ndarray | list[np.ndarray] | None]
    segmentation_contours: dict[str, np.ndarray | list[np.ndarray] | None]
    bounding_boxes: dict[str, list[int] | list[list[int]] | None]

    @classmethod
    def from_dict(cls, data: dict):
        """기존 dict 형태의 데이터로부터 생성"""
        return cls(
            segmentation_masks=data["segmentation_masks"],
            segmentation_contours=data["segmentation_contours"],
            bounding_boxes=data["bounding_boxes"],
        )

    def to_dict(self) -> dict:
        """dict 형태로 변환 (기존 코드 호환성)"""
        return {
            "segmentation_masks": self.segmentation_masks,
            "segmentation_contours": self.segmentation_contours,
            "bounding_boxes": self.bounding_boxes,
        }


@dataclass
class SegmentationResults(BaseSegmentationResults):
    """메인 뷰 카메라의 세그멘테이션 추론 결과를 담는 데이터 클래스"""

    @property
    def drum_body_mask(self) -> np.ndarray | None:
        """드럼 몸통 마스크"""
        return self.segmentation_masks.get("drum_body")

    @property
    def drum_body_contour(self) -> np.ndarray | None:
        """드럼 몸통 컨투어"""
        return self.segmentation_contours.get("drum_body")

    @property
    def drum_body_box(self) -> list[int] | None:
        """드럼 몸통 바운딩 박스"""
        return self.bounding_boxes.get("drum_body")

    @property
    def winding_cable_mask(self) -> np.ndarray | None:
        """권취 중인 전선 마스크"""
        return self.segmentation_masks.get("winding_cable")

    @property
    def winding_cable_contour(self) -> np.ndarray | None:
        """권취 중인 전선 컨투어"""
        return self.segmentation_contours.get("winding_cable")

    @property
    def winding_cable_box(self) -> list[int] | None:
        """권취 중인 전선 바운딩 박스"""
        return self.bounding_boxes.get("winding_cable")

    @property
    def left_drum_mask(self) -> np.ndarray | None:
        """왼쪽 드럼 마스크"""
        return self.segmentation_masks.get("left_drum")

    @property
    def left_drum_box(self) -> list[int] | None:
        """왼쪽 드럼 바운딩 박스"""
        return self.bounding_boxes.get("left_drum")

    @property
    def right_drum_mask(self) -> np.ndarray | None:
        """오른쪽 드럼 마스크"""
        return self.segmentation_masks.get("right_drum")

    @property
    def right_drum_box(self) -> list[int] | None:
        """오른쪽 드럼 바운딩 박스"""
        return self.bounding_boxes.get("right_drum")


@dataclass
class SideSegmentationResults(BaseSegmentationResults):
    """사이드 뷰 카메라의 세그멘테이션 추론 결과를 담는 데이터 클래스"""

    @property
    def side_drum_mask(self) -> np.ndarray | None:
        """사이드 드럼 마스크"""
        return self.segmentation_masks.get("side_drum")

    @property
    def side_drum_contour(self) -> np.ndarray | None:
        """사이드 드럼 컨투어"""
        return self.segmentation_contours.get("side_drum")

    @property
    def side_drum_box(self) -> list[int] | None:
        """사이드 드럼 바운딩 박스"""
        return self.bounding_boxes.get("side_drum")

    @property
    def side_drum_center_mask(self) -> np.ndarray | None:
        """사이드 드럼 중앙 마스크"""
        return self.segmentation_masks.get("side_drum_center")

    @property
    def side_drum_center_contour(self) -> np.ndarray | None:
        """사이드 드럼 중앙 컨투어"""
        return self.segmentation_contours.get("side_drum_center")

    @property
    def side_drum_center_box(self) -> list[int] | None:
        """사이드 드럼 중앙 바운딩 박스"""
        return self.bounding_boxes.get("side_drum_center")

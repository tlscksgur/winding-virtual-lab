import cv2
import numpy as np

from .mask import rotate_mask_for_flattened_top_line
from .roi import crop_edge_roi
from ..data.data_class import SegmentationResults


def check_edge_protrusion_by_height_difference(
    segmentation_results: SegmentationResults,
    roi_side: str,
    roi_narrowing_width: int,
    edge_region_ratio: float,
    downscale_factor: int | None = None,
    min_pixel_for_protrusion: int = 4,
    slope_degree: float | None = None,
) -> bool:
    """
    단일 프레임의 드럼 극단 영역 내에서 인접한 좌우 끝단의 상대적인 높이 차이를 비교하여 돌출 여부를 판단합니다.

    극단 영역의 좌측 끝과 우측 끝의 최상단 픽셀을 비교합니다.
    전선이 극단에 도달하여 층이 생기면 한쪽 끝이 다른 쪽보다 위로 솟아오릅니다.

    Args:
        segmentation_results: 세그멘테이션 결과 객체
        roi_side: 검사할 드럼 측면 ("left" 또는 "right")
        roi_narrowing_width: ROI 좁히기 너비
        edge_region_ratio: 극단 영역 비율
        downscale_factor: 다운스케일 팩터
        min_pixel_for_protrusion: 상단 돌출 여부를 판단하기 위한 최소 높이 차이 (픽셀)
        slope_degree: 드럼 상단 라인을 평탄화하기 위한 회전 각도 (도 단위)

    Returns:
        bool: 돌출이 감지되면 True, 아니면 False
    """

    rotated_body_mask, rotated_body_box = rotate_mask_for_flattened_top_line(
        segmentation_results.drum_body_mask,
        segmentation_results.drum_body_box,
        roi_side,
        slope_degree=slope_degree,
    )
    drum_edge_roi, _ = crop_edge_roi(
        rotated_body_mask,
        rotated_body_box,
        roi_side,
        margin=roi_narrowing_width,
        edge_region_ratio=edge_region_ratio,
        downscale_factor=downscale_factor,
        flatten=True,
    )

    if np.sum(drum_edge_roi) == 0:
        return False

    edge_roi_x, _, edge_roi_w, _ = cv2.boundingRect(drum_edge_roi)
    left_edge_top_y, _ = np.where(drum_edge_roi[:, edge_roi_x] == 1)[0][[0, -1]]
    right_edge_top_y, _ = np.where(drum_edge_roi[:, edge_roi_x + edge_roi_w - 1] == 1)[0][[0, -1]]

    if roi_side == "right":
        # 왼쪽에서 오른쪽으로 이동 중인 경우, 우측 끝이 좌측 끝보다 위로 솟았는지 검사
        if right_edge_top_y + min_pixel_for_protrusion < left_edge_top_y:
            return True
    else:
        # 오른쪽에서 왼쪽으로 이동 중인 경우, 좌측 끝이 우측 끝보다 위로 솟았는지 검사
        if left_edge_top_y + min_pixel_for_protrusion < right_edge_top_y:
            return True
    return False

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..utils.mask import downscale_mask, flatten_mask_edges

if TYPE_CHECKING:
    from ..modules.direction import Direction  # to avoid circular import


def extract_top_roi(
    box: list[int], mask: np.ndarray, side_top_percentage: float, roi_margin: int
) -> list[np.ndarray, int]:
    """
    Extract the top portion of a Region of Interest (ROI) from a mask with margins.

    Args:
        box (list[int]): Bounding box coordinates in format [x, y, width, height]
                                      where (x, y) is the top-left corner.
        mask (np.ndarray): 2D numpy array representing the image mask from which
                          the ROI will be extracted.
        side_top_percentage (float): Percentage (0.0 to 1.0) of the bounding box height
                                   to include in the top ROI. For example, 0.3 means
                                   the top 30% of the bounding box height.
        roi_margin (int): Number of pixels to add as margin around the ROI region.
                         This provides additional context around the extracted area.

    Returns:
        list[np.ndarray, int]: A list containing:
            - roi_mask (np.ndarray): Extracted ROI mask with shape (roi_height, roi_width)
            - roi_x (int): X-coordinate offset of the ROI in the original mask coordinates

    Example:
        >>> box = [100, 50, 200, 300]  # x=100, y=50, w=200, h=300
        >>> mask = np.random.randint(0, 2, (400, 500))
        >>> roi_mask, roi_x = extract_top_roi(box, mask, 0.2, 10)
        >>> print(roi_mask.shape)  # (70, 220) - top 20% + margins
        >>> print(roi_x)           # 90 - x coordinate with margin
    """
    x, y, w, h = box
    top_height = int(h * side_top_percentage)
    roi_x = max(x - roi_margin, 0)

    return [
        mask[
            max(y - roi_margin, 0) : y + top_height + roi_margin,
            roi_x : roi_x + w + roi_margin,
        ],
        roi_x,
    ]


def crop_edge_roi(
    mask: np.ndarray,
    box: list[int],
    roi_side: str,
    margin: int,
    edge_region_ratio: float,
    downscale_factor: int | None = None,
    flatten: bool = True,
):
    """
    Crop and process the edge region of interest (ROI) from a mask for edge analysis.

    This function extracts a specific edge region from a mask based on the specified side
    and applies optional downscaling and edge flattening operations.

    Args:
        mask (np.ndarray): Binary segmentation mask where 1 represents the object
            and 0 represents background.
        box (list[int]): Bounding box coordinates [x, y, width, height] defining
            the region of interest.
        roi_side (str): Side of the edge to extract. Must be either "left" or "right".
        margin (int): Number of pixels to add as margin around the ROI region.
            Provides additional context around the extracted edge area.
        edge_region_ratio (float): Ratio (0.0 to 1.0) of the bounding box width
            to include in the edge region. For example, 0.2 means the edge region
            covers 20% of the bounding box width from the specified side.
        downscale_factor (int | None, optional): Factor by which to downscale the ROI.
            If None, no downscaling is applied. Defaults to None.
        flatten (bool, optional): Whether to apply edge flattening to the ROI.
            Edge flattening helps in detecting protrusions by smoothing the edges.
            Defaults to True.

    Returns:
        tuple[np.ndarray, int]: A tuple containing:
            - drum_edge_roi (np.ndarray): The processed edge ROI mask with shape
              (roi_height, roi_width). May be downscaled and/or flattened based on
              the input parameters.
            - roi_x (int): X-coordinate offset of the ROI in the original mask
              coordinates, accounting for margins.

    """
    assert roi_side in ["left", "right"], "roi_side must be 'left' or 'right'"

    x, y, w, h = box

    if roi_side == "right":
        # 왼쪽에서 오른쪽으로 이동 중인 경우, 오른쪽 극단 영역을 검사합니다.
        roi_x = x - margin + int(w * (1 - edge_region_ratio))
        drum_edge_roi = mask[
            y - margin : y + margin + h,
            roi_x : x + margin + w,
        ]
    else:
        # 오른쪽에서 왼쪽으로 이동 중인 경우, 왼쪽 극단 영역을 검사합니다.
        roi_x = x - margin
        drum_edge_roi = mask[
            y - margin : y + margin + h,
            roi_x : x + int(w * edge_region_ratio) + margin,
        ]

    if downscale_factor is not None:
        drum_edge_roi = downscale_mask(drum_edge_roi, downscale_factor)
    if flatten:
        drum_edge_roi = flatten_mask_edges(drum_edge_roi, roi_side=roi_side)

    return drum_edge_roi, roi_x


def get_roi_side(state: str, cable_moving_direction: Direction) -> str | None:
    if cable_moving_direction.is_left_to_right():
        return "right" if state == "winding" else "left"
    elif cable_moving_direction.is_right_to_left():
        return "left" if state == "winding" else "right"
    return None


def get_angle_roi_side(state: str, cable_moving_direction: Direction) -> str | None:
    if cable_moving_direction.is_left_to_right():
        return "left" if state == "winding" else "right"
    elif cable_moving_direction.is_right_to_left():
        return "right" if state == "winding" else "left"
    return None

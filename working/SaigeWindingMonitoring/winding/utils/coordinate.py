import numpy as np


def convert_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    (x1, y1, x2, y2) 형식의 박스를 (x, y, w, h) 형식으로 변환하는 메서드.
    """
    if boxes is None:
        return None
    xywh_boxes = np.zeros_like(boxes)
    xywh_boxes[:, 0] = boxes[:, 0]
    xywh_boxes[:, 1] = boxes[:, 1]
    xywh_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh_boxes

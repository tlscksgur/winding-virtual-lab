import cv2
import numpy as np

SOBEL_KERNEL_SIZE = 25


def detect_corner_point(mask: np.ndarray) -> np.ndarray | None:
    # 1. 코너 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners: list[np.ndarray] = []
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)  # 정확도 조절
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        corners.append(approx)

    if len(corners) == 0:
        return None

    corners_arr = np.array(corners)
    corners_arr = corners_arr.reshape(-1, 2)

    return corners_arr


def compute_gradient(mask, points):
    grad_x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    grad_y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)

    xs, ys = points[:, 0], points[:, 1]
    gx = grad_x[ys, xs]
    gy = grad_y[ys, xs]

    angles = np.arctan2(gy, gx) * 180 / np.pi

    return angles


def detect_filtered_corner_point(
    mask: np.ndarray, box: list[int], angle_range: list[float]
) -> list[int] | None:
    """mask에서 코너 포인트를 추출하고 주어진 angle_range에 해당하는 포인트를 반환합니다."""

    _mask = mask.astype(np.uint8)
    x, y, w, h = box
    # XXX: 속도 향상을 위해 ROI로 자르기
    roi_mask = _mask[y : y + h, x : x + w]

    # 1. 코너 검출
    corners = detect_corner_point(roi_mask)
    if corners is None:
        return None

    # 2. gradient 계산
    angles = compute_gradient(roi_mask, corners)

    # 3. 필터 조건: y > threshold and angle in range
    ys = corners[:, 1]
    y_threshold = h // 2
    cond = (ys > y_threshold) & (angle_range[0] < angles) & (angles < angle_range[1])
    valid_corners = corners[cond]
    valid_angles = angles[cond]

    # 4. angle_range 중간값과 가장 가까운 포인트 선택
    if len(valid_corners) > 0:
        target_angle = (angle_range[0] + angle_range[1]) / 2
        idx = np.argmin(np.abs(valid_angles - target_angle))
        filtered = valid_corners[idx]
    else:
        return None

    # 좌표 조정: 원래 박스 좌표로 변환
    filtered = filtered + np.array([x, y])

    return [int(filtered[0]), int(filtered[1])]  # (x, y) 형태로 반환

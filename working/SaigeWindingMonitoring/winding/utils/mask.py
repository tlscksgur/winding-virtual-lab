import cv2
import numpy as np

ROI_MARGIN = 40
FITTING_RANGE = [0.15, 0.45]


def calculate_mask_center(mask: np.ndarray | None) -> tuple[float, float] | None:
    """
    세그멘테이션 마스크로부터 중심 좌표를 계산합니다.
    마스크의 모든 픽셀 좌표의 중앙값(median)을 반환합니다.

    Args:
        mask: 2D binary numpy array (H, W)

    Returns:
        (center_x, center_y): 중심 좌표 튜플, 마스크가 비어있으면 None
    """
    if mask is None or mask.size == 0 or mask.sum() == 0:
        return None

    # 마스크가 있는 모든 픽셀의 좌표 추출
    ys, xs = np.where(mask > 0)

    # 중앙값 계산
    center_x = float(np.median(xs))
    center_y = float(np.median(ys))

    return center_x, center_y


def flatten_mask_edges(
    mask: np.ndarray, roi_side: str | None = None, split_x: int | None = None, margin_ratio: float = 0.1
) -> np.ndarray:
    """
    Segmentation mask의 좌우 양끝을 평탄하게 만드는 함수
    ROI side에 따라 적응적으로 기준선을 설정합니다.

    - Right ROI: 왼쪽에서 25% 지점을 기준선으로
    - Left ROI: 왼쪽에서 75% 지점을 기준선으로
    - split_x 제공 시: split_x를 기준으로 좌우 margin_ratio 만큼 떨어진 구간만 평탄화

    Args:
        mask: 2D numpy array (binary or label mask)
        roi_side: "right" or "left" - ROI가 드럼의 어느 쪽인지 (split_x가 없을 경우 필수)
        split_x: winding cable의 x 위치 (제공 시 이를 기준으로 평탄화)
        margin_ratio: split_x 기준으로 좌우 마진 비율 (기본값 0.1 = 10%)

    Returns:
        flattened_mask: 평탄화된 mask

    Raises:
        ValueError: roi_side와 split_x 둘 다 제공되지 않은 경우
    """
    # roi_side와 split_x 둘 다 제공되지 않으면 예외 발생
    if split_x is None and roi_side not in ("right", "left"):
        raise ValueError(
            "Either 'split_x' must be provided or 'roi_side' must be 'right' or 'left'. "
            f"Got roi_side={roi_side}, split_x={split_x}"
        )

    flattened_mask = mask.copy()
    H, W = mask.shape

    # 각 행에 픽셀이 있는지 여부
    row_any = mask.any(axis=1)
    if not row_any.any():
        return flattened_mask

    # 각 행별 첫 픽셀(x 최소값) 계산
    row_min_x = np.full(H, W, dtype=int)
    row_min_x[row_any] = (mask[row_any] > 0).argmax(axis=1)

    # 각 행별 마지막 픽셀(x 최대값) 계산
    row_max_x = np.full(H, -1, dtype=int)
    rev = mask[:, ::-1] > 0
    idx_from_right = np.full(H, W, dtype=int)
    idx_from_right[row_any] = rev[row_any].argmax(axis=1)
    row_max_x[row_any] = W - 1 - idx_from_right[row_any]

    # 전체 마스크의 최소/최대 x
    min_x = row_min_x[row_any].min()
    max_x = row_max_x[row_any].max()
    total_width = max_x - min_x

    # split_x가 제공된 경우, 이를 기준으로 평탄화
    if split_x is not None:
        margin = int(total_width * margin_ratio)
        left_center_x = split_x - margin
        right_center_x = split_x + margin
        left_rows = row_any & (row_min_x <= left_center_x)
        right_rows = row_any & (row_max_x >= right_center_x)
    # ROI side에 따라 적응적 기준선 설정
    elif roi_side == "right":
        center_x = int(min_x + total_width * 0.25)
        left_rows = row_any & (row_min_x <= center_x)
        right_rows = row_any & (row_max_x > center_x)
    else:  # roi_side == "left"
        center_x = int(min_x + total_width * 0.75)
        left_rows = row_any & (row_min_x <= center_x)
        right_rows = row_any & (row_max_x > center_x)

    # 브로드캐스트용 x 좌표 그리드
    X = np.arange(W)[None, :]

    # 각 픽셀에 들어갈 값 배열 준비
    left_vals = np.zeros(H, dtype=mask.dtype)
    idx_left = np.nonzero(left_rows)[0]
    left_vals[idx_left] = mask[idx_left, row_min_x[idx_left]]

    right_vals = np.zeros(H, dtype=mask.dtype)
    idx_right = np.nonzero(right_rows)[0]
    right_vals[idx_right] = mask[idx_right, row_max_x[idx_right]]

    # 확장할 영역 마스크
    left_zone = (left_rows[:, None]) & (X >= min_x) & (X < row_min_x[:, None])
    right_zone = (right_rows[:, None]) & (X > row_max_x[:, None]) & (X <= max_x)

    # 값 브로드캐스트
    left_grid = np.broadcast_to(left_vals[:, None], (H, W))
    right_grid = np.broadcast_to(right_vals[:, None], (H, W))

    # 한 번의 벡터 연산으로 할당
    flattened_mask[left_zone] = left_grid[left_zone]
    flattened_mask[right_zone] = right_grid[right_zone]

    return flattened_mask


def downscale_mask(mask: np.ndarray, downscale_factor: int = 8, min_length: int = 3) -> np.ndarray:
    """드럼 몸통의 층 여부를 판단하기 위하여 저화질 마스크로 다운스케일링합니다."""
    if mask is None or mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
        return mask

    downscale_size = (
        max(min_length, mask.shape[1] // downscale_factor),
        max(min_length, mask.shape[0] // downscale_factor),
    )
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, downscale_size, interpolation=cv2.INTER_LINEAR)
    return mask


def find_top_edge_line_degree(
    mask: np.ndarray, roi_side: str | None, fitting_range: list[float] = FITTING_RANGE
) -> float | None:
    """
    양쪽 끝의 이상치를 제외하고 객체의 상단 경계를 가장 잘 나타내는 직선의 기울기를 찾습니다.
    """
    rows, cols = np.where(mask > 0)
    if len(cols) == 0:
        return None

    # 각 x에 대해 최상단 y를 벡터화 방식으로 추출
    max_x = cols.max() + 1
    min_y_per_x = np.full(max_x, np.inf, dtype=float)
    np.minimum.at(min_y_per_x, cols, rows)

    # 유효한 x,y만 추출
    valid_mask = np.isfinite(min_y_per_x)
    top_edge_x = np.nonzero(valid_mask)[0]
    top_edge_y = min_y_per_x[valid_mask]

    if len(top_edge_x) < 5:
        return None

    # 중앙부만 사용
    n = len(top_edge_x)
    if roi_side == "left":
        start_index = int(len(top_edge_x) * fitting_range[0])
        end_index = int(len(top_edge_x) * fitting_range[1])
    elif roi_side == "right":
        start_index = int(len(top_edge_x) * (1 - fitting_range[1]))
        end_index = int(len(top_edge_x) * (1 - fitting_range[0]))
    elif roi_side is None:
        # XXX: 극단을 판단하는 경우가 아닐 때, 중앙부 40% 고려
        start_index = int(len(top_edge_x) * 0.3)
        end_index = int(len(top_edge_x) * 0.7)

    central_x = top_edge_x[start_index:end_index]
    central_y = top_edge_y[start_index:end_index]

    if len(central_x) < 2:
        return None

    # 초기 직선 피팅
    initial_coeffs = np.polyfit(central_x, central_y, 1)
    coeff = initial_coeffs[0]

    # 기울기 → degree
    degree = np.arctan(coeff) * 180 / np.pi
    return degree


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """주어진 각도만큼 마스크를 회전시킵니다."""
    if mask is None or mask.size == 0:
        return mask

    # 회전 중심을 이미지의 중심으로 설정
    center = (mask.shape[1] // 2, mask.shape[0] // 2)

    # 회전 변환 행렬 생성
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 적용
    rotated_mask = cv2.warpAffine(
        mask, rotation_matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR
    )

    return rotated_mask


def calculate_rotation_angle_for_flattened_top_line(
    mask: np.ndarray,
    box: list[int],
    roi_side: str | None,
    fitting_range: list[float] = FITTING_RANGE,
    roi_margin: int = ROI_MARGIN,
) -> float | None:
    """Flattened top line을 기준으로 회전 각도를 계산합니다."""
    if mask is None or mask.size == 0:
        return None

    # Crop RoI
    mask_roi = mask[
        box[1] - roi_margin : box[1] + box[3] + roi_margin,
        box[0] - roi_margin : box[0] + box[2] + roi_margin,
    ].copy()

    slope_degree = find_top_edge_line_degree(mask_roi, roi_side, fitting_range)

    return slope_degree


def rotate_mask_for_flattened_top_line(
    mask: np.ndarray,
    box: list[int],
    roi_side: str | None,
    slope_degree: float | None = None,
    fitting_range: list[float] = FITTING_RANGE,
    roi_margin: int = ROI_MARGIN,
) -> tuple[np.ndarray, list[int]]:
    """Flattened top line을 기준으로 마스크를 회전시킵니다."""
    if mask is None or mask.size == 0:
        return mask, box

    # Crop RoI
    mask_roi = mask[
        box[1] - roi_margin : box[1] + box[3] + roi_margin,
        box[0] - roi_margin : box[0] + box[2] + roi_margin,
    ].copy()

    # 기울기가 주어지지 않으면, Flattened top line의 기울기를 찾습니다.
    slope_degree = (
        slope_degree
        if slope_degree is not None
        else find_top_edge_line_degree(mask_roi, roi_side, fitting_range)
    )

    if slope_degree is None:
        return mask, box

    # 회전시킵니다.
    rotated_mask_roi = rotate_mask(mask_roi, slope_degree)

    # Padding
    rotated_mask = np.zeros_like(mask)
    rotated_mask[
        box[1] - roi_margin : box[1] + box[3] + roi_margin,
        box[0] - roi_margin : box[0] + box[2] + roi_margin,
    ] = rotated_mask_roi

    # Rotated Box
    rotated_box = cv2.boundingRect(rotated_mask.astype(np.uint8))

    return rotated_mask, rotated_box

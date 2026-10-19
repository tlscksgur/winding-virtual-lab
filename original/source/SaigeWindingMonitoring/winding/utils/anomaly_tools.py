import numpy as np
from scipy.interpolate import make_smoothing_spline


def find_intervals(y: np.ndarray) -> list[tuple[int, int, float, int]]:
    dy = np.diff(y)
    dy_round = np.round(dy)
    signs = np.sign(dy_round)
    intervals = []
    start = None
    curr_sign = 0
    prev_i = 0
    for i, s in enumerate(signs):
        if s == 0 and curr_sign == 0:
            continue
        if s != curr_sign:
            if start is not None:
                end = prev_i
                if not np.isnan(y[start : end + 1]).all():
                    y_mean = float(np.nanmean(y[start : end + 1]))
                    if curr_sign != 0 and not np.isnan(y_mean):
                        intervals.append((start, end, y_mean, curr_sign))
            curr_sign = s
            start = i
        prev_i = i
    # finalize last run
    if start is not None:
        end = prev_i
        if not np.isnan(y[start : end + 1]).all():
            y_mean = float(np.nanmean(y[start : end + 1]))
            if curr_sign != 0 and not np.isnan(y_mean):
                intervals.append((start, end, y_mean, curr_sign))

    return intervals


def filter_same_direction_events(events: list[tuple[int, int, int, int, int]]):
    filtered = []
    for ev in events:
        if not filtered or ev[4] != filtered[-1][4]:
            filtered.append(ev)
    return filtered


def merge_boxes(boxes):
    if not boxes:
        return []
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    xe = [b[2] for b in boxes]
    ye = [b[3] for b in boxes]
    return [min(xs), min(ys), max(xe), max(ye)]


def trim_edge_slopes(mask_arr: np.ndarray, grad_arr: np.ndarray) -> np.ndarray:
    """경사 마스크의 시작/끝 True 구간을 False로 수축(테두리 잡음 제거)"""
    false_idx = np.where(~mask_arr)[0]
    if mask_arr[0] and false_idx.size:  # 앞쪽 자르기
        mask_arr[: false_idx[0]] = False
        grad_arr[: false_idx[0]] = 0.0
    if mask_arr[-1] and false_idx.size:  # 뒤쪽 자르기
        mask_arr[false_idx[-1] + 1 :] = False
        grad_arr[false_idx[-1] + 1 :] = 0.0
    return mask_arr


def interpolate_edge(idx: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """NaN 영역을 선형 보간으로 채운 배열 반환"""
    valid = ~np.isnan(arr)
    return np.interp(idx, idx[valid], arr[valid]) if valid.any() else arr


def smooth_gradient(idx: np.ndarray, edge: np.ndarray, smoothing_factor: float) -> np.ndarray:
    """실루엣을 미분 후 스무딩 Spline으로 부드럽게 만든 기울기 배열 반환"""
    grad = np.gradient(edge)
    nan_mask = np.isnan(grad)
    if (~nan_mask).any():  # 유효 구간이 있는 경우만 스무딩
        grad = make_smoothing_spline(idx[~nan_mask], grad[~nan_mask], lam=smoothing_factor)(idx)
    return grad


def convert_intervals_to_boxes(
    raw_events: list[tuple[int, int, float, int]],
    roi_x: int,
    roi_y: int,
    y_padding: int,
):
    """find_intervals 결과를 NG 박스 좌표로 변환"""
    return [
        (
            int(xs + roi_x),  # x1
            int(y_mean + roi_y) - y_padding,  # y1
            int(xe + roi_x),  # x2
            int(y_mean + roi_y) + y_padding,  # y2
            sign,  # 기울기 부호
        )
        for xs, xe, y_mean, sign in raw_events
    ]

import numpy as np
from scipy.interpolate import make_smoothing_spline


def smooth_ema(history, new_value, alpha=0.3):
    if not history:
        return new_value
    return alpha * new_value + (1 - alpha) * history[-1]


def filter_outlier(history, new_value, max_jump=0.3):
    if not history:
        return new_value
    prev = max(history)
    if new_value > prev + max_jump:
        return prev + max_jump
    return new_value


def calculate_smooth_slopes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate smoothed slopes from a series of 2D points.

    This function computes the slope between consecutive points and then applies
    smoothing using a cubic smoothing spline to reduce noise and create a more
    stable slope profile.

    Args:
        points (np.ndarray): Array of shape (n_points, 2) containing 2D points
                            in (x, y) format. Must have at least 2 points.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x_vals (np.ndarray): X-coordinates where slopes are calculated
                                 (length: n_points - 1)
            - smooth_slopes (np.ndarray): Smoothed slope values corresponding
    """
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    slopes = np.divide(dy, dx, out=np.full_like(dy, np.nan, dtype=float), where=dx != 0)
    x_vals = points[:-1, 0]
    smooth_func = make_smoothing_spline(x_vals, slopes, lam=5.0)
    smooth_slopes = smooth_func(x_vals)
    return x_vals, smooth_slopes

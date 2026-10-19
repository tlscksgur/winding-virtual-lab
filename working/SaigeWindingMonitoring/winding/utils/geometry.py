import numpy as np


def find_winding_top_x(mask: np.ndarray, percentage: float = 1.0, edge: str = "mean") -> int:
    """Estimate the X coordinate of the top percentage(%) of the mask."""
    ys, xs = np.where(mask == 1)
    if len(ys) == 0:
        return 0

    cutoff = np.percentile(ys, percentage)
    top_xs = xs[ys <= cutoff]
    if len(top_xs) == 0:
        return 0

    if edge == "min":
        return int(top_xs.min())
    if edge == "max":
        return int(top_xs.max())
    if edge == "mean":
        return int(top_xs.mean())
    raise ValueError(f"Invalid edge: {edge}")


def find_top_edge_points(roi: np.ndarray, roi_x: int) -> np.ndarray:
    """
    Find the top edge points of a Region of Interest (ROI).
    
    This function scans each column of the ROI from top to bottom and identifies
    the first non-zero pixel in each column, which represents the top edge point
    of the object within the ROI.
    
    Args:
        roi (np.ndarray): 2D numpy array representing the Region of Interest.
                         Non-zero values indicate the presence of an object.
        roi_x (int): X-coordinate offset to convert local ROI coordinates 
                    to global image coordinates.
    
    Returns:
        np.ndarray: Array of shape (n_points, 2) containing the top edge points
                   in (x, y) format. Each point represents the topmost pixel
                   of the object in each column of the ROI.
    
    Example:
        >>> roi = np.array([[0, 0, 0],
        ...                 [1, 1, 0],
        ...                 [1, 1, 1]])
        >>> points = find_top_edge_points(roi, 10)
        >>> print(points)
        [[10 1]   # First column: x=10, y=1
         [11 1]   # Second column: x=11, y=1
         [12 2]]  # Third column: x=12, y=2
    """
    points = []
    for x in range(roi.shape[1]):
        ys = np.where(roi[:, x] > 0)[0]
        if ys.size > 0:
            points.append((x + roi_x, ys.min()))
    return np.array(points)

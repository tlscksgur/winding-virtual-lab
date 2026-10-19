"""
Line detection utilities using LSD (Line Segment Detector)
"""

import numpy as np
from typing import Optional


def line_intersections(lines_a: np.ndarray, lines_b: np.ndarray):
    """선들의 교점을 계산합니다.
    Args:
        lines_a: 첫 번째 선들의 좌표 [N, 4] (x1, y1, x2, y2)
        lines_b: 두 번째 선들의 좌표 [M, 4] (x1, y1, x2, y2)
    Returns:
        교점들의 좌표 [N, M, 2] (x, y)
    """

    # Points in homogeneous coords: (x, y, 1)
    p1_a = np.concatenate([lines_a[:, 0:2], np.ones((lines_a.shape[0], 1))], axis=1)
    p2_a = np.concatenate([lines_a[:, 2:4], np.ones((lines_a.shape[0], 1))], axis=1)
    p1_b = np.concatenate([lines_b[:, 0:2], np.ones((lines_b.shape[0], 1))], axis=1)
    p2_b = np.concatenate([lines_b[:, 2:4], np.ones((lines_b.shape[0], 1))], axis=1)

    # Line in homogeneous form: l = p1 x p2  -> (a, b, c) with ax + by + c = 0
    l_a = np.expand_dims(np.cross(p1_a, p2_a), axis=1)  # (N, 1, 3)
    l_b = np.expand_dims(np.cross(p1_b, p2_b), axis=0)  # (1, M, 3)

    # Intersection in homogeneous coords: intersection = li x lj -> (x, y, w)
    intersection = np.cross(l_a, l_b)  # (N, M, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        points = np.divide(intersection[..., 0:2], intersection[..., 2:3])  # (N, M, 2)
    return points


def calculate_line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    """선의 길이를 계산합니다.

    Args:
        x1, y1: 선의 시작점 좌표
        x2, y2: 선의 끝점 좌표

    Returns:
        선의 길이 (pixels)
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_line_angle_from_center(
    x1: float, y1: float, x2: float, y2: float, cx: float, cy: float
) -> float:
    """선의 중점에서 드럼 중심까지의 각도를 계산합니다.

    Args:
        x1, y1: 선의 시작점 좌표
        x2, y2: 선의 끝점 좌표
        cx, cy: 드럼 중심 좌표

    Returns:
        각도 (degrees, 0-360 범위)
    """
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    angle = np.arctan2(mid_y - cy, mid_x - cx) * 180 / np.pi  # -180 ~ 180
    if angle < 0:
        angle += 360
    return angle


def is_radial_line(
    x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, angle_threshold: float = 25.0
) -> bool:
    """선이 중심을 향하는 방사형 선인지 판단합니다.

    Args:
        x1, y1: 선의 시작점 좌표
        x2, y2: 선의 끝점 좌표
        cx, cy: 드럼 중심 좌표
        angle_threshold: 방사형으로 간주할 각도 차이 임계값

    Returns:
        방사형 선 여부
    """
    # 선의 중점에서 중심으로의 각도
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    angle_to_center = np.arctan2(cy - mid_y, cx - mid_x) * 180 / np.pi

    # 선 자체의 방향
    line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    # 각도 차이 계산 (0-180도 범위로 정규화)
    angle_diff = abs(angle_to_center - line_angle)
    angle_diff = min(angle_diff, 360 - angle_diff)
    angle_diff = min(angle_diff, 180 - abs(angle_diff - 180))

    return angle_diff < angle_threshold


def detect_radial_lines(
    raw_lines: np.ndarray,
    center: tuple[float, float],
    angle_threshold: float = 25.0,
    min_angle: Optional[float] = None,
    max_angle: Optional[float] = None,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    LSD를 사용하여 방사형 선을 검출합니다.

    Args:
        raw_lines: 초기 line 검출 결과 [N, 4] (x1, y1, x2, y2)
        center: 드럼 중심 좌표 (x, y)
        angle_threshold: 방사형 선으로 간주할 각도 차이 임계값
        min_angle: 최소 각도 (degrees, None이면 제한 없음)
        max_angle: 최대 각도 (degrees, None이면 제한 없음)

    Returns:
        tuple of (lines, lengths, angles):
            - lines: 검출된 선들의 리스트 [N, 4] (x1, y1, x2, y2)
            - lengths: 각 선의 길이 리스트
            - angles: 각 선의 중심 각도 리스트 (degrees)
    """
    if len(raw_lines) == 0:
        return [], [], []

    # 방사형 선 필터링
    radial_lines = []
    lengths = []
    angles = []

    center_x, center_y = center

    for line in raw_lines.tolist():
        x1, y1, x2, y2 = line

        # 방사형 여부 확인
        if not is_radial_line(x1, y1, x2, y2, center_x, center_y, angle_threshold):
            continue

        length = calculate_line_length(x1, y1, x2, y2)
        angle = calculate_line_angle_from_center(x1, y1, x2, y2, center_x, center_y)

        # 각도 필터링
        if min_angle is not None and angle < min_angle:
            continue
        if max_angle is not None and angle > max_angle:
            continue

        radial_lines.append(line)
        lengths.append(length)
        angles.append(angle)

    return radial_lines, lengths, angles


def point_to_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """점에서 선분까지의 최단 거리를 계산합니다.

    Args:
        px, py: 점의 좌표
        x1, y1: 선의 시작점
        x2, y2: 선의 끝점

    Returns:
        점에서 선까지의 거리
    """
    # 선의 방향 벡터
    line_vec = np.array([x2 - x1, y2 - y1])
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq == 0:
        # 선이 점인 경우
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # 점에서 선의 시작점으로의 벡터
    point_vec = np.array([px - x1, py - y1])

    # 투영 비율 계산
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))

    # 선 위의 가장 가까운 점
    projection = np.array([x1, y1]) + t * line_vec

    # 거리 계산
    return np.sqrt((px - projection[0]) ** 2 + (py - projection[1]) ** 2)


def extend_line(x1: float, y1: float, x2: float, y2: float, extension_length: float = 100.0) -> tuple:
    """선을 양 끝에서 일정 거리만큼 연장합니다.

    Args:
        x1, y1: 선의 시작점
        x2, y2: 선의 끝점
        extension_length: 연장할 길이 (pixels)

    Returns:
        (ex1, ey1, ex2, ey2): 연장된 선의 끝점들
    """
    # 선의 방향 벡터
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        return x1, y1, x2, y2

    # 정규화된 방향 벡터
    ux = dx / length
    uy = dy / length

    # 양 끝에서 연장
    ex1 = x1 - ux * extension_length
    ey1 = y1 - uy * extension_length
    ex2 = x2 + ux * extension_length
    ey2 = y2 + uy * extension_length

    return ex1, ey1, ex2, ey2


def lines_can_merge_with_extension(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    angle_diff_threshold: float = 10.0,
    extension_distance_threshold: float = 50.0,
    collinearity_threshold: float = 15.0,
) -> bool:
    """선을 연장했을 때 병합 가능한지 확인합니다.

    Args:
        x1, y1, x2, y2: 첫 번째 선
        x3, y3, x4, y4: 두 번째 선
        angle_diff_threshold: 각도 차이 임계값
        extension_distance_threshold: 연장 후 인접성 거리 임계값
        collinearity_threshold: 직선성 거리 임계값

    Returns:
        병합 가능 여부
    """
    # 1. 선의 방향 각도 계산
    angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi

    angle_diff = abs(angle1 - angle2)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    if angle_diff > angle_diff_threshold:
        return False

    # 2. 선을 연장
    ex1_1, ey1_1, ex1_2, ey1_2 = extend_line(x1, y1, x2, y2, extension_distance_threshold)

    # 3. 연장된 선과 다른 선의 끝점들 사이의 거리 계산
    dist3_to_ext1 = point_to_line_distance(x3, y3, ex1_1, ey1_1, ex1_2, ey1_2)
    dist4_to_ext1 = point_to_line_distance(x4, y4, ex1_1, ey1_1, ex1_2, ey1_2)

    min_dist = min(dist3_to_ext1, dist4_to_ext1)

    if min_dist > collinearity_threshold:
        return False

    # 4. 실제 끝점 간 거리 체크
    distances = [
        np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2),
        np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2),
        np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
        np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
    ]
    min_actual_distance = min(distances)

    # 연장 거리의 1.5배 이내여야 함
    if min_actual_distance > extension_distance_threshold * 1.5:
        return False

    return True


def merge_adjacent_lines(
    lines: list[np.ndarray],
    angles: list[float],
    center: tuple[float, float],
    angle_diff_threshold: float = 10.0,
    distance_threshold: float = 30.0,
    collinearity_threshold: float = 10.0,
    max_iterations: int = 10,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """인접한 유사 각도의 선들을 반복적으로 merge합니다.

    선을 연장하여 같은 직선상에 있는 선들을 적극적으로 병합합니다.
    Union-Find 알고리즘을 사용하여 모든 선 쌍을 검사합니다.

    Args:
        lines: 검출된 선들의 리스트 [N, 4]
        angles: 각 선의 중심 각도 리스트 (degrees)
        center: 드럼 측면 중심 좌표 (x, y)
        angle_diff_threshold: merge를 위한 각도 차이 임계값 (degrees)
        distance_threshold: 인접성 판단을 위한 거리 임계값 (pixels)
        collinearity_threshold: 직선성 판단을 위한 거리 임계값 (pixels)
        max_iterations: 최대 병합 반복 횟수

    Returns:
        tuple of (merged_lines, merged_lengths, merged_angles)
    """
    if len(lines) == 0:
        return [], [], []

    current_lines = lines
    current_angles = angles
    center_x, center_y = center

    # 반복적으로 병합 수행 (Union-Find 기반)
    for _ in range(max_iterations):
        n_lines = len(current_lines)

        # 모든 선 쌍에 대해 병합 가능 여부를 확인
        merge_graph = [[False] * n_lines for _ in range(n_lines)]

        # 모든 선 쌍을 비교 (순서 무관, 모든 조합 검사)
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                x1, y1, x2, y2 = current_lines[i]
                x3, y3, x4, y4 = current_lines[j]

                angle_i = current_angles[i]
                angle_j = current_angles[j]

                # 각도 차이 확인 (360도 wrap 고려)
                angle_diff = abs(angle_i - angle_j)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                # 관대한 각도 임계값 사용 (기존보다 2배 관대)
                if angle_diff > angle_diff_threshold * 2.0:
                    continue

                # 연장하여 병합 가능한지 체크 (더 관대한 파라미터)
                can_merge = lines_can_merge_with_extension(
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                    x4,
                    y4,
                    angle_diff_threshold=angle_diff_threshold * 2.0,
                    extension_distance_threshold=distance_threshold * 3,
                    collinearity_threshold=collinearity_threshold * 2.0,
                )

                if can_merge:
                    merge_graph[i][j] = True
                    merge_graph[j][i] = True

        # Union-Find를 사용하여 연결된 선들을 그룹화
        parent = list(range(n_lines))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 병합 가능한 선들을 union
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                if merge_graph[i][j]:
                    union(i, j)

        # 그룹별로 선들을 모음
        groups = {}
        for i in range(n_lines):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # 각 그룹을 하나의 선으로 병합
        merged_lines = []
        merged_angles = []

        merge_occurred = False

        for group_indices in groups.values():
            if len(group_indices) > 1:
                merge_occurred = True

            # 그룹의 모든 끝점을 수집
            all_points = []
            angle_sum = 0

            for idx in group_indices:
                x1, y1, x2, y2 = current_lines[idx]
                all_points.extend([[x1, y1], [x2, y2]])
                angle_sum += current_angles[idx]

            all_points = np.array(all_points)

            # 중심에서 가장 먼 점과 가까운 점을 선택
            distances_from_center = [
                np.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) for p in all_points
            ]
            farthest_idx = np.argmax(distances_from_center)
            closest_idx = np.argmin(distances_from_center)

            # 병합된 선
            x1, y1 = all_points[closest_idx]
            x2, y2 = all_points[farthest_idx]

            merged_line = np.array([x1, y1, x2, y2])
            merged_lines.append(merged_line)

            # 평균 각도
            avg_angle = angle_sum / len(group_indices)
            merged_angles.append(avg_angle)

        # 병합이 발생하지 않았으면 종료
        if not merge_occurred or len(merged_lines) == len(current_lines):
            current_lines = merged_lines
            current_angles = merged_angles
            break

        # 다음 반복을 위해 업데이트
        current_lines = merged_lines
        current_angles = merged_angles

    # 최종 결과 계산
    final_lines = []
    final_lengths = []
    final_angles = []

    for line in current_lines:
        x1, y1, x2, y2 = line
        final_lines.append(line)

        length = calculate_line_length(x1, y1, x2, y2)
        final_lengths.append(length)

        # 각도 재계산 (정확도 향상)
        final_angle = calculate_line_angle_from_center(x1, y1, x2, y2, center_x, center_y)
        final_angles.append(final_angle)

    return final_lines, final_lengths, final_angles

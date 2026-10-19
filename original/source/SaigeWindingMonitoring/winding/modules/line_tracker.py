"""
Line-based rotation tracking module

This module replaces marker-based tracking with radial line detection for rotation angle calculation.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.data_class import SideSegmentationResults
from ..engine.config import RotationSettings
from ..utils.line_detection import detect_radial_lines, merge_adjacent_lines, line_intersections
from ..utils.mask import calculate_mask_center


class MultiLineTracker:
    """드럼 옆면의 방사형 선들을 tracking하여 회전 각도를 계산하는 클래스"""

    def __init__(
        self,
        rotation_config: RotationSettings,
        center: tuple[float, float] | None = None,
    ):
        self.rotation_config = rotation_config

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_NONE)

        # 드럼 중심 (동적으로 설정 가능, tuple로 관리)
        if center is not None:
            self.center: tuple[float, float] | None = center
            self.center_reset = False  # XXX: 수동으로 중심을 설정한 경우는, reset하지 않음.
        else:
            self.center = None
            self.center_reset = True

        # 개별 선 추적을 위한 딕셔너리
        self.lines: dict[int, TrackedLine] = {}
        self.next_line_id = 0

        # 일시적으로 소실된 선들을 추적 (ID: TrackedLine)
        self.lost_lines: dict[int, TrackedLine] = {}
        self.max_lost_frames = 0  # 직선의 소실을 허용할 프레임 수

        # 평균 프레임 각도 변화 히스토리
        self.average_frame_angle_changes: list[float] = []

        # 회전 방향 파라미터 파싱 및 초기화 ("clockwise", "counter_clockwise")
        rotation_direction_mode = self.rotation_config.rotation_direction.lower()

        # 회전 방향 추적 (1: 시계방향, -1: 반시계방향, 0: 미정/자동)
        if rotation_direction_mode == "clockwise":
            self.rotation_direction = 1
        elif rotation_direction_mode == "counter_clockwise":
            self.rotation_direction = -1
        else:
            raise ValueError(f"Invalid rotation direction: {rotation_direction_mode}")

        self.opposite_direction_margin = 0.1  # 반대 방향 허용 마진 (degree)

        # 프레임 간 픽셀 차이 기반 정지 감지를 위한 변수
        self.previous_frame: np.ndarray | None = None

    def update(
        self,
        image: np.ndarray,
        side_segmentation_results: SideSegmentationResults,
        angle_change_threshold: float = 0.2,
    ):
        """
        선 위치 업데이트 및 각도 변화 계산

        Args:
            image: 입력 이미지 (BGR)
            side_segmentation_results: 사이드 뷰 세그멘테이션 결과 (optional)
                - side_drum_mask: 드럼 영역 마스크
                - side_drum_center_mask: 드럼 중심 영역 마스크 (중심 좌표 계산용)
            angle_change_threshold: 최소 각도 변화 임계값 (degree)
                - 이 값 이하의 각도 변화는 무시됩니다. (noise 제거용)
        """

        side_drum_mask = side_segmentation_results.side_drum_mask
        side_drum_box = side_segmentation_results.side_drum_box

        if side_drum_mask is None or side_drum_box is None:
            return

        # 이미지 전처리
        masked_gray = self._preprocess_image(image, side_drum_mask)

        # 초기 line 검출
        raw_lines = self._detect_lines(masked_gray, side_drum_box)

        # 빈 드럼 등으로 인해 선이 검출되지 않은 경우 조기 반환
        if raw_lines is None or len(raw_lines) == 0:
            return

        # 방사형 라인으로부터 중심 좌표 자동 계산
        if self.center is None:
            self.center = self._calculate_lines_center(masked_gray, side_drum_box)

        # 중심 좌표 검증
        if self.center is None:
            return

        # 프레임 간 픽셀 차이 기반 정지 감지
        if self._is_frame_stationary(masked_gray, side_drum_mask, side_drum_box):
            # 정지 상태로 판단되면 각도 변화 0으로 기록
            self.average_frame_angle_changes.append(0.0)
            # 이전 프레임 정보 업데이트
            self.previous_frame = masked_gray.copy()
            return

        # 현재 프레임에서 선 검출
        current_lines_data = self._process_lines(raw_lines)

        # 이전 프레임과 선 매칭
        self._match_lines(current_lines_data)

        # 평균 프레임 각도 변화 계산
        average_frame_angle_change = self._calculate_average_frame_angle_change()
        # XXX: 노이즈 제거 (회전이 없을 때도 미세한 변화가 감지되는 문제 해결)
        if (
            average_frame_angle_change is not None
            and abs(average_frame_angle_change) < angle_change_threshold
        ):
            average_frame_angle_change = 0.0

        if average_frame_angle_change is not None:
            self.average_frame_angle_changes.append(average_frame_angle_change)

        # 이전 프레임 정보 업데이트
        self.previous_frame = masked_gray.copy()

    def _preprocess_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """이미지 전처리

        Args:
            image: 입력 이미지 (BGR)
            mask: 마스크 이미지 (0: 배경, 1: 드럼)
        Returns:
            masked_gray: 마스크 적용 이미지 (gray)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = self.clahe.apply(gray)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        return masked_gray

    def _detect_lines(self, gray_image: np.ndarray, side_drum_box: list[int]) -> np.ndarray:
        """side_drum_box 영역에서 line을 검출

        Args:
            gray_image: 입력 이미지 (gray)
            side_drum_box: line을 검출할 바운딩 박스 [x, y, w, h]
        Returns:
            lines: 길이가 최소 최대 길이 사이인 line 검출 결과 [N, 4] (x1, y1, x2, y2)
        """

        # RoI 크롭
        x, y, w, h = side_drum_box
        roi_gray_image = gray_image[y : y + h, x : x + w]

        # line 검출
        lines = self.lsd.detect(roi_gray_image)[0]  # [N, 1, 4] (x1, y1, x2, y2)

        # 빈 드럼 등으로 인해 선이 검출되지 않은 경우 빈 배열 반환
        if lines is None:
            return np.array([], dtype=np.float32).reshape(0, 4)

        lines = lines.reshape(-1, 4)  # [N, 4]

        # 길이 필터링
        lengths = np.sqrt(np.sum(np.square(lines[:, :2] - lines[:, 2:]), axis=1))
        is_valid = np.ones(lines.shape[0], dtype=bool)

        if self.rotation_config.min_line_length is not None:
            is_valid &= lengths >= self.rotation_config.min_line_length
        if self.rotation_config.max_line_length is not None:
            is_valid &= lengths <= self.rotation_config.max_line_length

        lines = lines[is_valid]

        # 원래 박스 좌표로 변환
        lines += np.array([x, y, x, y])

        return lines

    def _calibrate_image(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """이미지 왜곡 보정

        Args:
            image: 입력 이미지 (color or gray)
        Returns:
            calibrated_image: 왜곡 보정 이미지 (color or gray)
            calibrate_params: 왜곡 보정 파라미터 (camera_matrix, distort_coeffs, new_camera_matrix)
        """
        h, w = image.shape[:2]

        center_x, center_y = w / 2.0, h / 2.0
        focal = max(w, h) * 1.2
        camera_mat = np.array([[focal, 0, center_x], [0, focal, center_y], [0, 0, 1]])

        # Distortion Coefficients: (k1, k2, p1, p2, k3)
        distort_coeffs = np.array(self.rotation_config.camera_distortion_coeffs)
        new_camera_mat, _ = cv2.getOptimalNewCameraMatrix(camera_mat, distort_coeffs, (w, h), alpha=0.0)

        calibrate_params = (camera_mat, distort_coeffs, new_camera_mat)
        calibrated_image = cv2.undistort(image, camera_mat, distort_coeffs, None, new_camera_mat)

        return calibrated_image, calibrate_params

    def _decalibrate_points(
        self,
        points: np.ndarray,
        calibrate_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """왜곡 보정 이미지로부터 좌표를 변환

        Args:
            points: 왜곡 보정 이미지로부터 좌표 [N, 2] (x, y)
            calibrate_params: 왜곡 보정 파라미터 (camera_matrix, distort_coeffs, new_camera_matrix)
        Returns:
            decalibrated_points: 원본(왜곡된) 이미지로부터 좌표 [N, 2] (x, y)
        """
        camera_mat, distort_coeffs, new_camera_mat = calibrate_params

        # 보정 이미지 픽셀 -> 정규화 좌표 (x, y)
        focal_x, focal_y = new_camera_mat[0, 0], new_camera_mat[1, 1]
        center_x, center_y = new_camera_mat[0, 2], new_camera_mat[1, 2]
        x = (points[:, 0] - center_x) / focal_x
        y = (points[:, 1] - center_y) / focal_y

        # Points in homogeneous coords: (x, y, 1)
        points = np.stack([x, y, np.ones_like(x)], axis=1).reshape(-1, 1, 3)
        zeros = np.zeros((3, 1))

        decalibrated_points, _ = cv2.projectPoints(points, zeros, zeros, camera_mat, distort_coeffs)
        decalibrated_points = decalibrated_points.reshape(-1, 2)
        return decalibrated_points

    def _calculate_lines_center(
        self,
        masked_gray_image: np.ndarray,
        side_drum_box: list[int],
    ) -> tuple[float, float] | None:
        """선들의 중심 좌표를 계산

        Args:
            lines: 선들의 좌표 [N, 4] (x1, y1, x2, y2)
        Returns:
            center: 선들의 중심 좌표 (x, y)
        Notes:
            - 선들의 중심 좌표 범위를 지나는 선들과 주요 선들을 필터링하여 중심 좌표를 계산
            - 검출된 선들과 필터링된 선들의 개수가 10개 미만이면 오차가 커질 수 있기 때문에 None을 반환
        """

        x1, y1, x2, y2 = self.rotation_config.drum_center_point_range

        # Line 검출
        # XXX: 이미지 왜곡을 보정하기 위하여 미리 구한 line 검출 결과를 사용하지 않고 재계산
        calibrated_image, calibrate_params = self._calibrate_image(masked_gray_image)
        lines = self._detect_lines(calibrated_image, side_drum_box)

        if lines is None or len(lines) < 10:
            return None

        # 드럼 중심 좌표 범위와 선들의 교점을 계산
        pts = line_intersections(lines, np.asarray([[x1, y1, x2, y2]])).reshape(-1, 2)  # [N, 2]
        x, y = pts[:, 0], pts[:, 1]

        # 드럼 중심 좌표 범위를 지나는 선들과 주요 선들을 필터링
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        is_radial = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)

        hist, edges = np.histogram(y[is_radial], bins=9)
        max_idx = np.argmax(hist)
        min_edge, max_edge = edges[max(max_idx - 2, 0)], edges[min(max_idx + 3, len(edges) - 1)]

        is_center = (min_edge <= y) & (y <= max_edge)
        center_lines = lines[is_center]

        if len(center_lines) < 10:
            return None

        # 주요 선들의 교점을 계산
        center_pts = line_intersections(center_lines, center_lines).reshape(-1, 2)  # [N, 2]
        x, y = center_pts[:, 0], center_pts[:, 1]

        # 범위를 크게 벗어난 교점을 제거
        margin = 100
        is_inlier = (x1 - margin <= x) & (x <= x2 + margin) & (y1 - margin <= y) & (y <= y2 + margin)
        x, y = x[is_inlier], y[is_inlier]

        if len(x) < 10:
            return None

        # 중심 좌표 계산
        center_x1, center_x2 = np.nanquantile(x, [0.3, 0.7])
        center_y1, center_y2 = np.nanquantile(y, [0.3, 0.7])

        is_valid = (center_x1 <= x) & (x <= center_x2) & (center_y1 <= y) & (y <= center_y2)

        cx, cy = x[is_valid].mean(), y[is_valid].mean()

        # 왜곡 보정 이미지로부터 중심 좌표 계산
        decalib_points = self._decalibrate_points(np.asarray([[cx, cy]]), calibrate_params)
        cx, cy = decalib_points[0].tolist()

        return (cx, cy)

    def _is_frame_stationary(
        self,
        current_frame: np.ndarray,
        current_mask: np.ndarray,
        side_drum_box: list[int],
        motion_pixel_diff_threshold: float = 7.0,
    ) -> bool:
        """프레임 간 픽셀 차이를 계산하여 정지 상태인지 판단 (RoI 기반 최적화)

        Args:
            current_frame: 현재 프레임 이미지
            current_mask: 현재 프레임의 세그멘테이션 마스크
            side_drum_box: 사이드 드럼 바운딩 박스 [x, y, w, h]

        Returns:
            True if stationary (평균 픽셀 차이가 threshold 이하), False otherwise
        """
        # 이전 프레임이 없으면 정지 상태가 아님
        if self.previous_frame is None:
            return False

        # 프레임 크기가 다르면 비교 불가
        if current_frame.shape != self.previous_frame.shape:
            return False

        # RoI 계산: side_drum_box의 좌상단 1/4 영역
        x, y, w, h = side_drum_box
        roi_x2 = x + w // 2
        roi_y2 = y + h // 2

        # RoI 크롭
        current_roi = current_frame[y:roi_y2, x:roi_x2]
        previous_roi = self.previous_frame[y:roi_y2, x:roi_x2]
        mask_roi = current_mask[y:roi_y2, x:roi_x2]

        # 마스크 영역 내의 평균 차이만 계산
        pixel_diff = cv2.absdiff(current_roi, previous_roi)
        mean_diff = cv2.mean(pixel_diff, mask=mask_roi)[0]

        # 임계값과 비교
        return mean_diff < motion_pixel_diff_threshold

    def _process_lines(self, raw_lines: np.ndarray) -> list[dict]:
        """현재 프레임에서 방사형 선을 검출하고 정보를 추출합니다.
        Args:
            raw_lines: 초기 line 검출 결과 [N, 4] (x1, y1, x2, y2)
        Returns:
            lines: 방사형 선 검출 결과
            lengths: 방사형 선 길이
            angles: 방사형 선 각도
        """
        assert self.center is not None

        lines, lengths, angles = detect_radial_lines(
            raw_lines,
            center=self.center,
            angle_threshold=self.rotation_config.angle_threshold,
            min_angle=self.rotation_config.min_line_angle,
            max_angle=self.rotation_config.max_line_angle,
        )

        # 인접한 유사 각도의 선들을 merge
        if self.rotation_config.enable_line_merging and len(lines) > 0:
            lines, lengths, angles = merge_adjacent_lines(
                lines=lines,
                angles=angles,
                center=self.center,
                angle_diff_threshold=self.rotation_config.merge_angle_threshold,
                distance_threshold=self.rotation_config.merge_distance_threshold,
                collinearity_threshold=self.rotation_config.merge_collinearity_threshold,
                max_iterations=self.rotation_config.max_merge_iterations,
            )

        lines_data = []
        for line, length, angle in zip(lines, lengths, angles):
            x1, y1, x2, y2 = line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            lines_data.append(
                {
                    "line_coords": (x1, y1, x2, y2),
                    "center": (mid_x, mid_y),
                    "angle": angle,
                    "length": length,
                }
            )

        return lines_data

    def _match_lines(self, current_lines: list[dict], min_angle_change_threshold: float = 1.0) -> None:
        """현재 프레임의 선을 이전 프레임과 최적 할당 방식으로 매칭

        개선된 매칭 알고리즘:
        - 각도와 위치를 모두 고려
        - 이전 프레임의 회전 속도를 활용한 예측
        - 적응형 임계값
        """
        if not self.lines:
            # 첫 프레임인 경우 모든 선에 새 ID 할당
            for line_data in current_lines:
                line_id = self.next_line_id
                self.lines[line_id] = TrackedLine(line_id)
                self.lines[line_id].update(line_data)
                self.next_line_id += 1
            return

        if not current_lines:
            # 현재 감지된 선이 없으면 모든 이전 선을 삭제
            self.lines.clear()
            return

        # 현재 추적 중인 선 + 소실된 선을 모두 고려
        all_prev_lines = {**self.lines, **self.lost_lines}
        prev_ids = list(all_prev_lines.keys())

        if not prev_ids:
            # 이전 선이 없으면 모두 새로 할당
            for line_data in current_lines:
                line_id = self.next_line_id
                self.lines[line_id] = TrackedLine(line_id)
                self.lines[line_id].update(line_data)
                self.next_line_id += 1
            return

        # 이전 프레임의 평균 회전 속도 계산
        avg_rotation = self._calculate_average_frame_angle_change()
        if avg_rotation is None:
            avg_rotation = 0.0

        # 방향에 맞게 부호를 강제 조정
        avg_rotation = abs(avg_rotation) * self.rotation_direction

        # 비용 행렬 계산 (각도 + 위치 + 예측)
        cost_matrix = np.zeros((len(current_lines), len(prev_ids)))

        for i, current_line in enumerate(current_lines):
            current_angle = current_line["angle"]
            current_center = current_line["center"]

            for j, prev_id in enumerate(prev_ids):
                prev_line = all_prev_lines[prev_id]
                prev_angle = prev_line.last_angle

                # 예상 각도 계산 (이전 각도 + 평균 회전량)
                predicted_angle = prev_angle + avg_rotation

                angle_diff = current_angle - predicted_angle

                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360

                # 1. 각도 비용 (예측 기반)
                angle_cost = abs(angle_diff)

                # 1-1. 회전 방향 체크 / 각도 변화량이 매우 작다면 무시
                if abs(angle_diff) >= min_angle_change_threshold:

                    # 회전 방향과 각도 변화의 부호가 일치하는지 확인
                    expected_sign = self.rotation_direction
                    actual_sign = 1 if angle_diff > 0 else -1 if angle_diff < 0 else 0

                    # 방향이 반대이면 페널티 부여
                    if actual_sign != 0 and actual_sign != expected_sign:
                        # 반대 방향이지만 마진 범위 내인지 확인
                        within_margin = abs(angle_diff) <= self.opposite_direction_margin

                        if not within_margin:
                            # 임계값 이상의 반대 방향 변화는 매우 큰 비용으로 설정 (반대 방향 매칭 완전 차단)
                            if abs(angle_diff) > 1.0:
                                angle_cost = 10000.0  # 매우 큰 비용 (실질적으로 매칭 불가)
                            else:
                                # 작은 변화는 경고 페널티만
                                angle_cost += 100.0

                # 2. 위치 비용 (중심점 거리)
                if len(prev_line.positions) > 0:
                    prev_center = (prev_line.positions[-1][0], prev_line.positions[-1][1])
                    distance = np.sqrt(
                        (current_center[0] - prev_center[0]) ** 2
                        + (current_center[1] - prev_center[1]) ** 2
                    )
                    # 거리를 각도 스케일로 변환 (100 pixels = 50도 상당)
                    position_cost = distance * 0.5
                else:
                    position_cost = 0

                # 3. 선의 방향(orientation) 비용
                # 선의 끝점들을 사용하여 방향 계산
                if "line_coords" in current_line:
                    cx1, cy1, cx2, cy2 = current_line["line_coords"]
                    current_orientation = np.arctan2(cy2 - cy1, cx2 - cx1) * 180 / np.pi

                    if len(prev_line.positions) > 0 and len(prev_line.positions[-1]) >= 7:
                        px1, py1, px2, py2 = prev_line.positions[-1][3:7]
                        prev_orientation = np.arctan2(py2 - py1, px2 - px1) * 180 / np.pi

                        orientation_diff = abs(current_orientation - prev_orientation)
                        orientation_diff = min(orientation_diff, 360 - orientation_diff)
                        # 방향이 반대일 수도 있으므로 180도 차이도 허용
                        orientation_diff = min(orientation_diff, abs(180 - orientation_diff))
                        orientation_cost = orientation_diff * 0.5
                    else:
                        orientation_cost = 0
                else:
                    orientation_cost = 0

                # 4. 길이 유사성 비용
                current_length = current_line.get("length", 0)
                prev_length = (
                    prev_line.positions[-1][7]
                    if len(prev_line.positions) > 0 and len(prev_line.positions[-1]) > 7
                    else current_length
                )
                if prev_length > 0 and current_length > 0:
                    length_ratio = max(current_length, prev_length) / min(current_length, prev_length)
                    # 길이가 2배 이상 차이나면 페널티
                    length_cost = max(0, (length_ratio - 1.0) * 20)
                else:
                    length_cost = 0

                # 5. 총 비용 (가중 합)
                # 각도 50%, 위치 25%, 방향 15%, 길이 10%
                total_cost = (
                    angle_cost * 0.5 + position_cost * 0.25 + orientation_cost * 0.15 + length_cost * 0.1
                )

                cost_matrix[i, j] = total_cost

        # 헝가리안 알고리즘을 사용하여 최적의 매칭 찾기
        current_indices, prev_indices = linear_sum_assignment(cost_matrix)

        matched_current_indices = set()
        matched_prev_ids = set()

        # 적응형 임계값 계산
        # 회전 속도가 빠를수록 임계값을 높임
        base_threshold = 50.0
        adaptive_threshold = base_threshold + abs(avg_rotation) * 2
        adaptive_threshold = min(adaptive_threshold, 100.0)  # 최대 100도

        # 최적 매칭 결과 처리
        for i, j in zip(current_indices, prev_indices):
            cost = cost_matrix[i, j]

            # 적응형 임계값 확인
            if cost < adaptive_threshold:
                prev_id_to_match = prev_ids[j]
                current_line_data = current_lines[i]

                # 소실되었던 선이 다시 발견된 경우
                if prev_id_to_match in self.lost_lines:
                    self.lines[prev_id_to_match] = self.lost_lines[prev_id_to_match]
                    del self.lost_lines[prev_id_to_match]

                # 기존 선 업데이트
                self.lines[prev_id_to_match].update(current_line_data)

                # 매칭 성공 시 lost_frames 리셋
                self.lines[prev_id_to_match].lost_frames = 0

                matched_current_indices.add(i)
                matched_prev_ids.add(prev_id_to_match)

        # 매칭되지 않은 현재 선 -> 새 선으로 추가
        for i, line_data in enumerate(current_lines):
            if i not in matched_current_indices:
                new_id = self.next_line_id
                self.lines[new_id] = TrackedLine(new_id)
                self.lines[new_id].update(line_data)
                self.next_line_id += 1

        # 매칭되지 않은 이전 선 처리
        unmatched_prev_ids = set(prev_ids) - matched_prev_ids
        for line_id in unmatched_prev_ids:
            # 현재 추적 중인 선이면 lost_lines로 이동
            if line_id in self.lines:
                line = self.lines[line_id]
                line.lost_frames += 1

                if line.lost_frames <= self.max_lost_frames:
                    # 아직 완전히 소실되지 않음 - lost_lines로 이동
                    self.lost_lines[line_id] = line
                # else: 완전히 소실 - 삭제

                del self.lines[line_id]

            # lost_lines에 있던 선이 다시 매칭되지 않음
            elif line_id in self.lost_lines:
                line = self.lost_lines[line_id]
                line.lost_frames += 1

                if line.lost_frames > self.max_lost_frames:
                    # 완전히 소실 - lost_lines에서도 삭제
                    del self.lost_lines[line_id]

    def _calculate_average_frame_angle_change(self) -> float | None:
        """여러 선의 직전 프레임과의 각도 변화 평균 계산
        개소 별 각도 왜곡을 보정하기 위하여 angle_scale_factor 곱함.
        """
        valid_angle_changes = []

        for line in self.lines.values():
            if line.previous_angle_change is not None:
                angle_change = line.previous_angle_change

                # 올바른 방향의 각도 변화만 사용
                change_sign = 1 if angle_change > 0 else -1 if angle_change < 0 else 0
                # 방향이 맞지 않으면 마진 체크
                if change_sign != 0 and change_sign != self.rotation_direction:
                    # 반대 방향이지만 마진 범위 내인지 확인
                    if abs(angle_change) > self.opposite_direction_margin:
                        # 마진을 벗어나면 제외
                        continue

                # 개소 별 각도 왜곡 보정
                angle_change *= self.rotation_config.angle_scale_factor
                valid_angle_changes.append(angle_change)

        if not valid_angle_changes:
            return None

        # 이상치 제거 (IQR 방법)
        if len(valid_angle_changes) > 3:
            q1 = np.percentile(valid_angle_changes, 25)
            q3 = np.percentile(valid_angle_changes, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered_angle_changes = [v for v in valid_angle_changes if lower_bound <= v <= upper_bound]

            if filtered_angle_changes:
                valid_angle_changes = filtered_angle_changes

        # 단순 평균
        integrated_angle_change = abs(np.mean(valid_angle_changes).item())

        return integrated_angle_change

    def get_result(self, num_frames: int | None = None) -> float:
        """누적 회전 각도 반환 (절댓값)

        Args:
            num_frames (int | None, optional): 고려할 프레임 수. None인 경우 전체 히스토리 사용. Defaults to None.
        """
        if len(self.average_frame_angle_changes) < 1:
            return -1.0

        if num_frames is not None:
            if len(self.average_frame_angle_changes) < num_frames:
                return -1.0
            else:
                recent_changes = self.average_frame_angle_changes[-num_frames:]
                return abs(sum(recent_changes))

        return abs(sum(self.average_frame_angle_changes))

    @property
    def num_lines(self) -> int:
        """현재 추적 중인 선의 수 반환"""
        return len(self.lines)

    def get_rotation_direction(self) -> int:
        """현재 회전 방향 반환

        Returns:
            1: 시계방향, -1: 반시계방향, 0: 미정
        """
        return self.rotation_direction

    def get_rotation_direction_str(self) -> str:
        """현재 회전 방향을 문자열로 반환"""
        if self.rotation_direction == 1:
            return "Clockwise"
        elif self.rotation_direction == -1:
            return "Counter-Clockwise"
        else:
            return "Undetermined"

    def reset(self) -> None:
        """트래커 상태 초기화"""
        self.lines.clear()
        self.lost_lines.clear()
        self.next_line_id = 0
        self.average_frame_angle_changes.clear()

        # 중심 좌표 초기화
        if self.center_reset:
            self.center = None

        # 프레임 정보 초기화
        self.previous_frame = None

    def get_tracked_lines_info(self) -> list[dict]:
        """현재 추적 중인 모든 선의 정보 반환 (시각화용)

        Returns:
            list of dicts with keys: 'line_id', 'center', 'angle', 'line_coords'
        """
        lines_info = []
        for line_id, line in self.lines.items():
            if line.positions:
                last_pos = line.positions[-1]
                # last_pos: (mid_x, mid_y, angle, x1, y1, x2, y2, length)
                lines_info.append(
                    {
                        "line_id": line_id,
                        "center": (last_pos[0], last_pos[1]),
                        "angle": last_pos[2],
                        "line_coords": (
                            (last_pos[3], last_pos[4], last_pos[5], last_pos[6])
                            if len(last_pos) >= 7
                            else None
                        ),
                    }
                )
        return lines_info


class TrackedLine:
    """개별 추적 선 클래스

    기존 Marker 클래스와 동일한 구조를 가집니다.
    """

    def __init__(self, line_id: int):
        """
        TrackedLine 초기화

        Args:
            line_id: 선 고유 ID
        """
        self.line_id = line_id
        # 위치 히스토리: (mid_x, mid_y, angle, x1, y1, x2, y2, length)
        self.positions: list[tuple] = []

        # 현재 상태
        self.last_angle = None
        self.lost_frames: int = 0

        # 초기 각도 (선이 처음 등장했을 때의 각도)
        self.initial_angle = None
        self.previous_angle_change = None  # 최근 프레임에서의 각도 변화량

    def update(self, line_data: dict) -> None:
        """선 위치 업데이트"""
        mid_x, mid_y = line_data["center"]
        angle = line_data["angle"]
        x1, y1, x2, y2 = line_data["line_coords"]
        length = line_data.get("length", 0)

        position = (mid_x, mid_y, angle, x1, y1, x2, y2, length)
        self.positions.append(position)

        # 첫 번째 업데이트인 경우 초기 각도 설정
        if self.initial_angle is None:
            self.initial_angle = angle
            self.previous_angle_change = None
        else:
            # 이전 각도와의 차이 계산
            angle_diff = angle - self.last_angle

            # 각도 차이가 너무 크면 회전 방향이 바뀐 것으로 간주
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            # 누적 각도 변화량 업데이트
            self.previous_angle_change = angle_diff

        self.last_angle = angle

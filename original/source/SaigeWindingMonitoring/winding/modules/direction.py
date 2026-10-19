from __future__ import annotations

from collections import deque
from enum import Enum

import cv2
import numpy as np

from .base import BaseMonitor
from .drum import Drum
from ..data.data_class import SegmentationResults
from ..engine.config import MonitoringConfig
from ..utils.corner import detect_filtered_corner_point
from ..utils.smoothing import smooth_ema, filter_outlier
from ..utils.mask import downscale_mask, flatten_mask_edges, rotate_mask_for_flattened_top_line
from ..utils.protrusion import check_edge_protrusion_by_height_difference
from ..utils.geometry import find_winding_top_x


class Direction(str, Enum):
    UNKNOWN = "unknown"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"

    def opposite(self) -> Direction:
        if self.is_left_to_right():
            return Direction.RIGHT_TO_LEFT
        elif self.is_right_to_left():
            return Direction.LEFT_TO_RIGHT
        return Direction.UNKNOWN

    def is_left_to_right(self) -> bool:
        return self is Direction.LEFT_TO_RIGHT

    def is_right_to_left(self) -> bool:
        return self is Direction.RIGHT_TO_LEFT

    def is_unknown(self) -> bool:
        return self is Direction.UNKNOWN

    @staticmethod
    def from_str(value: str | None) -> Direction:
        if value is None:
            return Direction.UNKNOWN
        if value == Direction.LEFT_TO_RIGHT.value:
            return Direction.LEFT_TO_RIGHT
        if value == Direction.RIGHT_TO_LEFT.value:
            return Direction.RIGHT_TO_LEFT
        return Direction.UNKNOWN

    def to_str(self) -> str | None:
        if self.is_unknown():
            return None
        return self.value


class DirectionMonitor(BaseMonitor):
    """드럼의 진행 방향 및 진행률을 모니터링하는 클래스."""

    def __init__(
        self,
        monitoring_config: MonitoringConfig,
    ) -> None:
        super().__init__(monitoring_config)
        self.reset()

    def process_frame(
        self,
        segmentation_results: SegmentationResults,
        state: str | None = None,
        rotation_angle: float = 0.0,
    ) -> dict:
        """드럼의 진행 방향과 진행률을 계산합니다.
        또한 권취 중인 전선이 양극단에 도달하여 드럼 이동 방향 전환이 필요한지 여부를 판단합니다.

        Args:
            rotation_angle: AngleMonitor에서 계산된 최근 1프레임의 회전 각도
        """
        # 회전 각도 누적
        if rotation_angle > 0:
            self.cumulative_rotation_angle += rotation_angle
        masks = segmentation_results.segmentation_masks
        boxes = segmentation_results.bounding_boxes

        self._set_body_position(segmentation_results.drum_body_box)
        self._update_endpoints(boxes, masks)

        if self._require_direction_setting(boxes):
            self._decide_frame_direction(segmentation_results)

            # direction_history가 direction_confirmation_frames 이상 쌓이고, 모든 값이 동일하다면 해당 값을 사용
            if (
                len(self.direction_history) == self.direction_config.direction_confirmation_frames
            ) and all(direction == self.direction_history[0] for direction in self.direction_history):
                self.cable_moving_direction = self.direction_history[0]
                self.direction_history.clear()

        progress_rate = self._calculate_progress_rate(
            segmentation_results,
            state,
            rotation_angle,
        )

        return {
            "cable_moving_direction": self.cable_moving_direction.to_str(),
            "progress_rate": progress_rate,
        }

    def change_direction(self):
        """드럼의 진행 방향을 변경합니다."""
        if self.cable_moving_direction.is_unknown():
            raise ValueError("현재 드럼의 진행 방향이 설정되지 않았습니다.")

        # 진행률 히스토리 초기화
        self.cable_moving_direction = self.cable_moving_direction.opposite()
        self.progress_rate_history = [0.0]
        self._previous_progress_rate = 0.0

    def _set_body_position(self, body_box: list | None) -> None:
        if body_box is None:
            return
        x, y, w, h = body_box
        cx, cy = x + w // 2, y + h // 2
        self.drum.set_body_center_point([cx, cy])
        self.drum.set_body_left_point([x, cy])
        self.drum.set_body_right_point([x + w, cy])

    def _update_endpoints(
        self, boxes: dict[str, list | None], masks: dict[str, np.ndarray | None]
    ) -> None:
        for key in ["left_drum", "right_drum"]:
            box = boxes[key]
            if box is None:
                continue
            center_y = box[1] + box[3] // 2
            mask = masks[key]
            if key == "left_drum":
                right_x = np.where(mask[center_y, :] == 1)[0][-1]
                self.drum.set_head_right_point([right_x, center_y])
            else:
                left_x = np.where(mask[center_y, :] == 1)[0][0]
                self.drum.set_tail_left_point([left_x, center_y])

    def _get_bottom_region_extreme(
        self,
        drum_mask: np.ndarray,
        drum_box: list,
        choose_max: bool,
    ) -> tuple[int, int]:
        y_threshold = drum_box[1] + drum_box[3] // 2
        ys, xs = np.where(drum_mask == 1)
        valid_mask = ys > y_threshold
        filtered_xs = xs[valid_mask]
        filtered_ys = ys[valid_mask]
        if choose_max:
            index = np.argmax(filtered_xs)
        else:
            index = np.argmin(filtered_xs)
        return filtered_xs[index], filtered_ys[index]

    def _require_direction_setting(self, boxes: dict) -> bool:
        if self.cable_moving_direction.is_unknown():
            drum_body_box = boxes["drum_body"]
            drum_left_box = boxes["left_drum"]
            drum_right_box = boxes["right_drum"]
            if drum_body_box is None or drum_left_box is None or drum_right_box is None:
                return False

            # 드럼 몸통의 높이가 드럼 왼쪽과 오른쪽의 평균 높이보다 크면 드럼 이동 유효 영역 박스를 설정합니다.
            drum_body_height = drum_body_box[3]
            drum_heights = [box[3] for box in [drum_left_box, drum_right_box]]
            average_drum_height = sum(drum_heights) / len(drum_heights)
            return drum_body_height > average_drum_height * self.common_config.body_height_ratio

        return False

    def _get_winding_cable_moving_range(
        self,
        left_drum_mask,
        left_drum_box,
        right_drum_mask,
        right_drum_box,
        winding_cable_box,
        max_trials=10,
    ) -> None:
        """권취 중인 전선의 이동 범위를 계산합니다."""

        left_drum_point = detect_filtered_corner_point(
            left_drum_mask, left_drum_box, self.direction_config.left_drum_corner_angle_range
        )
        right_drum_point = detect_filtered_corner_point(
            right_drum_mask, right_drum_box, self.direction_config.right_drum_corner_angle_range
        )

        if left_drum_point is not None:
            self.winding_cable_moving_range_left_point = left_drum_point
        else:
            if self.cable_moving_direction.is_right_to_left():
                # 날개에 의해 권취 중인 전선이 가려지는 것을 보완하기 위하여 드럼 몸통 아래 가로선과 드럼 왼쪽 마스크의 교차점으로 설정.
                winding_cable_y = winding_cable_box[1]

                left_point_x = None
                left_point_y = winding_cable_y

                for _ in range(max_trials):
                    left_drum_x_candidates = np.where(left_drum_mask[left_point_y, :] == 1)[0]

                    if len(left_drum_x_candidates) > 0:
                        left_point_x = left_drum_x_candidates.max()
                        break

                    left_point_y -= 20

                if left_point_x is None:
                    # left drum 의 하위 마스크 중 x 값이 가장 큰 좌표 선택
                    self.winding_cable_moving_range_left_point = self._get_bottom_region_extreme(
                        left_drum_mask, left_drum_box, choose_max=True
                    )

                else:
                    self.winding_cable_moving_range_left_point = (
                        left_point_x,
                        left_point_y,
                    )

            else:
                # left drum 의 하위 마스크 중 x 값이 가장 큰 좌표 선택
                self.winding_cable_moving_range_left_point = self._get_bottom_region_extreme(
                    left_drum_mask, left_drum_box, choose_max=True
                )

        if right_drum_point is not None:
            self.winding_cable_moving_range_right_point = right_drum_point
        else:
            # right drum 의 하위 마스크 중 x 값이 가장 작은 좌표 선택
            self.winding_cable_moving_range_right_point = self._get_bottom_region_extreme(
                right_drum_mask, right_drum_box, choose_max=False
            )

    def _calculate_progress_rate(
        self,
        segmentation_results: SegmentationResults,
        state: str | None = None,
        rotation_angle: float = 0.0,
    ) -> float:
        winding_cable_mask = segmentation_results.winding_cable_mask

        if (
            winding_cable_mask is None
            or self.drum.head_right_point is None
            or self.drum.tail_left_point is None
            or self.cable_moving_direction.is_unknown()
        ):
            # 드럼의 양 끝점이 아직 계산되지 않음
            if len(self.progress_rate_history) > 0:
                return self.progress_rate_history[-1]
            else:
                return self.direction_config.default_progress_rate

        if state is not None and state in ["first_swing", "second_swing"]:
            # XXX: 드럼 스윙 이동 중에는 진행률을 업데이트 하지 않음.
            if len(self.progress_rate_history) > 0:
                return max(self.progress_rate_history)
            else:
                return self.direction_config.default_progress_rate

        # 1. 권취 전선의 이동 범위 계산.
        self._get_winding_cable_moving_range(
            left_drum_mask=segmentation_results.left_drum_mask,
            left_drum_box=segmentation_results.left_drum_box,
            right_drum_mask=segmentation_results.right_drum_mask,
            right_drum_box=segmentation_results.right_drum_box,
            winding_cable_box=segmentation_results.winding_cable_box,
        )

        # 2. 권취 중인 전선의 위치 기준점 설정
        top_rate = 0.01
        ys, xs = np.where(winding_cable_mask == 1)
        threshold = np.percentile(ys, top_rate * 100)
        top_y_indices = ys <= threshold
        top_x = xs[top_y_indices]
        top_y = ys[top_y_indices]
        winding_cable_x = top_x.mean()
        winding_cable_y = top_y.mean()

        if np.isnan(winding_cable_x):
            # 권취 중인 전선이 없는 경우
            return self.direction_config.default_progress_rate

        moving_range_left_x = self.winding_cable_moving_range_left_point[0]
        moving_range_right_x = self.winding_cable_moving_range_right_point[0]

        if self.cable_moving_direction.is_left_to_right():
            winding_cable_x = top_x.max()
            current_winding_length = winding_cable_x - self.winding_cable_moving_range_left_point[0]
        elif self.cable_moving_direction.is_right_to_left():
            winding_cable_x = top_x.min()
            current_winding_length = self.winding_cable_moving_range_right_point[0] - winding_cable_x
        self.winding_cable_point = [int(winding_cable_x), int(winding_cable_y)]

        total_drum_length = moving_range_right_x - moving_range_left_x

        # 3. 진행률 계산
        progress_rate = min(max(current_winding_length, 0), total_drum_length) / total_drum_length

        # 4. 진행률 히스토리 업데이트
        raw = filter_outlier(
            self.progress_rate_history,
            progress_rate,
            max_jump=self.direction_config.progress_rate_max_change,
        )
        smoothed = smooth_ema(
            self.progress_rate_history, raw, alpha=self.direction_config.progress_rate_ema_alpha
        )
        self.progress_rate_history.append(smoothed)

        if self.progress_rate_history:
            progress_rate = self._correct_progress_rate_by_angle(rotation_angle)
            return progress_rate
        return self.direction_config.default_progress_rate

    def _check_first_layer(self, drum_body_box: np.ndarray) -> bool:
        """첫번째 층 권취 진행 중인지 확인 (드럼 몸통이 아직 좌우에 가득차지 않은 경우)
        아직 첫번째 층 권취가 진행 중이면 True를 반환합니다. 그 외의 경우 False를 반환합니다.
        """
        drum_body_frame_width = self.drum.tail_left_point[0] - self.drum.head_right_point[0]
        _, _, body_w, _ = drum_body_box
        is_first_layer = body_w < drum_body_frame_width * self.direction_config.first_layer_width_ratio
        return is_first_layer

    def _decide_frame_direction(
        self,
        segmentation_results: SegmentationResults,
    ) -> None:
        drum_body_box = segmentation_results.drum_body_box

        _is_first_layer = self._check_first_layer(drum_body_box)
        _is_winding_cable_at_edge, roi_side = self.is_winding_cable_at_edge(segmentation_results)

        # 1) 첫번째 바퀴 권취 중인 경우
        if _is_first_layer:
            cable_moving_direction = self._decide_direction_at_first_layer(segmentation_results)

        # 2) 극단 부에서 권취 중인 경우 - 극단 부의 경우, 위층 여부를 바탕으로 방향 결정
        elif _is_winding_cable_at_edge:
            cable_moving_direction = self._decide_direction_at_edge(segmentation_results, roi_side)

        # 3) 두번째 이상 바퀴
        else:
            cable_moving_direction = self._decide_direction_after_first_layer(segmentation_results)

        if cable_moving_direction is not None:
            self.direction_history.append(cable_moving_direction)

    def is_winding_cable_at_edge(
        self, segmentation_results: SegmentationResults, edge_ratio: float = 0.9
    ) -> tuple[bool, str | None]:
        """권취 중인 전선이 드럼의 좌/우 극단 근처에 위치하는지 판정합니다.

        edge_ratio가 0.9라면 좌측 10% 또는 우측 10% 구간에 위치하면 True를 반환합니다.
        즉, 유효 이동 구간의 바깥쪽 (1 - edge_ratio) 비율이 극단 영역입니다.
        """
        winding_cable_box = segmentation_results.winding_cable_box
        if (
            winding_cable_box is None
            or self.drum.head_right_point is None
            or self.drum.tail_left_point is None
        ):
            return False, None
        winding_cable_center_x = winding_cable_box[0] + winding_cable_box[2] // 2

        left_bound_x = self.drum.head_right_point[0]
        right_bound_x = self.drum.tail_left_point[0]
        total_range = right_bound_x - left_bound_x
        if total_range <= 0:
            return False, None

        margin_ratio = max(min(1.0 - edge_ratio, 1.0), 0.0)
        left_threshold = left_bound_x + total_range * margin_ratio
        right_threshold = right_bound_x - total_range * margin_ratio

        if winding_cable_center_x <= left_threshold:
            return True, "left"
        if winding_cable_center_x >= right_threshold:
            return True, "right"
        return False, None

    def _decide_direction_at_first_layer(self, segmentation_results: SegmentationResults) -> Direction:
        """첫번째 바퀴 권취 중인 경우, 방향 결정"""
        drum_body_box = segmentation_results.drum_body_box
        body_x, _, body_w, _ = drum_body_box

        distance_to_left = max(body_x - self.drum.head_right_point[0], 0)
        distance_to_right = max(self.drum.tail_left_point[0] - (body_x + body_w), 0)
        if distance_to_left < distance_to_right:
            cable_moving_direction = Direction.LEFT_TO_RIGHT
        else:
            cable_moving_direction = Direction.RIGHT_TO_LEFT
        return cable_moving_direction

    def _decide_direction_after_first_layer(
        self, segmentation_results: SegmentationResults
    ) -> Direction | None:
        drum_body_mask = segmentation_results.drum_body_mask
        drum_body_box = segmentation_results.drum_body_box
        boxes = segmentation_results.bounding_boxes

        winding_cable_x = find_winding_top_x(segmentation_results.winding_cable_mask)

        rotated_body_mask, rotated_body_box = rotate_mask_for_flattened_top_line(
            drum_body_mask, drum_body_box, None
        )

        body_x, body_y, body_w, body_h = rotated_body_box

        margin = self.common_config.roi_narrowing_width
        drum_body_roi = rotated_body_mask[
            body_y - margin : body_y + margin + body_h,
            body_x - margin : body_x + margin + body_w,
        ]

        roi_winding_cable_x = winding_cable_x - body_x + margin  # ROI에 맞춰 winding_cable_x 조정
        drum_body_roi = downscale_mask(drum_body_roi, self.common_config.downscale_factor)
        scaled_winding_x = (
            roi_winding_cable_x // self.common_config.downscale_factor
        )  # downscale에 맞춰 winding_cable_x 조정
        drum_body_roi = flatten_mask_edges(drum_body_roi, split_x=scaled_winding_x)

        if np.sum(drum_body_roi) == 0:
            # Downscale 이후 Segmentation Mask가 소멸됨
            return None

        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(drum_body_roi)
        left_y_indices = np.where(drum_body_roi[:, roi_x] == 1)
        right_y_indices = np.where(drum_body_roi[:, roi_x + roi_w - 1] == 1)

        if len(left_y_indices[0]) == 0 or len(right_y_indices[0]) == 0:
            return None
        left_top_y, left_bottom_y = left_y_indices[0][[0, -1]]
        right_top_y, right_bottom_y = right_y_indices[0][[0, -1]]

        left_height = left_bottom_y - left_top_y
        right_height = right_bottom_y - right_top_y

        # 3-1) 두번째 바퀴 이후, 드럼이 오른쪽에서 왼쪽으로 권취되는 경우 (상하 조건 모두 만족)
        #   <<==진행===   #
        #       -------- #
        # -----/       | #
        # |   드럼 몸통  | #
        # -----\       | #
        #       -------- #
        #   <<==진행===   #
        if right_height > left_height + 1:
            cable_moving_direction = Direction.RIGHT_TO_LEFT
            return cable_moving_direction

        # 3-2) 두번째 바퀴 이후, 드럼이 왼쪽에서 오른쪽으로 권취되는 경우 (상하 조건 모두 만족)
        #   ===진행===>>   #
        #   --------      #
        #   |       \---  #
        #   |  드럼 몸통 |  #
        #   |       /---  #
        #   --------      #
        #   ===진행===>>   #
        elif left_height > right_height + 1:
            cable_moving_direction = Direction.LEFT_TO_RIGHT
            return cable_moving_direction

        # 3-3) first_layer_width_ratio를 넘었으나 층이 발생하지 않은 경우, 권취 중인 전선의 위치를 바탕으로 결정
        winding_cable_box = boxes["winding_cable"]
        if winding_cable_box is None:
            return None
        winding_cable_center_x = winding_cable_box[0] + winding_cable_box[2] // 2
        distance_to_left = max(winding_cable_center_x - self.drum.head_right_point[0], 0)
        distance_to_right = max(self.drum.tail_left_point[0] - winding_cable_center_x, 0)
        if distance_to_left < distance_to_right:
            cable_moving_direction = Direction.RIGHT_TO_LEFT
        else:
            cable_moving_direction = Direction.LEFT_TO_RIGHT
        return cable_moving_direction

    def _decide_direction_at_edge(
        self, segmentation_results: SegmentationResults, roi_side: str
    ) -> Direction:
        """극단 부에서 권취 중인 경우, 위층 여부를 바탕으로 방향 결정"""
        if check_edge_protrusion_by_height_difference(
            segmentation_results,
            roi_side,
            self.common_config.roi_narrowing_width,
            self.common_config.edge_region_ratio,
            self.common_config.downscale_factor,
            self.direction_config.min_pixel_for_protrusion,
        ):
            cable_moving_direction = (
                Direction.LEFT_TO_RIGHT if roi_side == "left" else Direction.RIGHT_TO_LEFT
            )
        else:
            cable_moving_direction = (
                Direction.RIGHT_TO_LEFT if roi_side == "left" else Direction.LEFT_TO_RIGHT
            )
        return cable_moving_direction

    def _correct_progress_rate_by_angle(
        self,
        rotation_angle: float,
    ) -> float:
        """
        각도 회전량을 고려하여 진행률이 점진적으로 증가하도록 보정합니다.

        1. 현재까지의 진행률 변화량과 각도 회전량을 바탕으로 각도 당 진행률 변화량을 calibration합니다.
        2. Calibration 이후 프레임부터는 각도 회전량을 바탕으로 진행률을 보정합니다.
        3. SEG 기반 진행률과 각도 기반 진행률 중 더 큰 값을 최종 진행률로 사용합니다.

        Args:
            rotation_angle: AngleMonitor에서 계산된 최근 1프레임의 회전 각도
        """
        seg_based_progress_rate = max(self.progress_rate_history)

        # Calibration이 완료되지 않은 경우
        if not self._is_angle_calibration_ready():
            self._calibrate_angle_progress_ratio(seg_based_progress_rate)
            return seg_based_progress_rate

        # 진행률이 보정 시작 임계값 이하인 경우
        if seg_based_progress_rate <= self.direction_config.correct_progress_rate_start:
            return seg_based_progress_rate

        # Calibration이 완료된 경우: 각도 기반 진행률 계산
        angle_based_progress_rate = self._calculate_angle_based_progress_rate(rotation_angle)

        # SEG 기반과 각도 기반 진행률 중 더 큰 값을 사용
        corrected_progress_rate = max(seg_based_progress_rate, angle_based_progress_rate)
        self._previous_progress_rate = corrected_progress_rate

        return corrected_progress_rate

    def _is_angle_calibration_ready(self) -> bool:
        """각도 기반 진행률 보정을 위한 calibration이 완료되었는지 확인합니다."""
        return self._angle_calibration_ratio is not None

    def _calibrate_angle_progress_ratio(
        self,
        current_seg_progress_rate: float,
        min_progress_rate_increase: float = 0.01,
    ) -> None:
        """각도당 진행률 변화량을 calibration합니다.

        Args:
            current_seg_progress_rate: 현재 SEG 기반 진행률
            min_progress_rate_increase: calibration을 위한 최소 진행률 증가량
        """
        # 첫 번째 호출: 기준 진행률 저장
        if self._previous_progress_rate is None:
            self._previous_progress_rate = current_seg_progress_rate
            return

        # 진행률 변화량 계산
        progress_rate_change = current_seg_progress_rate - self._previous_progress_rate

        # 충분한 진행률 변화가 있는 경우에만 calibration 수행
        if progress_rate_change > min_progress_rate_increase:
            if self.cumulative_rotation_angle > 0:
                # 각도당 진행률 변화량 계산 (상한값 제한)
                calibration_ratio = progress_rate_change / self.cumulative_rotation_angle
                self._angle_calibration_ratio = min(
                    calibration_ratio, self.common_config.max_calibration_ratio
                )

    def _calculate_angle_based_progress_rate(
        self,
        rotation_angle: float,
    ) -> float:
        """각도 회전량을 바탕으로 진행률을 계산합니다.

        Args:
            rotation_angle: AngleMonitor에서 계산된 최근 1프레임의 회전 각도

        Returns:
            각도 기반으로 계산된 진행률 (0.0 ~ 1.0)
        """
        # 이전 진행률 + (회전 각도 × calibration 비율)
        angle_based_progress_rate = (
            self._previous_progress_rate + rotation_angle * self._angle_calibration_ratio
        )

        # 진행률은 최대 1.0을 넘을 수 없음
        return min(angle_based_progress_rate, 1.0)

    def reset(self) -> None:
        """Reset the engine state."""
        self.drum = Drum()
        self.progress_rate_history: deque[float] = deque(
            maxlen=self.direction_config.progress_rate_history_length
        )
        self.cable_moving_direction: Direction = Direction.UNKNOWN
        self.winding_cable_moving_range_left_point: list[int] | None = None
        self.winding_cable_moving_range_right_point: list[int] | None = None
        self.winding_cable_point: list[int] | None = None
        self.direction_history: deque[Direction] = deque(
            maxlen=self.direction_config.direction_confirmation_frames
        )

        # 각도 기반 진행률 보정을 위한 변수
        self._angle_calibration_ratio: float | None = None  # 각도당 진행률 변화량
        self._previous_progress_rate: float | None = None  # 이전 프레임의 보정된 진행률
        self.cumulative_rotation_angle: float = 0.0  # 누적 회전 각도

    @property
    def monitoring_options(self) -> dict:
        """Get current monitoring options."""
        return {}

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        """Set monitoring options, dynamically updating config attributes."""
        params = options.get("params", {})
        self.update_config_from_params(params)

    @property
    def default_monitoring_options(self) -> dict:
        """Get default monitoring options."""
        return {}

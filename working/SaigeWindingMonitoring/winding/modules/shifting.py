from collections import deque

import numpy as np

from ..data.data_class import SegmentationResults, SideSegmentationResults
from ..engine.config import MonitoringConfig
from ..error import (
    MonitoringOptionShiftAlarmIntervalTypeError,
    MonitoringOptionShiftAlarmIntervalValueError,
    MonitoringOptionShiftSensitivityTypeError,
    MonitoringOptionShiftSensitivityValueError,
)
from ..utils.mask import (
    calculate_rotation_angle_for_flattened_top_line,
    rotate_mask_for_flattened_top_line,
)
from ..utils.roi import (
    get_roi_side,
    crop_edge_roi,
)
from ..utils.geometry import find_winding_top_x
from . import logging
from .base import BaseMonitor
from .direction import Direction
from .process_state import ProcessState
from .score import ShiftScoreCalculator, SwingScoreCalculator, TighteningScoreCalculator
from .shift_state import (
    AutoSwingSkipResetState,
    BaseState,
    FirstSwingState,
    FirstWrappingState,
    SecondSwingState,
    SwingTermState,
    TighteningState,
    WindingState,
)

MILLISECONDS_IN_SECOND = 1000


def exponential_moving_average(prev_value: float | None, value: float | int, alpha: float = 0.5):
    if prev_value is None:
        return value

    return (1 - alpha) * prev_value + alpha * value


class ShiftMonitor(BaseMonitor):
    def __init__(
        self,
        monitoring_config: MonitoringConfig,
    ) -> None:
        super().__init__(monitoring_config)
        self.drum_shift_score_calculator = ShiftScoreCalculator(
            default_shifting_score=self.shift_config.default_shifting_score,
            shifting_box_ratio=self.shift_config.shifting_box_ratio,
            right_over_left_expansion_ratio=self.shift_config.right_over_left_expansion_ratio,
        )
        self.swing_score_calculator = SwingScoreCalculator(
            default_swing_score=self.shift_config.default_tightening_score,
            swing_range_factors=self.shift_config.swing_range_factors,
        )
        self.tightening_score_calculator = TighteningScoreCalculator(
            default_tightening_score=self.shift_config.default_tightening_score,
            tightening_range_factor=self.shift_config.tightening_range_factor,
        )

        self.reset()

        self.target_swing_score = self.shift_config.target_swing_score
        # mapping
        self.direction_to_tightening_score_threshold = {
            # XXX: 방향이 전환되기 전임. (tightening 후에 방향이 전환됨.)
            Direction.LEFT_TO_RIGHT: self.shift_config.tightening_score_thresholds[1],
            Direction.RIGHT_TO_LEFT: self.shift_config.tightening_score_thresholds[0],
        }
        self.direction_to_swing_angle_threshold = {
            Direction.LEFT_TO_RIGHT: self.shift_config.angle_thresholds[0],
            Direction.RIGHT_TO_LEFT: self.shift_config.angle_thresholds[1],
        }
        self.direction_to_protrusion_angle_threshold = {
            Direction.LEFT_TO_RIGHT: self.shift_config.protrusion_angle_thresholds[0],
            Direction.RIGHT_TO_LEFT: self.shift_config.protrusion_angle_thresholds[1],
        }
        self.direction_to_swing_score_threshold = {
            Direction.LEFT_TO_RIGHT: self.shift_config.swing_score_thresholds[0],
            Direction.RIGHT_TO_LEFT: self.shift_config.swing_score_thresholds[1],
        }
        self.direction_to_swing_term_angle = {
            Direction.LEFT_TO_RIGHT: self.shift_config.swing_term_angles[0],
            Direction.RIGHT_TO_LEFT: self.shift_config.swing_term_angles[1],
        }
        self.direction_to_reset_delay_angle = {
            Direction.LEFT_TO_RIGHT: self.shift_config.reset_delay_angles[0],
            Direction.RIGHT_TO_LEFT: self.shift_config.reset_delay_angles[1],
        }

        self.previous_monitoring_result = self._get_default_monitoring_results()

        # 초기 박스 생성 시, 이미 진행 중인 상태라면 회전량을 기다리지 않고 진행률 기반으로 처리
        self.is_mid_start_progress_threshold = self.shift_config.is_mid_start_progress_threshold

    def _get_default_monitoring_results(self) -> dict:
        return {
            "drum_shifting_score": self.shift_config.default_shifting_score,
            "tightening_score": self.shift_config.default_tightening_score,
            "drum_swing_score": self.shift_config.default_tightening_score,
            "require_drum_shift": False,
            "cable_shifting_box": self.drum_shift_score_calculator.shifting_box_out,
            "drum_moving_direction": None,
            "require_direction_change": False,
            "current_state": None,
            "process_state": self.process_state.value,
            "rotation_angle": (
                self.cumulative_rotation_angle
                if self.process_state in self.return_rotation_angle_state
                else None
            ),
            "reference_angle": (
                self.reference_angle if self.process_state in self.return_rotation_angle_state else None
            ),
            "cumulative_rotation_angle": self.cumulative_rotation_angle,
        }

    def process_frame(
        self,
        segmentation_results: SegmentationResults,
        direction_monitoring_results: dict,
        side_segmentation_results: SideSegmentationResults,
        rotation_angle: float = 0.0,
    ) -> dict:
        """권취 중인 전선이 임계 영역을 벗어났는지 여부를 바탕으로 드럼 이동 여부를 판단합니다.

        Args:
            segmentation_results (SegmentationResults): SEG inference handler에서 얻은 추론 결과입니다.
            direction_monitoring_results (dict): DirectionMonitor에서 얻은 추론 결과입니다.
            rotation_angle (float): AngleMonitor에서 계산된 최근 1프레임의 회전 각도

        Returns:
            dict: Monitoring result containing the drum shifting score and valid region center point
        """
        self._update_rotation_angle(rotation_angle)
        logging.log(f"[state={self.state.name}]")

        # Initialization
        require_drum_shift = False
        require_direction_change = False

        # Find valid region center point
        winding_cable_box = segmentation_results.winding_cable_box
        cable_moving_direction: Direction = Direction.from_str(
            direction_monitoring_results["cable_moving_direction"]
        )
        progress_rate = direction_monitoring_results["progress_rate"]

        if winding_cable_box is None or cable_moving_direction.is_unknown():
            return self._get_default_monitoring_results()

        # 1. [Winding] 일반 드럼 이동
        drum_shifting_score, require_drum_shift = self._calculate_drum_shifting(
            segmentation_results, winding_cable_box, cable_moving_direction, progress_rate
        )

        paused_result = self._handle_rotation_pause(rotation_angle, drum_shifting_score)
        if paused_result is not None:
            return paused_result

        # 2. [Tightening] 극단 밀착 점수 계산
        tightening_score = self.tightening_score_calculator.calculate_score(
            segmentation_results, cable_moving_direction, self.state
        )

        # Update cumulative average for auto swing skip (2 frames after FirstWrappingState)
        if segmentation_results.winding_cable_mask is not None:
            if isinstance(self.state, FirstWrappingState):
                self._first_wrapping_frame_count += 1
                # Start updating average after 2 frames
                if self._first_wrapping_frame_count >= 2:
                    self.swing_score_calculator.update_cumulative_avg_top_x(
                        segmentation_results.winding_cable_mask
                    )
                    # Calculate cable width for auto swing skip threshold
                    self.swing_score_calculator.calculate_cable_width_from_mask(
                        segmentation_results.winding_cable_mask, use_history=True
                    )

        # 3. [FirstWrapping] 돌출부 상태 업데이트 및 스윙 준비
        is_first_swing_ready = self._prepare_first_swing(segmentation_results, cable_moving_direction)

        # 4. [Tightening / FirstWrapping] 드럼 이동
        require_drum_shift, drum_moving_direction = self._check_side_drum_shift_requirements(
            require_drum_shift,
            tightening_score,
            segmentation_results,
            cable_moving_direction,
            is_first_swing_ready,
        )

        # 5. [FirstSwing] 드럼 스윙 점수 계산
        drum_swing_score, drum_swing_excess_travel = (
            self.swing_score_calculator.calculate_score_and_excess_travel(
                segmentation_results.winding_cable_mask,
                segmentation_results.drum_body_box,
                cable_moving_direction,
                self.state,
                segmentation_results.bounding_boxes,
                segmentation_results.segmentation_masks,
            )
        )

        # 6. [FirstSwing, SwingTerm, SecondSwing] 드럼 스윙 이동
        require_drum_shift, drum_moving_direction = self._check_side_drum_swing(
            drum_swing_score,
            drum_swing_excess_travel,
            require_drum_shift,
            cable_moving_direction,
            drum_moving_direction,
        )

        # 7. [Winding] 밀착 준비 감지를 위한 distance_ema 계산
        distance_ema, cable_width = self._calculate_tightening_distance(
            segmentation_results, cable_moving_direction
        )

        transition_events = self.handle_state_transition(
            {
                "progress_rate": progress_rate,
                "segmentation_results": segmentation_results,
                "side_segmentation_results": side_segmentation_results,
                "cable_moving_direction": cable_moving_direction,
                "tightening_score": tightening_score,
                "is_first_swing_ready": is_first_swing_ready,
                "drum_swing_score": drum_swing_score,
                "distance_ema": distance_ema,
                "cable_width": cable_width,
            },
        )

        require_drum_shift, drum_moving_direction, require_direction_change = (
            self._apply_transition_events(
                transition_events, require_drum_shift, drum_moving_direction, require_direction_change
            )
        )

        monitoring_result = self._build_monitoring_result(
            drum_shifting_score=drum_shifting_score,
            tightening_score=tightening_score,
            drum_swing_score=drum_swing_score,
            require_drum_shift=require_drum_shift,
            drum_moving_direction=drum_moving_direction,
            require_direction_change=require_direction_change,
        )

        self.previous_monitoring_result = monitoring_result

        logging.log(f"[state={self.state.name}]")
        return monitoring_result

    def _update_rotation_angle(self, rotation_angle: float) -> None:
        self.rotation_angle = rotation_angle
        if rotation_angle > 0:
            self.cumulative_rotation_angle += rotation_angle
            if (
                isinstance(self.state, WindingState)
                and self.drum_shift_score_calculator.shifting_box_in is None
            ):
                self._initial_shifting_box_rotation_angle += rotation_angle
                self.rotation_angle_history.append(rotation_angle)

    def _calculate_drum_shifting(
        self,
        segmentation_results: SegmentationResults,
        winding_cable_box: list | tuple | np.ndarray,
        cable_moving_direction: Direction,
        progress_rate: float,
    ) -> tuple[float, bool]:
        drum_shifting_score = self.drum_shift_score_calculator.calculate_score(
            winding_cable_box, cable_moving_direction
        )
        require_drum_shift, require_box_reset = self._check_drum_shift_and_box_reset(
            segmentation_results.bounding_boxes, drum_shifting_score, progress_rate
        )
        if require_box_reset:
            self.drum_shift_score_calculator.set_shifting_box(winding_cable_box)

        return drum_shifting_score, require_drum_shift

    def _handle_rotation_pause(self, rotation_angle: float, drum_shifting_score: float) -> dict | None:
        if rotation_angle != 0:
            return None

        # 생산이 멈춘 경우(드럼 회전이 멈춤), 동작 중지. 단, 이동 박스를 그리기 위해 winding 로직은 동작 시킴.
        logging.log("[Pause] Drum rotation is stopped. All actions are paused.")
        self.previous_monitoring_result["require_drum_shift"] = False
        self.previous_monitoring_result["require_direction_change"] = False
        self.previous_monitoring_result["cable_shifting_box"] = (
            self.drum_shift_score_calculator.shifting_box_out
        )
        self.previous_monitoring_result["drum_shifting_score"] = drum_shifting_score

        if isinstance(self.state, FirstSwingState):
            # XXX: 스윙 중에 멈췄다면, 동작 중지 직전에(shift_strength_second)동안 보낸 드럼 이동 횟수
            avg_rotation_angle = (
                np.mean(self.rotation_angle_history) if len(self.rotation_angle_history) > 0 else 0
            )
            self.additional_swing_delay_angle = (
                avg_rotation_angle * sum(self.state.shift_alarm_history) * self.shift_strength_second
            )

        return self.previous_monitoring_result

    def _prepare_first_swing(
        self,
        segmentation_results: SegmentationResults,
        cable_moving_direction: Direction,
    ) -> bool:
        self._update_first_wrapping_protrusion_state(segmentation_results, cable_moving_direction)
        roi_side = get_roi_side(self.state.name, cable_moving_direction)
        return self._is_first_swing_ready(cable_moving_direction, segmentation_results, roi_side)

    def _apply_transition_events(
        self,
        transition_events: dict,
        require_drum_shift: bool,
        drum_moving_direction: Direction,
        require_direction_change: bool,
    ) -> tuple[bool, Direction, bool]:
        # AutoSwingSkipResetState에서 드럼 이동 요청 처리
        if isinstance(self.state, AutoSwingSkipResetState):
            if transition_events.get("require_drum_shift"):
                require_drum_shift = True
                drum_moving_direction = transition_events.get("drum_moving_direction", Direction.UNKNOWN)

        require_direction_change = transition_events.get(
            "require_direction_change", require_direction_change
        )
        return require_drum_shift, drum_moving_direction, require_direction_change

    def _build_monitoring_result(
        self,
        drum_shifting_score: float,
        tightening_score: float,
        drum_swing_score: float,
        require_drum_shift: bool,
        drum_moving_direction: Direction,
        require_direction_change: bool,
    ) -> dict:
        shift_strength = (
            self.shift_config.default_swing_shift_strength
            if self.process_state in (ProcessState.FIRST_SWING, ProcessState.SECOND_SWING)
            else self.shift_strength
        )
        return {
            "drum_shifting_score": drum_shifting_score,
            "tightening_score": tightening_score,
            "drum_swing_score": drum_swing_score,
            "require_drum_shift": require_drum_shift,
            "cable_shifting_box": self.drum_shift_score_calculator.shifting_box_out,
            "drum_moving_direction": drum_moving_direction.to_str(),
            "require_direction_change": require_direction_change,
            "current_state": self.state.name,
            # XXX: UI 로그 용도
            "process_state": self.process_state.value,  # current_state보다 세분화 되어있음.
            "shift_strength": shift_strength,
            "rotation_angle": (
                self.cumulative_rotation_angle
                if self.process_state in self.return_rotation_angle_state
                and self.reference_angle is not None
                else None
            ),
            "reference_angle": (
                self.reference_angle if self.process_state in self.return_rotation_angle_state else None
            ),
            # XXX: Debugging 용도 - 연구소만 사용
            "cumulative_rotation_angle": self.cumulative_rotation_angle,
        }

    def set_state(self, new_state: BaseState) -> None:
        # FirstWrappingState로 전환 시 프레임 카운터 초기화
        if isinstance(new_state, FirstWrappingState):
            self._first_wrapping_frame_count = 0
        self.state = new_state

    def set_process_state(self, new_process_state: str) -> None:
        self.process_state = new_process_state

    def record_rotation_angle(self, new_rotation_angle: float) -> None:
        self.cumulative_rotation_angle = new_rotation_angle

    def set_reference_angle(self, new_reference_angle: float | None) -> None:
        self.reference_angle = new_reference_angle

    def _check_drum_shift_and_box_reset(
        self, boxes: dict, drum_shifting_score: float, progress_rate: float
    ) -> tuple[bool, bool]:
        """드럼 이동 유효 영역 박스를 설정해야 하는지 여부를 확인합니다.

        권취 작업 시작 단계에서는 전선이 한 바퀴 정도 감긴 후,
        드럼 양옆의 높이와 전선이 감긴 몸통의 높이가 비슷(body_height_ratio)해졌을 때 드럼 이동을 시작합니다.
        """
        drum_body_box = boxes["drum_body"]
        drum_left_box = boxes["left_drum"]
        drum_right_box = boxes["right_drum"]
        if (
            drum_body_box is None
            or drum_left_box is None
            or drum_right_box is None
            or not isinstance(self.state, WindingState)
        ):
            require_drum_shift, require_box_reset = False, False

            return require_drum_shift, require_box_reset

        # (1) 드럼 이동 유효 영역 박스가 설정되어 있지 않은 경우
        if self.drum_shift_score_calculator.shifting_box_in is None:
            # (1.1) 드럼 몸통의 높이가 드럼 왼쪽과 오른쪽의 평균 높이보다 크면 드럼 이동 유효 영역 박스를 설정합니다.
            require_drum_shift = False

            drum_body_height = drum_body_box[3]
            drum_heights = [box[3] for box in [drum_left_box, drum_right_box]]
            average_drum_height = sum(drum_heights) / len(drum_heights)
            height_condition = (
                drum_body_height > average_drum_height * self.common_config.body_height_ratio
            )

            # N 바퀴의 회전수를 달성했는지도 체크합니다.
            min_turn_angle = max(0.0, self.shift_config.min_turns_for_shifting_box) * 360.0
            has_sufficient_turns = self._initial_shifting_box_rotation_angle >= min_turn_angle

            # 이미 케이블이 감기는 도중에 시작한 경우:
            # progress_rate가 유효하고(0~1 범위) 일정 수준 이상이면 진행 중 시작으로 판단
            is_mid_progress_start = (
                height_condition
                and 0.0 <= progress_rate <= 1.0
                and progress_rate >= self.is_mid_start_progress_threshold
            )

            require_box_reset = height_condition and (has_sufficient_turns or is_mid_progress_start)

        # (2) 드럼 이동 유효 영역 박스가 설정되어 있는 경우
        else:
            # (2.1) 이전 주기에 권취 불량 알람이 발생했던 경우
            if self._has_previous_interval_alarm_occurred:
                # 주기 도래 시 판정
                if self.tick_timer():
                    if drum_shifting_score >= self.shift_config.shift_alarm_threshold:
                        require_drum_shift, require_box_reset = True, False
                    else:
                        self._has_previous_interval_alarm_occurred = False
                        require_drum_shift, require_box_reset = False, False
                else:
                    require_drum_shift, require_box_reset = False, False

            # (2.2) 이전 주기에 권취 불량 알람이 발생하지 않았던 경우
            else:
                # (2.2.1) 드럼 이동 점수가 임계값을 초과한 경우
                if drum_shifting_score >= self.shift_config.shift_alarm_threshold:
                    self._has_previous_interval_alarm_occurred = True
                    require_drum_shift, require_box_reset = True, False

                # (2.2.2) 드럼 이동 점수가 임계값 이하인 경우
                else:
                    require_drum_shift, require_box_reset = False, False

        return require_drum_shift, require_box_reset

    def _check_side_drum_shift_requirements(
        self,
        require_drum_shift: bool,
        tightening_score: float,
        segmentation_results: SegmentationResults,
        cable_moving_direction: Direction,
        is_first_swing_ready: bool,
    ) -> tuple[bool, Direction]:
        """드럼 이동이 필요한지와 이동 방향을 판단합니다.
        tightening 또는 winding_first_wrap 상태에서만 동작합니다.
        """

        drum_moving_direction: Direction = cable_moving_direction.opposite()

        # tightening 상태에서 밀착 점수가 임계값 이상인 경우, 드럼 이동 필요. (단, 주기적으로 신호를 보냄.)
        if isinstance(self.state, TighteningState):
            tightening_score_threshold = self.direction_to_tightening_score_threshold.get(
                cable_moving_direction
            )
            if self.tightening_score_calculator.side_drum_shift_x_range is not None:
                if self.tick_timer():
                    if tightening_score >= tightening_score_threshold:
                        require_drum_shift = True
                    else:
                        require_drum_shift = False
                else:
                    require_drum_shift = False

        return require_drum_shift, drum_moving_direction

    def detect_edge_protrusion(self, segmentation_results: SegmentationResults, roi_side: str) -> bool:
        # slope 구하기
        slope_degree = calculate_rotation_angle_for_flattened_top_line(
            mask=segmentation_results.drum_body_mask,
            box=segmentation_results.drum_body_box,
            roi_side=roi_side,
        )
        if slope_degree is None:
            return False

        self.slope_degree = exponential_moving_average(self.slope_degree, slope_degree, alpha=0.03)
        logging.log(f"    >>> slope degree: {self.slope_degree}")
        logging.log_metrics(
            current_slope_degree=slope_degree,
            slope_degree=self.slope_degree,
        )

        # 회전된 마스크와 박스를 사용하여 평탄화된 상단 라인 기준으로 회전
        rotated_body_mask, rotated_body_box = rotate_mask_for_flattened_top_line(
            segmentation_results.drum_body_mask,
            segmentation_results.drum_body_box,
            roi_side,
            slope_degree=self.slope_degree,
        )

        # 극단 ROI 자르기
        drum_edge_roi, _ = crop_edge_roi(
            rotated_body_mask,
            rotated_body_box,
            roi_side,
            margin=0,
            edge_region_ratio=self.common_config.edge_region_ratio,
            flatten=True,
        )

        if not np.any(drum_edge_roi):
            return False

        # 극단 ROI의 좌우 끝단 최상단 픽셀 위치 계산
        box_top_y = rotated_body_box[1]
        left_edge_top_y = np.flatnonzero(drum_edge_roi[:, 0])[0] + box_top_y
        right_edge_top_y = np.flatnonzero(drum_edge_roi[:, -1])[0] + box_top_y

        # EMA 계산 및 히스토리 업데이트
        ema_left_edge_top_y = exponential_moving_average(
            prev_value=self.left_y_top_history[-1] if self.left_y_top_history else None,
            value=left_edge_top_y,
        )
        ema_right_edge_top_y = exponential_moving_average(
            prev_value=self.right_y_top_history[-1] if self.right_y_top_history else None,
            value=right_edge_top_y,
        )

        ema_left_edge_top_y = round(ema_left_edge_top_y)
        ema_right_edge_top_y = round(ema_right_edge_top_y)

        self.left_y_top_history.append(ema_left_edge_top_y)
        self.right_y_top_history.append(ema_right_edge_top_y)

        # 좌우 극단별 값 선택
        # NOTE: base_top_y에 min을 사용하는 이유는 드럼 회전에 따라 중간 부분에 권취된 전선의 높이가 변하는 경우 때문
        if roi_side == "right":
            edge_top_y = ema_right_edge_top_y
            base_top_y = min(self.left_y_top_history)
        else:
            edge_top_y = ema_left_edge_top_y
            base_top_y = min(self.right_y_top_history)

        # 돌출 감지 (극단과 중간 부분의 높이 차이 기준)
        height_diff = base_top_y - edge_top_y

        logging.log(
            f"    >>> height difference ({roi_side}): {height_diff} (base: {base_top_y}, edge: {edge_top_y})"
        )
        logging.log_metrics(
            base_top_y=base_top_y,
            edge_top_y=edge_top_y,
            height_difference=height_diff,
        )

        # 극단과 중간 부분의 높이 차이가 임계값 이하인 경우
        if height_diff <= self.shift_config.min_pixel_for_protrusion:
            return False

        # 돌출이 발생했으면
        return True

    def _update_first_wrapping_protrusion_state(
        self,
        segmentation_results: SegmentationResults,
        cable_moving_direction: Direction,
    ) -> None:
        """
        FirstWrapping 상태에서 극단의 상단 돌출부를 감지합니다.

        전선이 극단에 도달하여 새로운 층이 생기면 극단의 최상단 픽셀이 기준 높이보다 위로 올라갑니다.
        연속된 프레임에서 돌출이 감지되어야 확정됩니다.

        Args:
            rotation_angle: AngleMonitor에서 계산된 최근 1프레임의 회전 각도
        """
        if not isinstance(self.state, FirstWrappingState):
            return

        if not self._is_edge_protrusion_detected:
            roi_side = get_roi_side(self.state.name, cable_moving_direction)

            # XXX: logic 5에서 사용하기 위한 각도 참조
            angle_threshold = self.direction_to_protrusion_angle_threshold.get(cable_moving_direction)
            self.set_reference_angle(angle_threshold)
            self.record_rotation_angle(self.cumulative_rotation_angle)
            logging.log(
                f"    >>> protrusion detection... angle={self.cumulative_rotation_angle:.2f} / {angle_threshold:.2f}"
            )

            tf = self.detect_edge_protrusion(segmentation_results, roi_side=roi_side)
            self._protrusion_history.append(tf)

            # 연속으로 돌출부가 감지되어야 확정
            reset_rotation_angle = False
            protrusion_window = self.shift_config.protrusion_confirmation_window

            logging.log(
                f"    >>> protrusion confirmed: {sum(self._protrusion_history[-protrusion_window :])}"
                f" / {protrusion_window}"
            )

            if len(self._protrusion_history) < protrusion_window:
                self._is_edge_protrusion_detected = False
            elif (
                sum(self._protrusion_history[-protrusion_window:])
                >= self.shift_config.protrusion_confirmation_frames
            ):
                self._is_edge_protrusion_detected = True
                reset_rotation_angle = True
            elif self.cumulative_rotation_angle >= angle_threshold:
                self._is_edge_protrusion_detected = True
                reset_rotation_angle = True
            else:
                self._is_edge_protrusion_detected = False

            if reset_rotation_angle:
                self.record_rotation_angle(0.0)
                self.set_reference_angle(None)

    def _check_side_drum_swing(
        self,
        drum_swing_score: float,
        drum_swing_excess_travel: float,
        require_drum_shift: bool,
        cable_moving_direction: Direction,
        drum_moving_direction: Direction,
    ) -> tuple[bool, Direction]:
        """양극단에서 두 번째 권취를 위한 이동 로직"""

        swing_score_threshold = self.direction_to_swing_score_threshold.get(cable_moving_direction)

        # 1) First swing: 극단에서 전선을 넘기기 위해 드럼 이동
        if isinstance(self.state, FirstSwingState):
            # FIRST_SWING_START 상태에서는 스킵 (handle_state_transition에서 상태 전환 처리 후 다음 프레임에서 실행)
            if self.process_state == ProcessState.FIRST_SWING_START:
                return require_drum_shift, drum_moving_direction

            # 극단 스윙 점수가 임계값을 초과한 경우
            if drum_swing_score >= swing_score_threshold:
                require_drum_shift = True
                self.num_swing_alarms += 1
                self.state.update_shift_alarm_history(True)
            else:
                self.state.update_shift_alarm_history(False)

            if drum_swing_score <= 0.0:
                # 극단 스윙 점수가 0에 도달한 경우, 스윙 & 스윙 복귀 대기 단계 스킵
                self.state.target_reached = True

        # 2) Swing term: first swing 종료 후 전선이 일정 수준 감기는 동안 드럼 이동 중지
        elif isinstance(self.state, SwingTermState):
            if self.state.waiting_drum_shift:
                # XXX: 스윙 종료 후, 드럼 이동 멈추기를 기다리는 중
                require_drum_shift = False
            else:
                cable_thickness = self.swing_score_calculator.get_cable_width()

                # 과도한 이동 여부 판단
                threshold = cable_thickness * self.shift_config.swing_excess_factor
                self.state.excess_travel = True if drum_swing_excess_travel > threshold else False

                # XXX: swing score를 측정하며, 스윙 드럼 이동 신호 추가 발생이 필요한지 판단.
                if drum_swing_score < self.target_swing_score:
                    # swing score가 목표 점수보다 낮은 경우,
                    require_drum_shift = False
                    self.state.target_reached = True

                    # XXX: 오른쪽 극단에서 과도한 이동이 감지된 경우, 복귀 드럼 이동 신호 1회 생성
                    if self.state.excess_travel and not self.state.excess_travel_prevented:
                        drum_moving_direction = cable_moving_direction
                        require_drum_shift = True
                        self.num_swing_alarms -= 1
                        self.state.excess_travel_prevented = True
                        logging.log(
                            f"    >>> excess travel prevented in right swing term: {drum_swing_excess_travel} > {threshold}"
                        )

                else:
                    # swing score가 목표 점수보다 높은 경우, 스윙 드럼 이동 신호 추가 1회 생성
                    if self.state.additional_move_timer.tick():
                        require_drum_shift = True
                        self.num_swing_alarms += 1
                        logging.log(
                            f"    >>> additional drum swing shift: swing_score={drum_swing_score:.2f} >= {self.target_swing_score}"
                        )

        # 3) Second swing: first swing 동안 이동한 드럼을 원래 위치로 복원
        elif isinstance(self.state, SecondSwingState):
            if self.num_swing_alarms > 0:
                require_drum_shift = True
                self.num_swing_alarms -= 1
                drum_moving_direction = cable_moving_direction

        return require_drum_shift, drum_moving_direction

    def _handle_process_state_in_first_wrapping(self) -> None:
        if self.process_state == ProcessState.PROTRUSION_DETECTION:
            self.set_process_state(ProcessState.PROTRUSION_DETECTION_END)
            return
        if self.process_state == ProcessState.PROTRUSION_DETECTION_END:
            # Freeze cumulative average when entering WAITING_FOR_FIRST_SWING
            self.swing_score_calculator.freeze_cumulative_avg()
            self.set_process_state(ProcessState.WAITING_FOR_FIRST_SWING)
            return

    def _is_first_swing_ready(
        self,
        cable_moving_direction: Direction,
        segmentation_results: SegmentationResults | None = None,
        roi_side: str | None = None,
    ) -> bool:
        is_first_swing_ready = False

        if not self._is_edge_protrusion_detected or not isinstance(self.state, FirstWrappingState):
            return is_first_swing_ready

        self._handle_process_state_in_first_wrapping()

        # 자동 스윙 건너뛰기 체크 (WAITING_FOR_FIRST_SWING 상태의 모든 프레임에서 체크, 왼쪽 RoI에서만)
        if self.process_state == ProcessState.WAITING_FOR_FIRST_SWING:
            if segmentation_results is not None and roi_side == "left" and not self._auto_swing_skipped:
                # Cumulative average error based logic (left RoI only)
                should_skip, _ = self._check_cable_position_by_avg_error(segmentation_results)
                if should_skip:
                    self._auto_swing_skipped = True

        angle_threshold = self.direction_to_swing_angle_threshold.get(cable_moving_direction)

        self.set_reference_angle(angle_threshold)
        self.record_rotation_angle(self.cumulative_rotation_angle)
        logging.log(
            f"    >>> wating for first swing... angle={self.cumulative_rotation_angle:.2f} / {angle_threshold:.2f}"
        )

        if self.cumulative_rotation_angle >= angle_threshold:
            is_first_swing_ready = True

        return is_first_swing_ready

    def _calculate_tightening_distance(
        self, segmentation_results: SegmentationResults, cable_moving_direction: Direction
    ) -> tuple[float | None, float | None]:
        """
        right drum의 1:3 division point와 winding cable edge 사이의 거리를 계산하여 EMA 적용.
        밀착 준비 상태 감지에 사용됨.

        Returns:
            tuple: (distance_ema, cable_width)
        """
        # right_drum bbox와 winding_cable mask 확인
        right_drum_box = segmentation_results.bounding_boxes.get("right_drum")
        winding_cable_mask = segmentation_results.winding_cable_mask

        if right_drum_box is None or winding_cable_mask is None:
            return None, None

        if cable_moving_direction.is_unknown():
            return None, None

        # 1:3 division point 계산
        right_x, right_y, right_w, right_h = right_drum_box
        left_min_x = right_x
        division_point_x = int(left_min_x + right_w / 4)

        # Winding cable edge detection (top 1% y-coordinates 방식)
        # direction.py와 동일한 방식으로 구현
        top_rate = 0.01
        ys, xs = np.where(winding_cable_mask == 1)

        if len(ys) == 0 or len(xs) == 0:
            return None, None

        threshold = np.percentile(ys, top_rate * 100)
        top_y_indices = ys <= threshold
        top_x = xs[top_y_indices]

        if len(top_x) == 0:
            return None, None

        # Direction에 따라 edge 선택
        if cable_moving_direction.is_left_to_right():
            winding_top_x = int(top_x.max())  # 오른쪽 edge (진행 방향)
        elif cable_moving_direction.is_right_to_left():
            winding_top_x = int(top_x.min())  # 왼쪽 edge (진행 방향)
        else:
            return None, None

        # Distance 계산 (절대값)
        distance = abs(winding_top_x - division_point_x)

        # EMA 적용 (기존값 0.6, 새로운 값 0.4)
        if self._distance_ema is None:
            self._distance_ema = float(distance)
        else:
            self._distance_ema = self._distance_ema * 0.6 + distance * 0.4

        # Cable width 계산 (히스토리에 추가)
        self.swing_score_calculator.calculate_cable_width_from_mask(
            winding_cable_mask, use_history=True
        )
        cable_width = self.swing_score_calculator.get_cable_width()

        return self._distance_ema, cable_width

    def handle_state_transition(self, context: dict | None = None) -> dict:
        return self.state.handle_transition(self, context)

    @property
    def shift_sensitivity(self) -> int:
        """드럼 이동을 감지하기 위한 [1, 100] 사이의 민감도."""
        return self._shift_sensitivity

    @shift_sensitivity.setter
    def shift_sensitivity(self, value: int) -> None:
        if not isinstance(value, int):
            raise MonitoringOptionShiftSensitivityTypeError
        if not (1 <= value <= 100):
            raise MonitoringOptionShiftSensitivityValueError

        self._shift_sensitivity = value

    @property
    def shift_alarm_interval(self) -> int:
        """드럼 이동 알람 주기."""
        return self._shift_alarm_interval

    @shift_alarm_interval.setter
    def shift_alarm_interval(self, value: int) -> None:
        if not isinstance(value, int):
            raise MonitoringOptionShiftAlarmIntervalTypeError
        if value <= 0:
            raise MonitoringOptionShiftAlarmIntervalValueError(
                "shift_alarm_interval must be greater than 0."
            )
        self._shift_alarm_interval = value
        if isinstance(self.state, (WindingState, TighteningState, FirstWrappingState)):
            self.state.timer.set_interval(self.shift_alarm_interval)

    @property
    def monitoring_options(self) -> dict:
        monitoring_options = {
            "shift_sensitivity": self.shift_sensitivity,
            "shift_alarm_interval": self.shift_alarm_interval,
            "shift_strength": self.shift_strength,
        }
        return monitoring_options

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        params = options.get("params", {})
        self.update_config_from_params(params)

        # Update basic monitoring options
        if "shift_sensitivity" in params:
            self.shift_sensitivity = params["shift_sensitivity"]
        if "shift_strength" in params:
            self.shift_strength = params["shift_strength"]
            self.shift_strength_second = max(
                1, round(self.shift_strength / MILLISECONDS_IN_SECOND)
            )  # ms to s
        if "shift_alarm_interval" in params:
            self.shift_alarm_interval = params["shift_alarm_interval"]

        _scaling_factor = (
            (self.shift_config.shift_sensitivity - self.shift_sensitivity)
            / self.shift_config.shift_sensitivity
            * self.shift_config.sensitivity_scaling_ratio_limit
        )  # [-sensitivity_scaling_ratio_limit, sensitivity_scaling_ratio_limit] 범위

        self.drum_shift_score_calculator.shifting_box_ratio = (
            self.shift_config.shifting_box_ratio + _scaling_factor
        )

    @property
    def default_monitoring_options(self) -> dict:
        default_monitoring_options = {
            "shift_sensitivity": self.shift_config.shift_sensitivity,
            "shift_alarm_interval": self.shift_config.shift_alarm_interval,
            "shift_strength": self.common_config.shift_strength,
        }
        return default_monitoring_options

    def tick_timer(self) -> bool:
        return self.state.timer.tick()

    def reset(self):
        """Reset the engine state."""
        self.drum_shift_score_calculator.reset()
        self.swing_score_calculator.reset()
        self.tightening_score_calculator.reset()

        self.num_swing_alarms: int = 0  # 두번쨰 바퀴 이동 알람 횟수
        self._has_previous_interval_alarm_occurred = (
            False  # 이전 알람 주기에서 점수가 임계값을 초과했는지 여부
        )

        # Options (self.shift_config 내부값은 default 값이므로 변하지 않고, 인스턴스 변수로 조절 가능)
        self._shift_sensitivity = self.shift_config.shift_sensitivity
        self._shift_alarm_interval = self.shift_config.shift_alarm_interval
        self.swing_shift_alarm_interval = self.shift_config.swing_shift_alarm_interval
        self.shift_strength = self.common_config.shift_strength  # 밀리초

        # Distance EMA for tightening detection
        self._distance_ema: float | None = None

        self.state: BaseState = WindingState(self.shift_alarm_interval)

        # swing 시작 시점 관련 (극단 돌출 감지)
        self._protrusion_history: list[bool] = []  # 돌출 감지 히스토리 (연속 프레임 확인용)
        self._is_edge_protrusion_detected = False  # 극단 돌출부 확정 여부
        self._auto_swing_skipped: bool = False  # 자동 스윙 건너뛰기 플래그

        # Interval params
        self.shift_strength_second = max(
            1, round(self.shift_strength / MILLISECONDS_IN_SECOND)
        )  # ms to s

        # swing logic 관련
        self.rotation_angle_history: deque[float] = deque(maxlen=10)  # 최근 프레임의 회전 각도 히스토리
        self.additional_swing_delay_angle: float = (
            0.0  # swing 복귀 추가 대기 각도. 스윙 중 동작 멈춘 경우 누적.
        )

        # XXX: 제품 UI 로그를 위하여 추가됨.
        self.process_state: ProcessState = ProcessState.WINDING
        self.cumulative_rotation_angle: float = 0.0  # 누적 회전 각도
        self.reference_angle: float | None = None
        self._initial_shifting_box_rotation_angle: float = 0.0  # 유효 영역 생성 전 누적 회전 각도
        self.return_rotation_angle_state: list[ProcessState] = [
            ProcessState.WAITING_TIGHTENING,
            ProcessState.WAITING_FOR_FIRST_SWING,
            ProcessState.SWING_TERM,
        ]  # 이 3가지 상태에서만 rotation_angle과 reference_angle 반환

        # XXX: 돌출 감지를 위하여 추가됨
        self.rotation_angle: float = 0.0  # 현재 프레임의 회전 각도
        self.left_y_top_history: list[int] = []
        self.right_y_top_history: list[int] = []

        # FirstWrappingState 진입 후 프레임 카운트 (avg 업데이트 지연용)
        self._first_wrapping_frame_count: int = 0
        self.slope_degree: float | None = None

    def _check_cable_position_by_avg_error(
        self,
        segmentation_results: SegmentationResults,
        skip_threshold_factor: float = 0.3,
        excessive_threshold_factor: float = 0.7,
    ) -> tuple[bool, bool]:
        """
        Average(frozen or current) 대비 전선 위치를 체크하여 auto skip과 excessive 여부를 한번에 판단.
        왼쪽 RoI에서만 동작.

        Args:
            segmentation_results: 세그멘테이션 결과
            skip_threshold_factor: Auto skip 임계값 배수 (기본값 0.3, cable_width * 0.3)
            excessive_threshold_factor: Excessive 임계값 배수 (기본값 0.7, cable_width * 0.7)

        Returns:
            tuple[bool, bool]: (should_auto_skip, is_excessive)
        """
        # Use frozen average if available, otherwise use current average
        frozen_avg_top_x = self.swing_score_calculator.get_frozen_avg_top_x()
        if frozen_avg_top_x is not None:
            avg_top_x = frozen_avg_top_x
        else:
            avg_top_x = self.swing_score_calculator.get_current_avg_top_x()

        if avg_top_x is None:
            return False, False

        # 필요한 마스크 확인
        winding_cable_mask = segmentation_results.segmentation_masks.get("winding_cable")
        if winding_cable_mask is None:
            return False, False

        # 전선 두께 계산
        cable_width = self.swing_score_calculator.get_cable_width()
        if cable_width is None:
            return False, False

        # 현재 winding cable의 top x 좌표
        current_top_x = find_winding_top_x(winding_cable_mask)
        if current_top_x is None:
            return False, False

        # Calculate error (positive direction = cable moved to right)
        error = current_top_x - avg_top_x
        skip_threshold = cable_width * skip_threshold_factor
        excessive_threshold = cable_width * excessive_threshold_factor
        avg_type = "frozen" if frozen_avg_top_x is not None else "current"

        # 두 가지 조건 모두 체크
        should_auto_skip = error > 0 and error >= skip_threshold
        is_excessive = error > 0 and error >= excessive_threshold

        # 로그 출력
        if is_excessive:
            logging.log(
                f"    >>> Excessive cable position: error={error:.1f} >= excessive_threshold={excessive_threshold:.1f} "
                f"(current_x={current_top_x:.1f}, {avg_type}_avg={avg_top_x:.1f}, cable_width={cable_width})"
            )
        elif should_auto_skip:
            logging.log(
                f"    >>> Auto skip condition met: error={error:.1f} >= skip_threshold={skip_threshold:.1f} "
                f"(current_x={current_top_x:.1f}, {avg_type}_avg={avg_top_x:.1f}, cable_width={cable_width})"
            )
        else:
            logging.log(
                f"    >>> Cable position normal: error={error:.1f} < skip_threshold={skip_threshold:.1f} "
                f"(current_x={current_top_x:.1f}, {avg_type}_avg={avg_top_x:.1f}, cable_width={cable_width})"
            )

        return should_auto_skip, is_excessive

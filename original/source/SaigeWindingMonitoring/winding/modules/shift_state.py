from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .direction import Direction
from .process_state import ProcessState
from ..modules import logging
from ..utils.timer import IntervalTimer
from ..data.data_class import SegmentationResults, SideSegmentationResults

if TYPE_CHECKING:
    from .shifting import ShiftMonitor  # to avoid circular import


def _reset_rotation_and_reference(monitor: ShiftMonitor, reference_angle: float | None) -> None:
    monitor.record_rotation_angle(0.0)
    monitor.set_reference_angle(reference_angle)


class BaseState:
    name: str = "winding"

    def handle_transition(self, monitor: ShiftMonitor, context: dict | None = None) -> dict:
        if context is None or context.get("cable_moving_direction").is_unknown():
            return {}
        events = self._handle_transition(monitor, context)
        return events or {}

    def _handle_transition(self, monitor: ShiftMonitor, context: dict | None = None) -> dict | None:
        raise NotImplementedError


class WindingState(BaseState):
    name = "winding"

    def __init__(self, shift_alarm_interval: int) -> None:
        self.timer = IntervalTimer(shift_alarm_interval)
        self.waiting_phase = False

    def _handle_transition(self, monitor: ShiftMonitor, context: dict) -> None:
        if not monitor.process_state in [ProcessState.WINDING, ProcessState.WAITING_TIGHTENING]:
            monitor.set_process_state(ProcessState.WINDING)

        progress_rate: float = context.get("progress_rate")
        segmentation_results: SegmentationResults = context.get("segmentation_results")
        if progress_rate is None or segmentation_results is None:
            return

        if not self.waiting_phase and self._is_tightening_ready(monitor, context):
            monitor.set_process_state(ProcessState.WAITING_TIGHTENING)
            self.waiting_phase = True

            # 각도 관련 값 초기화 - 밀착 대기 각도 측정을 위하여.
            _reset_rotation_and_reference(monitor, monitor.shift_config.tightening_delay_angle)

        if self.waiting_phase:
            rotation_angle = monitor.cumulative_rotation_angle
            cable_moving_direction: Direction = context.get("cable_moving_direction")

            # left_to_right 방향일 때만 밀착 거리 체크: distance_ema가 cable_width의 threshold 이하면 즉시 tightening 시작
            if cable_moving_direction == Direction.LEFT_TO_RIGHT:
                distance_ema: float = context.get("distance_ema")
                cable_width: float = context.get("cable_width")

                if distance_ema is not None and cable_width is not None:
                    early_trigger_threshold = (
                        cable_width * monitor.shift_config.tightening_early_trigger_distance_factor
                    )
                    if distance_ema <= early_trigger_threshold:
                        logging.log(
                            f"    Early tightening trigger: distance_ema={distance_ema:.1f} <= threshold={early_trigger_threshold:.1f}"
                        )
                        logging.log("    State conversion: Winding -> Tightening")
                        monitor.reset()
                        monitor.set_state(TighteningState(monitor.shift_alarm_interval))
                        monitor.set_process_state(ProcessState.TIGHTENING_START)
                        return

            logging.log(
                f"    >>> waiting for tightening... angle={rotation_angle:.2f} / {monitor.shift_config.tightening_delay_angle:.2f}"
            )

            if rotation_angle >= monitor.shift_config.tightening_delay_angle:
                logging.log("    State conversion: Winding -> Tightening")
                monitor.reset()
                monitor.set_state(TighteningState(monitor.shift_alarm_interval))
                monitor.set_process_state(ProcessState.TIGHTENING_START)

    def _is_tightening_ready(
        self,
        monitor: ShiftMonitor,
        context: dict,
    ) -> bool:
        cable_moving_direction: Direction = context.get("cable_moving_direction")
        
        # right_to_left 방향일 때는 기존 progress_rate 기반 로직 사용
        if cable_moving_direction == Direction.RIGHT_TO_LEFT:
            progress_rate: float = context.get("progress_rate")
            if progress_rate is None:
                return False
            
            # 진행률이 기준치 이상일 경우, 밀착 준비 완료로 간주.
            if progress_rate >= monitor.shift_config.progress_rate_threshold:
                return True
            return False
        
        # left_to_right 방향일 때만 distance_ema 기반 로직 사용
        distance_ema: float = context.get("distance_ema")
        cable_width: float = context.get("cable_width")

        if distance_ema is None or cable_width is None:
            return False

        # EMA 적용된 거리가 전선 두께의 threshold factor배 이하일 경우, 밀착 준비 완료로 간주.
        ready_threshold = cable_width * monitor.shift_config.tightening_ready_distance_factor
        if distance_ema <= ready_threshold:
            return True

        return False


class TighteningState(BaseState):
    name = "tightening"

    def __init__(self, shift_alarm_interval: int) -> None:
        self.timer = IntervalTimer(shift_alarm_interval, prime=True)

    def _handle_transition(self, monitor: ShiftMonitor, context: dict) -> dict:
        if monitor.process_state == ProcessState.TIGHTENING_START:
            monitor.set_process_state(ProcessState.TIGHTENING)

        tightening_score = context.get("tightening_score")
        if tightening_score is None:
            return {}

        cable_moving_direction: Direction = context.get("cable_moving_direction")
        tightening_score_threshold = monitor.direction_to_tightening_score_threshold.get(
            cable_moving_direction
        )
        logging.log(
            f"    >>> tightening... score={tightening_score:.2f} / {tightening_score_threshold:.2f}"
        )

        if tightening_score < tightening_score_threshold:
            if monitor.tightening_score_calculator.side_drum_shift_x_range is not None:
                logging.log("    State conversion: Tightening -> First Wrapping")

                monitor.tightening_score_calculator.reset()
                monitor.set_state(FirstWrappingState(monitor.shift_alarm_interval))
                monitor.set_process_state(ProcessState.TIGHTENING_END)

                # 각도 관련 값 초기화
                _reset_rotation_and_reference(monitor, None)

                return {"require_direction_change": True}

        return {}


class FirstWrappingState(BaseState):
    name = "winding_first_wrap"

    def __init__(self, shift_alarm_interval: int) -> None:
        self.timer = IntervalTimer(shift_alarm_interval, prime=True)
        # shift_alarm_interval 주기로 초과 이동 여부 확인
        self.shift_check_timer = IntervalTimer(shift_alarm_interval)

    def _handle_process_state(self, monitor: ShiftMonitor) -> None:
        if monitor.process_state == ProcessState.TIGHTENING_END:
            monitor.set_process_state(ProcessState.PROTRUSION_DETECTION_START)
            return
        if monitor.process_state == ProcessState.PROTRUSION_DETECTION_START:
            monitor.set_process_state(ProcessState.PROTRUSION_DETECTION)
            return

    def _handle_transition(self, monitor: ShiftMonitor, context: dict) -> dict:
        self._handle_process_state(monitor)

        is_first_swing_ready: bool = context.get("is_first_swing_ready")
        segmentation_results: SegmentationResults = context.get("segmentation_results")

        # PROTRUSION_DETECTION 중에 초과 체크 (shift_alarm_interval 주기)
        events = {}
        if monitor.process_state == ProcessState.PROTRUSION_DETECTION:
            if self.shift_check_timer.tick() and segmentation_results is not None:
                _, is_excessive = monitor._check_cable_position_by_avg_error(segmentation_results)
                if is_excessive:
                    # 초과 시 드럼 이동 신호 발생
                    logging.log(
                        "    >>> Cable position exceeds threshold during PROTRUSION_DETECTION - drum shift required"
                    )
                    events = {
                        "require_drum_shift": True,
                        "drum_moving_direction": Direction.LEFT_TO_RIGHT,
                    }

        if not is_first_swing_ready or segmentation_results is None:
            return events

        # 자동 스윙 건너뛰기 체크: first_swing을 거치지 않고 바로 auto_swing_skip_reset으로 전환
        if monitor._auto_swing_skipped:
            logging.log("    State conversion: First Wrapping -> Auto Swing Skip Reset")
            _reset_delay_seconds = (
                monitor.shift_strength_second * monitor.shift_config.reset_delay_factor
            )
            monitor.set_state(
                AutoSwingSkipResetState(
                    reset_delay_seconds=_reset_delay_seconds,
                    shift_alarm_interval=monitor.shift_alarm_interval,
                )
            )
            monitor.set_process_state(ProcessState.AUTO_SWING_SKIP_RESET)

            # 각도 초기화 - reset delay angle 대기를 위해
            reset_delay_angle = monitor.direction_to_reset_delay_angle.get(Direction.LEFT_TO_RIGHT)
            _reset_rotation_and_reference(monitor, reset_delay_angle)
            return events

        logging.log("    State conversion: First Wrapping -> First Swing")
        monitor.set_state(FirstSwingState(monitor.shift_strength_second))
        monitor.set_process_state(ProcessState.FIRST_SWING_START)

        # 각도 관련 값 초기화
        _reset_rotation_and_reference(monitor, None)

        return events


class FirstSwingState(BaseState):
    name = "first_swing"

    def __init__(self, shift_strength_second: int) -> None:
        self._target_reached = False
        # 이전 shift_strength_second 동안의 이동 알람 이력을 저장하는 큐
        self._shift_alarm_history = deque(maxlen=shift_strength_second)

    @property
    def target_reached(self) -> bool:
        # 첫번째 스윙 목표 지점 도달 여부
        return self._target_reached

    @target_reached.setter
    def target_reached(self, value: bool) -> None:
        self._target_reached = value

    @property
    def shift_alarm_history(self) -> deque:
        return self._shift_alarm_history

    def update_shift_alarm_history(self, alarm_triggered: bool) -> None:
        self._shift_alarm_history.append(alarm_triggered)

    def _handle_transition(self, monitor: ShiftMonitor, context: dict) -> None:
        if monitor.process_state == ProcessState.FIRST_SWING_START:
            # XXX: 이미 목표 지점에 도달한 경우는 바로 second swing 상태로 전환
            if self.target_reached:
                logging.log(
                    "    Already reached first swing target region. Transitioning to Second Swing."
                )
                _reset_delay_seconds = (
                    monitor.shift_strength_second * monitor.shift_config.reset_delay_factor
                )
                monitor.set_state(
                    SecondSwingState(
                        reset_delay_seconds=_reset_delay_seconds,
                    )
                )
                monitor.set_process_state(ProcessState.SECOND_SWING_START)
                return

            else:
                monitor.set_process_state(ProcessState.FIRST_SWING)
                return

        drum_swing_score: float = context.get("drum_swing_score")
        if drum_swing_score is None:
            return

        cable_moving_direction: Direction = context.get("cable_moving_direction")
        swing_score_threshold = monitor.direction_to_swing_score_threshold.get(cable_moving_direction)
        logging.log(f"    >>> first swing... score={drum_swing_score:.2f} / {swing_score_threshold:.2f}")

        if drum_swing_score < swing_score_threshold:
            logging.log(f"    State conversion: First Swing -> Swing Term")
            monitor.set_state(
                SwingTermState(monitor.shift_strength_second, monitor.shift_config.shift_alarm_interval)
            )
            monitor.set_process_state(ProcessState.FIRST_SWING_END)


class SwingTermState(BaseState):
    name = "swing_term"

    def __init__(self, wait_seconds: int, additional_wait_seconds: int) -> None:
        self.timer = IntervalTimer(wait_seconds)
        # 추가 이동 완료 대기 타이머
        self.additional_move_timer = IntervalTimer(additional_wait_seconds, prime=True)
        self._waiting_drum_shift = True
        self._excess_travel = False
        self._excess_travel_prevented = False
        self._target_reached = False
        self.angle_reset = False  # 각도 초기화 여부

    @property
    def excess_travel(self) -> bool:
        # 스윙 복귀 대기 동안, 전선이 스윙 목표 지점 대비 초과 이동했는지 여부.
        return self._excess_travel

    @excess_travel.setter
    def excess_travel(self, value: bool) -> None:
        self._excess_travel = value

    @property
    def excess_travel_prevented(self) -> bool:
        # 스윙 복귀 대기 동안, 전선의 초과 이동이 방지되었는지 여부.
        # 오른쪽 극단 스윙에서만 사용됩니다.
        return self._excess_travel_prevented

    @excess_travel_prevented.setter
    def excess_travel_prevented(self, value: bool) -> None:
        self._excess_travel_prevented = value

    @property
    def waiting_drum_shift(self) -> bool:
        return self._waiting_drum_shift

    @property
    def target_reached(self) -> bool:
        # 스윙 복귀 대기 동안, 전선이 스윙 목표 지점에 도달했는지 여부.
        return self._target_reached

    @target_reached.setter
    def target_reached(self, value: bool) -> None:
        self._target_reached = value

    # 종료 시점 판단 함수
    def check_swing_term_end(
        self, cable_moving_direction: Direction, rotation_angle: float, swing_term_angle: float
    ) -> bool:
        if cable_moving_direction == Direction.LEFT_TO_RIGHT and self.excess_travel:
            # 왼쪽 극단일 경우, 초과 이동했다면, 즉시 대기 종료.
            logging.log("    >>> excess travel detected in left swing term.")
            return True

        if rotation_angle >= swing_term_angle:
            return True
        return False

    def _handle_transition(self, monitor: ShiftMonitor, context: dict) -> None:
        cable_moving_direction = context.get("cable_moving_direction")
        monitor.set_process_state(ProcessState.SWING_TERM)
        swing_term_angle = (
            monitor.direction_to_swing_term_angle.get(cable_moving_direction)
            + monitor.additional_swing_delay_angle
        )

        if self._waiting_drum_shift:
            if self.timer.tick():
                self._waiting_drum_shift = False
            logging.log("    >>> waiting for the previous drum shift...")
            return

        segmentation_results: SegmentationResults = context.get("segmentation_results")
        if segmentation_results is None:
            return

        if self.target_reached:
            # 스윙 목표 지점에 도달했을 경우,
            if not self.angle_reset:
                # 각도 관련 값 초기화 - 스윙 대기 각도 측정을 위하여.
                _reset_rotation_and_reference(monitor, swing_term_angle)
                self.angle_reset = True
            else:
                rotation_angle = monitor.cumulative_rotation_angle
                logging.log(f"    >>> swing term... angle={rotation_angle:.2f} / {swing_term_angle:.2f}")

                _reset_delay_seconds = (
                    monitor.shift_strength_second * monitor.shift_config.reset_delay_factor
                )

                if self.check_swing_term_end(cable_moving_direction, rotation_angle, swing_term_angle):
                    logging.log("    State conversion: Swing Term -> Second Swing")
                    monitor.set_state(
                        SecondSwingState(
                            reset_delay_seconds=_reset_delay_seconds,
                        )
                    )
                    monitor.set_process_state(ProcessState.SECOND_SWING_START)


class AutoSwingSkipResetState(BaseState):
    """
    자동 스윙 건너뛰기 후 리셋 대기 상태.
    왼쪽 RoI에서 SWING 없이도 전선이 우측으로 이동한 경우 사용.
    대기 중 전선이 두께의 threshold_factor배 이상 이동하면 드럼 이동 신호 발생.
    """

    name = "auto_swing_skip_reset"

    def __init__(
        self,
        reset_delay_seconds: int,
        shift_alarm_interval: int,
        max_excessive_drum_shifts: int = 2,
    ) -> None:
        self.reset_delay_timer = IntervalTimer(reset_delay_seconds)
        self.delay_completed = False
        # shift_alarm_interval 주기로 드럼 이동 필요 여부 확인
        self.shift_check_timer = IntervalTimer(shift_alarm_interval)
        self.max_excessive_drum_shifts = max_excessive_drum_shifts
        self.excessive_drum_shift_count = 0

    def _handle_transition(self, monitor: ShiftMonitor, context: dict) -> dict:
        cable_moving_direction = Direction.LEFT_TO_RIGHT  # 왼쪽 전용 로직
        reset_delay_angle = monitor.direction_to_reset_delay_angle.get(cable_moving_direction)

        # 초기 대기 시간 경과 체크
        if not self.delay_completed:
            if self.reset_delay_timer.tick():
                self.delay_completed = True
                logging.log("    >>> Initial delay completed, waiting for rotation angle...")
            else:
                logging.log("    >>> Auto swing skip: waiting for initial delay...")
                return {}

        # 각도 대기
        logging.log(
            f"    >>> Auto swing skip reset delay... angle={monitor.cumulative_rotation_angle:.2f} / {reset_delay_angle:.2f}"
        )

        # 드럼 이동 필요 여부 체크 (shift_alarm_interval 주기)
        events = {}
        if self.shift_check_timer.tick():
            segmentation_results: SegmentationResults = context.get("segmentation_results")
            if segmentation_results is not None:
                _, is_excessive = monitor._check_cable_position_by_avg_error(segmentation_results)
                if is_excessive:
                    if self.excessive_drum_shift_count < self.max_excessive_drum_shifts:
                        self.excessive_drum_shift_count += 1
                        logging.log(
                            "    >>> Cable position exceeds threshold - drum shift required "
                            f"({self.excessive_drum_shift_count}/{self.max_excessive_drum_shifts})"
                        )
                        events = {
                            "require_drum_shift": True,
                            "drum_moving_direction": Direction.LEFT_TO_RIGHT,
                        }
                    else:
                        logging.log(
                            "    >>> Cable position exceeds threshold but excessive drum shift "
                            f"limit reached ({self.max_excessive_drum_shifts})"
                        )

        if monitor.cumulative_rotation_angle >= reset_delay_angle:
            logging.log("    State conversion: Auto Swing Skip Reset -> Winding")
            monitor.reset()
            monitor.set_process_state(ProcessState.WINDING)

        return events


class SecondSwingState(BaseState):
    name = "second_swing"

    def __init__(self, reset_delay_seconds: int) -> None:
        self.reset_delay_timer = IntervalTimer(reset_delay_seconds)
        self.semi_delay_completed = False

    def _handle_transition(self, monitor: ShiftMonitor, context: dict | None = None) -> None:
        logging.log(f"    Returning: second swing count={monitor.num_swing_alarms}")

        if monitor.process_state == ProcessState.SECOND_SWING_START:
            monitor.set_process_state(ProcessState.SECOND_SWING)

        cable_moving_direction: Direction = context.get("cable_moving_direction")
        reset_delay_angle = monitor.direction_to_reset_delay_angle.get(cable_moving_direction)

        if not self.semi_delay_completed:
            if monitor.num_swing_alarms == 0 and self.reset_delay_timer.tick():
                self.semi_delay_completed = True

                # 각도 관련 값 초기화 - 스윙 복귀 후, 권취 대기 각도 측정을 위하여.
                _reset_rotation_and_reference(monitor, reset_delay_angle)

        else:
            logging.log(
                f"    >>> reset delay... angle={monitor.cumulative_rotation_angle:.2f} / {reset_delay_angle:.2f}"
            )

            if monitor.cumulative_rotation_angle >= reset_delay_angle:
                logging.log("    State conversion: Second Swing -> Winding")
                monitor.reset()
                monitor.set_process_state(ProcessState.SECOND_SWING_END)

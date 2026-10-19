from .base import BaseMonitor
from ..engine.config import MonitoringConfig
from ..error import (
    MonitoringOptionSpongeWrappingThresholdTypeError,
    MonitoringOptionSpongeWrappingThresholdValueError,
)


class SpongeMonitor(BaseMonitor):
    """스펀지 보양 알림을 관리하는 클래스"""

    def __init__(self, monitoring_config: MonitoringConfig) -> None:
        super().__init__(monitoring_config)
        self.reset()

    def process_frame(self, progress_rate) -> dict:
        require_sponge_wrapping = False
        if not self._has_sponge_alarm_occurred:
            require_sponge_wrapping = bool(progress_rate * 100 >= self.sponge_wrapping_threshold)
            if require_sponge_wrapping:
                self._has_sponge_alarm_occurred = True

        return {"require_sponge_wrapping": require_sponge_wrapping}

    def reset_alarm(self) -> None:
        """알람 상태를 초기화합니다."""
        self._has_sponge_alarm_occurred = False

    @property
    def sponge_wrapping_threshold(self) -> int:
        """스펀지 보양을 시작할 때의 진행률을 결정하는 [1, 100] 사이의 값."""
        return self._sponge_wrapping_threshold

    @sponge_wrapping_threshold.setter
    def sponge_wrapping_threshold(self, value: int) -> None:
        if not isinstance(value, int):
            raise MonitoringOptionSpongeWrappingThresholdTypeError
        if not (1 <= value <= 100):
            raise MonitoringOptionSpongeWrappingThresholdValueError

        if self._sponge_wrapping_threshold != value:
            self._has_sponge_alarm_occurred = False  # 스펀지 보양 임계값이 변경되면 알람 상태 초기화

        self._sponge_wrapping_threshold = value

    @property
    def monitoring_options(self) -> dict:
        """모니터링 옵션을 반환합니다."""
        return {"sponge_wrapping_threshold": self.sponge_wrapping_threshold}

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        """모니터링 옵션을 설정합니다."""
        params = options.get("params", {})
        self.update_config_from_params(params)

        # Update basic monitoring options
        if "sponge_wrapping_threshold" in params:
            self.sponge_wrapping_threshold = params["sponge_wrapping_threshold"]

    @property
    def default_monitoring_options(self) -> dict:
        """기본 모니터링 옵션을 반환합니다."""
        return {"sponge_wrapping_threshold": self.sponge_config.sponge_wrapping_threshold}

    def reset(self) -> None:
        """알람 상태를 초기화합니다."""
        self._has_sponge_alarm_occurred = False
        self._sponge_wrapping_threshold = self.sponge_config.sponge_wrapping_threshold

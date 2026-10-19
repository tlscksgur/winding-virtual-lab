"""AngleMonitor - 드럼 회전 각도를 독립적으로 계산하는 모니터"""

import numpy as np

from .base import BaseMonitor
from .line_tracker import MultiLineTracker
from ..data.data_class import SideSegmentationResults
from ..engine.config import MonitoringConfig


class AngleMonitor(BaseMonitor):
    """측면 뷰 프레임을 기반으로 드럼의 회전 각도를 계산하는 클래스.

    DirectionMonitor와 ShiftMonitor에서 중복으로 계산하던 rotation angle을
    독립적으로 계산하여 제공합니다.
    """

    def __init__(self, monitoring_config: MonitoringConfig) -> None:
        super().__init__(monitoring_config)
        self.multi_line_tracker = MultiLineTracker(
            self.rotation_config, center=self.rotation_config.drum_center_point
        )
        self.current_frame: np.ndarray | None = None

    def process_frame(
        self,
        side_view_frame: np.ndarray,
        side_segmentation_results: SideSegmentationResults,
    ) -> float:
        """측면 뷰 프레임을 기반으로 회전 각도를 계산합니다.

        Args:
            side_view_frame: 측면 카메라에서 촬영된 프레임
            side_segmentation_results: 측면 프레임에 대한 세그멘테이션 결과

        Returns:
            float: 최근 1프레임의 회전 각도
        """
        if side_view_frame is None or side_segmentation_results is None:
            return 0.0

        # Line-based rotation angle calculation
        self.multi_line_tracker.update(
            side_view_frame,
            side_segmentation_results=side_segmentation_results,
        )

        # 최근 1프레임의 회전 각도만 반환
        rotation_angle = self.get_single_frame_rotation_angle()

        self.current_frame = side_view_frame

        return rotation_angle

    def get_cumulative_rotation_angle(self) -> float:
        """누적 회전 각도를 반환합니다."""
        return self.multi_line_tracker.get_result()

    def get_single_frame_rotation_angle(self) -> float:
        """최근 1프레임의 회전 각도를 반환합니다."""
        return self.multi_line_tracker.get_result(num_frames=1)

    def reset(self) -> None:
        """회전 각도 추적 상태를 초기화합니다."""
        self.multi_line_tracker.reset()
        self.current_frame = None

    @property
    def monitoring_options(self) -> dict:
        """Get current monitoring options."""
        return {}

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        """Set monitoring options, including rotation config parameters."""
        params = options.get("params", {})
        self.update_config_from_params(params)

from __future__ import annotations

import numpy as np

from .base import BaseMonitor

from ..data.data_class import SegmentationResults
from ..engine.config import MonitoringConfig, AnomalySettings
from ..utils.anomaly_tools import (
    find_intervals,
    filter_same_direction_events,
    merge_boxes,
    trim_edge_slopes,
    interpolate_edge,
    smooth_gradient,
    convert_intervals_to_boxes,
)
from ..utils.coordinate import convert_xyxy_to_xywh
from ..error import (
    MonitoringOptionAnomalySensitivityTypeError,
    MonitoringOptionAnomalySensitivityValueError,
)


class AnomalyBox:
    """
    권취 불량 영역 클래스
    """

    def __init__(self, ng_box: np.ndarray) -> None:
        self.ng_box = ng_box

    def is_same_anomaly_box(self, other_box: AnomalyBox, iou_threshold: float) -> bool:
        """
        다른 AnomalyBox 객체와 동일한지 비교하는 메서드.
        """
        return self._calculate_iou(other_box) >= iou_threshold

    def _calculate_iou(self, other_box: AnomalyBox) -> float:
        x1_min, y1_min, x1_max, y1_max = self.ng_box
        x2_min, y2_min, x2_max, y2_max = other_box.ng_box

        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


class AnomalyBoxes:
    """
    AnomalyBox 객체들의 리스트를 관리하는 클래스.
    """

    def __init__(self, anomaly_config: AnomalySettings) -> None:
        self.boxes: list[AnomalyBox] = []
        self.anomaly_config = anomaly_config

    def is_same_anomaly_box(self, other_box: AnomalyBox) -> bool:
        if len(self.boxes) > 0:
            for box in self.boxes:
                if box.is_same_anomaly_box(other_box, self.anomaly_config.overlap_iou_threshold):
                    # 동일한 권취 불량 영역이면 추가하지 않음
                    return True
        self.boxes.append(other_box)
        return False


class AnomalyMonitor(BaseMonitor):
    def __init__(
        self,
        monitoring_config: MonitoringConfig,
    ) -> None:
        super().__init__(monitoring_config)
        self.reset()

    def process_frame(
        self, segmentation_results: SegmentationResults, direction_monitoring_results: dict
    ) -> dict:

        drum_body_mask = segmentation_results.drum_body_mask
        cable_moving_direction = direction_monitoring_results.get("cable_moving_direction")

        if drum_body_mask is None or cable_moving_direction is None:
            return {"is_ng": False, "ng_boxes": None}

        ng_boxes = self.detect_anomaly_boxes(drum_body_mask, segmentation_results.bounding_boxes)
        is_ng = ng_boxes is not None

        return {
            "is_ng": is_ng,
            "ng_boxes": ng_boxes,
        }

    def detect_anomaly_boxes(
        self,
        mask: np.ndarray,
        boxes: dict,
    ) -> np.ndarray | None:
        """
        드럼 몸통 실루엣의 기울기를 분석하여 상·하단 NG(불량) 영역을 찾는 함수.

        1) ROI(관심영역) 추출
        2) 상·하단 실루엣 추출 및 벡터화
        3) NaN 보간
        4) 기울기 계산 + 스무딩 + 마스킹
        5) 이벤트 박스 변환·병합 후 결과 리턴
        """

        # 예외 및 기본값 처리
        if (
            mask is None
            or boxes is None
            or any(boxes[k] is None for k in ("drum_body", "left_drum", "right_drum"))
        ):
            return None

        drum_body_box = boxes["drum_body"]
        left_drum_box = boxes["left_drum"]
        right_drum_box = boxes["right_drum"]

        # ROI(관심영역) 설정
        body_x, body_y, body_w, body_h = drum_body_box
        left_edge = left_drum_box[0] + left_drum_box[2]  # 왼쪽 드럼 테두리 끝
        right_edge = right_drum_box[0]  # 오른쪽 드럼 시작

        # 드럼 몸통 범위를 양옆 드럼 기준으로 잘라내기
        roi_x_min = max(body_x, left_edge)
        roi_x_max = min(body_x + body_w, right_edge)
        roi_w = roi_x_max - roi_x_min

        roi_y, roi_h = body_y, body_h

        # ROI 유효성 체크
        roi_narrowing_width = self.common_config.roi_narrowing_width
        if roi_w < roi_narrowing_width * 3:
            return None

        roi = mask[roi_y : roi_y + roi_h, roi_x_min : roi_x_min + roi_w].copy()
        roi[:, :roi_narrowing_width] = 0  # 좌측 여유 패딩
        roi[:, -roi_narrowing_width:] = 0  # 우측 여유 패딩
        H, W = roi.shape

        # 상·하단 실루엣(윤곽) 벡터화
        mask_roi = roi == 1
        col_has_pixel = mask_roi.any(axis=0)

        # 상단: 각 열(col)에서 첫 번째 True(=전면) 위치
        y_top = np.where(col_has_pixel, mask_roi.argmax(axis=0), np.nan)

        # 하단: 뒤집어서 첫 True를 찾은 뒤 원위치 복원
        rev_mask = mask_roi[::-1, :]
        y_bot_rev = rev_mask.argmax(axis=0)
        y_bot = np.where(col_has_pixel, H - y_bot_rev - 1, np.nan)

        # NaN(결측) 보간 – 가장자리만 대상
        idx = np.arange(W)
        y_top = interpolate_edge(idx, y_top)
        y_bot = interpolate_edge(idx, y_bot)

        # 기울기 계산 + 스무딩 + 마스크링
        grad_top = smooth_gradient(idx, y_top, self.anomaly_config.gradient_smoothing_lambda)
        grad_bot = smooth_gradient(idx, y_bot, self.anomaly_config.gradient_smoothing_lambda)

        # 경사 마스크 (절대값이 임계값 이상인 지점)
        mask_top = np.abs(grad_top) >= self.anomaly_config.slope_threshold
        mask_bot = np.abs(grad_bot) >= self.anomaly_config.slope_threshold

        # 마스크의 양 끝이 True로 시작/끝나는 경우 잘라내기
        mask_top = trim_edge_slopes(mask_top, grad_top)
        mask_bot = trim_edge_slopes(mask_bot, grad_bot)

        # NaN 처리 후 마스크 영역만 보간
        y_top[~mask_top] = np.nan
        y_bot[~mask_bot] = np.nan
        y_top = interpolate_edge(idx, y_top)
        y_bot = interpolate_edge(idx, y_bot)

        # 이벤트 추출
        top_intervals = find_intervals(y_top)
        bot_intervals = find_intervals(y_bot)

        # 박스 변환
        top_events = convert_intervals_to_boxes(
            top_intervals, roi_x_min, roi_y, self.anomaly_config.ng_box_y_padding
        )
        bot_events = convert_intervals_to_boxes(
            bot_intervals, roi_x_min, roi_y, self.anomaly_config.ng_box_y_padding
        )

        # 동일 방향 이벤트 필터링
        top_filtered = filter_same_direction_events(top_events)
        bot_filtered = filter_same_direction_events(bot_events)

        # 박스 병합
        upper_box_ng = merge_boxes(top_filtered)
        lower_box_ng = merge_boxes(bot_filtered)

        ng_boxes = []
        if len(top_filtered) > 1:
            ng_boxes.append([upper_box_ng])
        if len(bot_filtered) > 1:
            ng_boxes.append([lower_box_ng])

        ng_boxes = np.concatenate(ng_boxes, axis=0) if ng_boxes else None
        ng_boxes = self._filter_same_anomaly_box(ng_boxes)
        ng_boxes = convert_xyxy_to_xywh(ng_boxes)
        return ng_boxes

    def _filter_same_anomaly_box(self, ng_boxes: np.ndarray | None) -> np.ndarray | None:
        """
        동일한 권취 불량을 감지한 경우, 중복된 박스를 제거합니다.
        """
        if ng_boxes is None:
            return None

        filtered_ng_boxes = []
        for ng_box in ng_boxes:
            anomaly_box = AnomalyBox(ng_box=ng_box)
            if not self._anomaly_boxes.is_same_anomaly_box(anomaly_box):
                filtered_ng_boxes.append([ng_box])
        filtered_ng_boxes = np.concatenate(filtered_ng_boxes, axis=0) if filtered_ng_boxes else None
        return filtered_ng_boxes

    @property
    def anomaly_sensitivity(self) -> int:
        """권취 불량을 감지하기 위한 [1, 100] 사이의 민감도."""
        return self._anomaly_sensitivity

    @anomaly_sensitivity.setter
    def anomaly_sensitivity(self, value: int) -> None:
        if not isinstance(value, int):
            raise MonitoringOptionAnomalySensitivityTypeError
        if not (1 <= value <= 100):
            raise MonitoringOptionAnomalySensitivityValueError

        self._anomaly_sensitivity = value

    @property
    def monitoring_options(self) -> dict:
        monitoring_options = {
            "anomaly_sensitivity": self.anomaly_sensitivity,
        }
        return monitoring_options

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        params = options.get("params", {})
        self.update_config_from_params(params)

        # Update basic monitoring options (properties with custom setters)
        if "anomaly_sensitivity" in params:
            self.anomaly_sensitivity = params["anomaly_sensitivity"]

        _scaling_factor = (
            (self.anomaly_sensitivity - self.anomaly_config.anomaly_sensitivity)
            / self.anomaly_config.anomaly_sensitivity
            * self.anomaly_config.sensitivity_scaling_slope_limit
        )  # [-sensitivity_scaling_slope_limit, sensitivity_scaling_slope_limit] 범위
        self._slope_threshold = self.anomaly_config.slope_threshold - _scaling_factor
        print(
            f"Monitoring option changed: anomaly_sensitivity: {self.anomaly_sensitivity}, slope_threshold: {self._slope_threshold}"
        )

    @property
    def default_monitoring_options(self) -> dict:
        default_monitoring_options = {"anomaly_sensitivity": self.anomaly_config.anomaly_sensitivity}
        return default_monitoring_options

    def reset(self):
        """Reset the engine state."""
        self._anomaly_boxes = AnomalyBoxes(self.anomaly_config)

        # Options (self.anomaly_config 내부값은 default 값이므로 변하지 않고, 인스턴스 변수로 조절 가능)
        self._anomaly_sensitivity = self.anomaly_config.anomaly_sensitivity
        self._slope_threshold = self.anomaly_config.slope_threshold

from collections import deque
from typing import Any

import json
import numpy as np

from SaigeToolkit.util.timer import Timer

from .config import MonitoringConfig
from ..data.data_class import SegmentationResults, SideSegmentationResults
from ..modules import (
    logging,
    BaseMonitor,
    DirectionMonitor,
    AnomalyMonitor,
    ShiftMonitor,
    SpongeMonitor,
    SegmentationEngine,
    AngleMonitor,
)
from ..modules.process_state import ProcessState
from ..error import (
    DirectionMonitoringError,
    AngleMonitoringError,
    AnomalyMonitoringError,
    ShiftMonitoringError,
    SpongeMonitoringError,
    MissingSideViewFrameError,
)


class WindingMonitoringEngine(BaseMonitor):

    def __init__(
        self,
        monitoring_config: MonitoringConfig,
        segmentation_engine: SegmentationEngine,
        angle_monitor: AngleMonitor,
        direction_monitor: DirectionMonitor,
        sponge_monitor: SpongeMonitor,
        anomaly_monitor: AnomalyMonitor,
        shift_monitor: ShiftMonitor,
    ) -> None:
        super().__init__(monitoring_config)

        self.segmentation_engine = segmentation_engine
        self.angle_monitor = angle_monitor
        self.direction_monitor = direction_monitor
        self.sponge_monitor = sponge_monitor
        self.anomaly_monitor = anomaly_monitor
        self.shift_monitor = shift_monitor

        self.check_time = True  # Enable timing by default
        self.reset()

    @classmethod
    def build(
        cls,
        segmentation: dict,
        monitoring_options: dict | None = None,
    ):
        logging.log("Building Segmentation Engine...")
        segmentation_engine = SegmentationEngine.build(**segmentation)

        monitoring_config = MonitoringConfig()
        monitoring_options = monitoring_options or {}

        angle_monitor = AngleMonitor(monitoring_config)
        direction_monitor = DirectionMonitor(monitoring_config)
        anomaly_monitor = AnomalyMonitor(monitoring_config)

        shift_monitor = ShiftMonitor(monitoring_config)
        sponge_monitor = SpongeMonitor(monitoring_config)

        handler = cls(
            monitoring_config=monitoring_config,
            segmentation_engine=segmentation_engine,
            angle_monitor=angle_monitor,
            direction_monitor=direction_monitor,
            sponge_monitor=sponge_monitor,
            anomaly_monitor=anomaly_monitor,
            shift_monitor=shift_monitor,
        )

        if monitoring_options:
            handler.monitoring_options = monitoring_options

        handler.warmup()

        return handler

    def process_frame(
        self,
        frame: np.ndarray,
        side_view_frame: np.ndarray,
        segmentation_cache_path: str | None = None,
        side_segmentation_cache_path: str | None = None,
    ) -> tuple[dict, SideSegmentationResults]:
        """입력 받은 프레임에 대해 권취 모니터링을 수행합니다.

        Args:
            frame (np.ndarray): 권취 모니터링을 수행할 프레임 입니다.
            side_view_frame (np.ndarray): 측면 카메라에서 촬영된 프레임으로 각도 측정에 사용됩니다.
            segmentation_cache_path (str | None): 세그멘테이션 결과를 캐싱할 경로입니다.
            side_segmentation_cache_path (str | None): 측면 프레임에 대한 세그멘테이션 결과를 캐싱할 경로입니다.

        Returns:
            tuple[dict, SideSegmentationResults]:
                dict: 권취 모니터링 결과를 담고 있는 dictionary 입니다.
                    ```python
                    {
                        "state_index": int, # 입력 받은 프레임의 state index입니다.
                                            # 1: 정상적으로 작동 중인 기본 상태입니다.
                                            # 2: 드럼 이동이 필요한 상태입니다.
                                            # 3: 드럼 사이드 부에 도달하여 방향 전환이 필요한 상태입니다.
                                            # 4: 프레임에서 권취 전선이 검출되지 않은 상태입니다.
                                            # 5: API 오류가 발생한 상태입니다.
                        "is_ng": bool, # 불량 발생 여부
                        "ng_boxes": (None, np.ndarray),  # 오류가 발생한 영역에 대한 박스들입니다. (N, 4) 형태이며 (x_left, y_top, width, height) 입니다.
                        "require_sponge_wrapping": bool, # 스펀지 보양 작업 필요 여부
                        "progress_rate": float, # 특정 방향으로의 권취 작업률 입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다.
                                                # 증가할 수록 작업 진행률이 높아 양끝에 가까워 진 상태입니다.
                        "segmentation_contours": {
                                "left_drum": (None, np.ndarray)  # 왼쪽 드럼에 대한 contour (N, 2) 입니다.
                                "right_drum": (None, np.ndarray)  # 오른쪽 드럼에 대한 contour (N, 2) 입니다.
                                "winding_cable": (None, np.ndarray)  # 권취 중인 전선에 대한 contour (N, 2) 입니다.
                                "drum_body": (None, np.ndarray)  # 드럼 몸통에 권취 완료된 전선에 대한 contour (N, 2) 입니다.
                        },
                        "drum_shifting_score": float,  # 드럼을 이동시킬지 여부에 대한 점수입니다. [0.0, 1.0] 범위의 실수 값입니다. 증가할 수록 드럼 이동 시점에 가까워진 상태입니다.
                        "tightening_score": float,  # 드럼 사이드 부에 도달하여 극단 밀착을 위한 이동에 대한 점수입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다.
                        "drum_swing_score": float,  # 두번째 바퀴 로직을 위한 이동 점수입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다.
                        "shift_strength": int,  # 드럼이 한 번에 이동하는 시간 (밀리초)입니다.
                                               # FirstSwing/SecondSwing 상태에서는 1100ms로 고정 출력됩니다.
                        "cable_shifting_box": (None, list[int]), # 권취 중인 전선이 위치할 수 있는 유효 범위를 나타내는 박스 좌표 [x_top, y_left, width, height]
                        "cable_moving_direction": (None, str), # 권취 중인 전선이 움직이는 방향입니다. ["left_to_right", "right_to_left"]
                        "drum_moving_direction": (None, str), # 드럼이 움직이는 방향입니다. ["left_to_right", "right_to_left"]
                                                            # cable_moving_direction의 반대 방향에 해당합니다.
                        "time": (None, dict), # 각 단계별 수행 시간입니다.
                                                            # "segmentation_time": segmentation 수행 시간 (ms)
                                                            # "direction_monitoring_time": 권취 방향 및 진행률 예측 시간 (ms)
                                                            # "sponge_monitoring_time": 스펀지 보양 예측 시간 (ms)
                                                            # "anomaly_monitoring_time": 권취 불량 예측 시간 (ms)
                                                            # "shifting_monitoring_time": 드럼 이동 여부 예측 시간 (ms)
                                                            # "total_time": 전체 수행 시간 (ms)
                        "error_message": (None, str), # API 오류 발생 시 오류 메시지입니다. 정상 작동 시 None입니다.
                    }
                    ```
                SideSegmentationResults: 측면 카메라에서 촬영된 프레임에 대한 세그멘테이션 결과입니다. (시각화 목적으로 반환)
        """
        if side_view_frame is None:
            raise MissingSideViewFrameError

        main_segmentation_results, side_segmentation_results = self.infer_segmentation_handler(
            frame, side_view_frame, segmentation_cache_path, side_segmentation_cache_path
        )
        results = self.monitor(main_segmentation_results, side_segmentation_results, side_view_frame)
        return (
            results,
            side_segmentation_results,
        )  # XXX: side_segmentation_results은 시각화를 위해 반환합니다.

    def update_monitoring_config(self, config_updates: dict | str) -> None:
        """MonitoringConfig 기반 설정값을 갱신합니다.

        각 모니터(`direction_monitor`, `sponge_monitor`, `anomaly_monitor`, `shift_monitor`)
        가 참조하는 섹션(`common`, `direction`, `anomaly`, `shift`, `sponge`, `rotation`)의
        dataclass 필드 값을 직접 갱신합니다.

        Args:
            config_updates: 다음 포맷 중 하나를 지원합니다.
                - 중첩 dict:
                    {
                      "common": {"shift_strength": 4000},
                      "direction": {"first_layer_width_ratio": 0.85},
                      "anomaly": {"slope_threshold": 0.75},
                      "shift": {"shift_alarm_interval": 7},
                      "sponge": {"sponge_wrapping_threshold": 92},
                    }
                - JSON 문자열: 위 구조의 JSON
                - 점 표기 키 dict:
                    {"common.shift_strength": 4000, "direction.first_layer_width_ratio": 0.85, ...}
        """

        def _flatten(prefix: str, obj: Any, out: dict) -> None:
            # 3가지 타입의 값을 점 표기 키로 변환
            if isinstance(obj, dict):
                for k, v in obj.items():
                    dotted = f"{prefix}.{k}" if prefix else str(k)
                    _flatten(dotted, v, out)
            else:
                out[prefix] = obj

        raw: Any = config_updates
        if isinstance(config_updates, str):
            raw = json.loads(config_updates)
        if not isinstance(raw, dict):
            raise TypeError("config_updates must be a dict or JSON string representing a dict")

        flattened: dict[str, Any] = {}
        _flatten("", raw, flattened)

        monitors = [
            self.direction_monitor,
            self.sponge_monitor,
            self.anomaly_monitor,
            self.shift_monitor,
        ]

        for dotted_key, value in flattened.items():
            if "." not in dotted_key:
                continue
            section, field_path = dotted_key.split(".", 1)
            section_config_attr = f"{section}_config"
            target_objs = []
            for monitor in monitors:
                cfg_obj = getattr(monitor, section_config_attr, None)
                if cfg_obj is not None:
                    target_objs.append(cfg_obj)
            if not target_objs:
                raise ValueError(f"Unknown section on monitors: {section}")

            if "." in field_path:
                raise ValueError("Nested attribute updates beyond section.field are not supported")

            try:
                first_obj = target_objs[0]
                if not hasattr(first_obj, field_path):
                    raise AttributeError(f"Unknown field on {section}: {field_path}")
                current = getattr(first_obj, field_path)
                if current is not None and not isinstance(value, type(current)):
                    raise TypeError(
                        f"Type mismatch: field type {type(current).__name__}, input type {type(value).__name__}"
                    )
                for obj in target_objs:
                    setattr(obj, field_path, value)
            except Exception as e:
                raise e

        # Config와 연관된 monitoring_options를 갱신합니다.
        current_options = self.monitoring_options
        self.monitoring_options = current_options

    def infer_segmentation_handler(
        self,
        frame: np.ndarray,
        side_view_frame: np.ndarray,
        segmentation_cache_path: str | None = None,
        side_segmentation_cache_path: str | None = None,
    ) -> tuple[SegmentationResults, SideSegmentationResults]:
        with self.timers["segmentation"]:
            main_segmentation_results, side_segmentation_results = self.segmentation_engine.inference(
                frame, side_view_frame, segmentation_cache_path, side_segmentation_cache_path
            )

        return main_segmentation_results, side_segmentation_results

    def monitor(
        self,
        segmentation_results: SegmentationResults,
        side_segmentation_results: SideSegmentationResults,
        side_view_frame: np.ndarray,
    ) -> dict:
        with self.timers["angle_monitoring"]:
            try:
                rotation_angle = self.angle_monitor.process_frame(
                    side_view_frame, side_segmentation_results
                )
            except Exception as e:
                raise AngleMonitoringError from e

        with self.timers["direction_monitoring"]:
            try:
                direction_monitoring_results = self.direction_monitor.process_frame(
                    segmentation_results,
                    self.previous_state,
                    rotation_angle,
                )
                logging.log_metrics(previous_state=self.previous_state)
            except Exception as e:
                raise DirectionMonitoringError from e

        with self.timers["sponge_monitoring"]:
            try:
                sponge_alarm_results = self.sponge_monitor.process_frame(
                    direction_monitoring_results["progress_rate"]
                )
            except Exception as e:
                raise SpongeMonitoringError from e

        with self.timers["anomaly_monitoring"]:
            try:
                anomaly_monitoring_results = self.anomaly_monitor.process_frame(
                    segmentation_results, direction_monitoring_results
                )
            except Exception as e:
                raise AnomalyMonitoringError from e

        with self.timers["shift_monitoring"]:
            try:
                shift_monitoring_results = self.shift_monitor.process_frame(
                    segmentation_results,
                    direction_monitoring_results,
                    side_segmentation_results,
                    rotation_angle,
                )

                if shift_monitoring_results.get("require_direction_change", False):
                    self.direction_monitor.change_direction()
                    self.sponge_monitor.reset_alarm()

                state = shift_monitoring_results.pop("current_state", None)
                logging.log_metrics(state=state)
                if state is not None:
                    self.previous_state = state

            except Exception as e:
                raise ShiftMonitoringError from e

        results = {
            "segmentation_contours": segmentation_results.segmentation_contours,
            "error_message": None,  # 정상 작동
            **direction_monitoring_results,
            **sponge_alarm_results,
            **anomaly_monitoring_results,
            **shift_monitoring_results,
        }
        results = self.check_state_index(results)
        if self.check_time:
            results = self.add_time_to_result(results)

        if self._require_engine_reset(segmentation_results):
            logging.log(
                f"No drum body detected in the last {self.common_config.num_empty_sequences_to_reset} frames. Resetting engine."
            )
            self.reset()

        return results

    def warmup(self, num_warmup=10):
        """모델을 미리 실행하여 초기화합니다."""
        # TODO: main / side 프레임 크기가 동일한지 확인 필요함.
        dummy_image_np = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for _ in range(num_warmup):
            self.segmentation_engine.main_inference_handler.infer_and_postprocess(
                images=[dummy_image_np]
            )
            self.segmentation_engine.side_inference_handler.infer_and_postprocess(
                images=[dummy_image_np]
            )

    def reset(self):
        self.angle_monitor.reset()
        self.direction_monitor.reset()
        self.shift_monitor.reset()
        self.anomaly_monitor.reset()

        self.timers: dict[str, Timer] = {
            "segmentation": Timer(is_enabled=self.common_config.timer_enabled),
            "angle_monitoring": Timer(is_enabled=self.common_config.timer_enabled),
            "direction_monitoring": Timer(is_enabled=self.common_config.timer_enabled),
            "sponge_monitoring": Timer(is_enabled=self.common_config.timer_enabled),
            "anomaly_monitoring": Timer(is_enabled=self.common_config.timer_enabled),
            "shift_monitoring": Timer(is_enabled=self.common_config.timer_enabled),
        }

        self._has_drum_body_detected_list: deque[bool] = deque(
            maxlen=self.common_config.num_empty_sequences_to_reset
        )

        self.previous_state: str | None = None

    def check_state_index(self, results: dict) -> dict:
        # 기본 상태는 권취 중인 전선이 정상적으로 작동 중인 상태입니다.
        state_index = self.common_config.state_index["winding"]

        # 권취 중인 전선이 양극단에 도달하여 드럼 이동 방향 전환이 필요한지 여부를 확인합니다.
        if results.pop("require_direction_change", False):
            state_index = self.common_config.state_index["change_direction"]
        # 권취 중인 전선이 유효 영역을 벗어나 이동해야 하는지 여부를 확인합니다. (방향 전환의 우선순위가 더 높음.)
        elif results.pop("require_drum_shift", False):
            state_index = self.common_config.state_index["shift_drum"]

        # 권취 중인 전선이 예측되지 않은 경우
        if results["segmentation_contours"]["winding_cable"] is None:
            state_index = self.common_config.state_index["no_winding_cable"]

        results["state_index"] = state_index
        return results

    def add_time_to_result(self, predictions: dict) -> dict:
        inference_time_dict = {f"{k}_time": v.time_stack[-1] for k, v in self.timers.items()}
        predictions["time"] = {k: round(v, 2) for k, v in inference_time_dict.items()}
        predictions["time"]["total_time"] = sum(inference_time_dict.values())

        return predictions

    def _require_engine_reset(self, segmentation_results: SegmentationResults) -> bool:
        """
        드럼 몸통이 연속적으로 예측되지 않는 경우 엔진을 초기화
        """
        _has_drum_body_detected = False
        if segmentation_results.drum_body_mask is not None:
            _has_drum_body_detected = True
        self._has_drum_body_detected_list.append(_has_drum_body_detected)

        if (
            not any(self._has_drum_body_detected_list)
            and len(self._has_drum_body_detected_list) == self.common_config.num_empty_sequences_to_reset
        ):
            return True
        return False

    @property
    def monitoring_options(self) -> dict:
        monitoring_options = {
            "check_time": self.check_time,
            "params": {
                **self.anomaly_monitor.monitoring_options,
                **self.direction_monitor.monitoring_options,
                **self.shift_monitor.monitoring_options,
                **self.sponge_monitor.monitoring_options,
            },
        }
        return monitoring_options

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        # Set check_time if provided
        if "check_time" in options:
            self.check_time = options["check_time"]

        if "params" in options:
            for monitor in [
                self.angle_monitor,
                self.direction_monitor,
                self.anomaly_monitor,
                self.shift_monitor,
                self.sponge_monitor,
            ]:
                try:
                    current = monitor.monitoring_options
                    # Merge current params with new params (new params override)
                    merged_params = {**current, **options["params"]}
                    monitor.monitoring_options = {"params": merged_params}
                except Exception as e:
                    # Skip monitors that fail to update
                    logging.log(f"Warning: Could not update monitor options: {e}")
                    pass

    @property
    def default_monitoring_options(self) -> dict:
        default_monitoring_options = {
            "params": {
                **self.direction_monitor.default_monitoring_options,
                **self.anomaly_monitor.default_monitoring_options,
                **self.shift_monitor.default_monitoring_options,
                **self.sponge_monitor.default_monitoring_options,
            },
        }
        return default_monitoring_options

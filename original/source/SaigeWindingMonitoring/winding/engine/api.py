from __future__ import annotations

from typing import Any

import numpy as np
from SaigeToolkit.error.base import SaigeWindingMonitoringError
from SaigeToolkit.error.handler import get_error_handler

from . import monitoring
from ..modules import logging
from ..engine.config import DirectionSettings, ShiftSettings
from ..modules.process_state import ProcessState

error_handler = get_error_handler(SaigeWindingMonitoringError)


class WindingMonitoringEngine:
    """권취 상태를 감지하기 위한 WindingMonitoringEngine class 입니다.

    Usage:
        ```python
        expected_output = {
            "state_index": int,
            "is_ng": bool,
            "ng_boxes": (type(None), np.ndarray),
            "require_sponge_wrapping": bool,
            "progress_rate": float,
            "segmentation_contours": {
                "left_drum": (type(None), np.ndarray),
                "right_drum": (type(None), np.ndarray),
                "winding_cable": (type(None), np.ndarray),
                "drum_body": (type(None), np.ndarray)
            },
            "drum_shifting_score": float,
            "tightening_score": float,
            "drum_swing_score": float,
            "cable_shifting_box": (type(None), list),
            "cable_moving_direction": (type(None), str),
            "drum_moving_direction": (type(None), str),
            "time": (type(None), dict),
        }


        # 1. WindingMonitoringEngine 빌드
        winding_engine_build_config = {
            "segmentation": {
                "checkpoint_path": DEMO_CHECKPOINT,
                "password": PASSWORD,
                "device": DEVICE,
            },
        }
        error, message, winding_engine = WindingMonitoringEngine.build(winding_engine_build_config)
        assert error >= 0, f"[CODE] ({error}) {message}"
        print("WindingMonitoringEngine built successfully")

        monitoring_options = {
            "params.sponge_wrapping_threshold": 90,
            "params.shift_sensitivity": 50,
            "params.anomaly_sensitivity": 50,
            "params.shift_alarm_interval": 5,
            "params.shift_strength": 3000,
        }

        # 2. 기본 모니터링 옵션 확인
        for key in monitoring_options:
            error, message, value = winding_engine.get_default_monitoring_option(key)
            assert error >= 0, f"[CODE] ({error}) {message}"
            print(f"    Default monitoring option '{key}': {value}")

        # 3. 모니터링 옵션 변경
        for key, value in monitoring_options.items():
            error, message, _ = winding_engine.set_monitoring_option(key, value)
            assert error >= 0, f"[CODE] ({error}) {message}"

        # 4. 모니터링 옵션이 잘 변경되었는지 확인
        print("Monitoring options:")
        for key in monitoring_options:
            error, message, value = winding_engine.get_monitoring_option(key)
            assert error >= 0, f"[CODE] ({error}) {message}"
            print(f"    {key}: {value}")

        # 5. 권취 모니터링 수행
        image_paths = os.listdir(DATA)
        image_paths = [
            path for path in image_paths if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        for i, image_path in enumerate(image_paths):
            print(f"Processing frame: {image_path}")

            # 프레임 별 검사 중간에 monitoring option 변경.
            if i == len(image_paths) // 2:
                monitoring_options = {
                    "params.sponge_wrapping_threshold": 95,
                    "params.anomaly_sensitivity": 40,
                    "params.shift_sensitivity": 70,
                    "params.shift_alarm_interval": 10,
                    "params.shift_strength": 5000,
                }
                for key, value in monitoring_options.items():
                    error, message, _ = winding_engine.set_monitoring_option(key, value)
                    assert error >= 0, f"[CODE] ({error}) {message}"
                    print(f"    Monitoring option '{key}' changed to {value}")

            frame = cv2.imread(os.path.join(DATA, image_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            error, message, result = winding_engine.process_frame(frame)
            assert error >= 0, f"[CODE] ({error}) {message}"

            # 결과 확인
            for key, expected_type in expected_output.items():
                assert key in result, f"Key '{key}' not found in result"
                assert isinstance(result[key], expected_type), f"Key '{key}' has wrong type"
                if isinstance(result[key], dict):
                    print(f"    {key}:")
                    for _key, _value in result[key].items():
                        if isinstance(_value, np.ndarray):
                            print(f"        {_key}: {_value.shape}")
                        else:
                            print(f"        {_key}: {_value}")
                elif isinstance(result[key], np.ndarray):
                    print(f"    {key}: {result[key].shape}")
                else:
                    print(f"    {key}: {result[key]}")

        # 6. 새로운 뷰에서 권취 모니터링 수행할 경우,
        error, message, _ = winding_engine.reset()
        assert error >= 0, f"[CODE] ({error}) {message}"
        print("Monitoring in a new view")

        for i, image_path in enumerate(image_paths):
            print(f"Processing frame: {image_path}")
            frame = cv2.imread(os.path.join(DATA, image_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            error, message, results = winding_engine.process_frame(frame)
            assert error >= 0, f"[CODE] ({error}) {message}"

            # 결과 확인
            for key, expected_type in expected_output.items():
                assert key in result, f"Key '{key}' not found in result"
                assert isinstance(result[key], expected_type), f"Key '{key}' has wrong type"
                if isinstance(result[key], dict):
                    print(f"    {key}:")
                    for _key, _value in result[key].items():
                        if isinstance(_value, np.ndarray):
                            print(f"        {_key}: {_value.shape}")
                        else:
                            print(f"        {_key}: {_value}")
                elif isinstance(result[key], np.ndarray):
                    print(f"    {key}: {result[key].shape}")
                else:
                    print(f"    {key}: {result[key]}")
        ```

    """

    def __init__(self, winding_engine: monitoring.WindingMonitoringEngine):
        self.winding_engine = winding_engine

    @classmethod
    @error_handler
    def build(cls, config: dict) -> WindingMonitoringEngine:
        """WindingMonitoringEngine class의 instance를 생성합니다.

        Args:
            config (dict): WindingMonitoringEngine를 생성하기 위한 config가 담겨있는 dictionary 입니다.
                ```python
                {
                    "segmentation": dict, # Segmentation InferenceHandler를 생성하기 위한 config
                        {
                            "checkpoint_path": str, # segmentation model 체크포인트 경로
                            "password": str, # 체크포인트 패스워드
                            "device": int | str, # GPU 번호 (int) or "cpu" (str) (default "cpu")
                        }
                    "monitoring_options": Optional[dict], # 권취 모니터링 옵션 config (default None)
                                                          # 구조는 `get_default_monitoring_option`의 Keys 참고
                }
                ```

        Returns:
            WindingMonitoringEngine: build가 완료된 winding.WindingMonitoringEngine class의 instance를 반환합니다.
        """
        winding_engine = monitoring.WindingMonitoringEngine.build(**config)
        return cls(winding_engine=winding_engine)

    @error_handler
    def get_default_monitoring_option(self, key: str) -> Any:
        """권취 모니터링 옵션의 기본 설정 값을 반환합니다.

        Args:
            key (str): 옵션 key

        Returns:
            Any: 옵션 value

        Keys:
            사용 가능한 key와 value 목록:
                ```python
                {
                    "params.sponge_wrapping_threshold": int, # 스펀지 보양을 시작할 때의 진행률을 결정하는 [1, 100] 사이의 값.
                    "params.shift_sensitivity": int, # 드럼 이동을 감지하기 위한 [1, 100] 사이의 민감도.
                    "params.anomaly_sensitivity": int, # 권취 불량을 감지하기 위한 [1, 100] 사이의 민감도.
                    "params.shift_alarm_interval": 5,  # 드럼 이동 신호 주기를 조절하기 위한 1 이상의 정수 값. (단위: 초)
                    "params.shift_strength": 3000,  # 드럼이 한 번에 이동하는 시간 (밀리초).
                }
                ```

            사용 예시:
                ```python
                winding_engine.get_default_monitoring_option(key="params.sponge_wrapping_threshold")
                ```

        """
        return self.winding_engine.get_default_monitoring_option(key)

    @error_handler
    def get_monitoring_option(self, key: str) -> Any:
        """현재 설정된 권취 모니터링 옵션 값을 읽습니다.

        Args:
            key (str): 옵션 key

        Returns:
            Any: 옵션 value

        Keys:
            `get_default_monitoring_option`의 Keys 참고
        """
        return self.winding_engine.get_monitoring_option(key)

    @error_handler
    def set_monitoring_option(self, key: str, value: Any) -> None:
        """권취 모니터링 옵션을 설정합니다.

        Args:
            key (str): 설정하고자 하는 옵션 key
            value (Any): 성정하고자 하는 옵션 value

        Returns:
            None: None

        Keys:
            `get_default_monitoring_option`의 Keys 참고.
        """
        return self.winding_engine.set_monitoring_option(key, value)

    @error_handler
    def update_monitoring_config(self, config_updates: dict | str) -> None:
        """모니터별 MonitoringConfig 값을 엔진에 업데이트합니다.

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
                      "rotation": {"optical_flow_method": "Dummy"}
                    }
                - JSON 문자열: 위 구조의 JSON
                - 점 표기 키 dict:
                    {"common.shift_strength": 4000, "direction.first_layer_width_ratio": 0.85, ...}

        """
        return self.winding_engine.update_monitoring_config(config_updates)

    @error_handler
    def process_frame(self, frame: np.ndarray, side_view_frame: np.ndarray) -> dict:
        """입력 받은 프레임에 대해 권취 모니터링을 수행합니다.

        Args:
            frame (np.ndarray): 권취 모니터링을 수행할 프레임 입니다.
            side_view_frame (np.ndarray): 측면 뷰 프레임입니다. 드럼의 회전 각도를 계산하는데 사용됩니다.

        Returns:
            dict: 권취 모니터링 결과를 담고 있는 dictionary 입니다.
                ```python
                {
                    "state_index": int, # 입력 받은 프레임의 state index입니다.
                                        # 1: 정상적으로 작동 중인 기본 상태입니다.
                                        # 2: 드럼 이동이 필요한 상태입니다.
                                        # 3: 드럼 사이드 부에 도달하여 방향 전환이 필요한 상태입니다.
                                        # 4: 프레임에서 권취 전선이 검출되지 않은 상태입니다.
                                        # 5: API 오류가 발생한 상태입니다.
                    "process_state": str, # 현재 권취 프로세스 상태입니다.
                                          # winding: 일반 권취 상태
                                          # waiting_tightening: 밀착 대기
                                          # tightening_start: 밀착 시작
                                          # tightening: 밀착 진행 중
                                          # tightening_end: 밀착 종료
                                          # protrusion_detection_start: 돌출 판단 시작
                                          # protrusion_detection: 돌출 판단 중
                                          # protrusion_detection_end: 돌출이 감지되어, 돌출 판단 끝
                                          # waiting_for_first_swing: 극단 최종 권취 단계
                                          # first_swing_start: 스윙 시작
                                          # first_swing: 스윙 중
                                          # first_swing_end: 스윙 끝
                                          # swing_term: 스윙 복귀 대기
                                          # second_swing_start: 스윙 복귀 시작
                                          # second_swing: 스윙 복귀 중
                                          # second_swing_end: 스윙 복귀 끝
                    "is_ng": bool, # 불량 발생 여부
                    "ng_boxes": (None, np.ndarray),  # 오류가 발생한 영역에 대한 박스들입니다. (N, 4) 형태이며 (x_top, y_left, width, height) 입니다.
                    "require_sponge_wrapping": bool, # 스펀지 보양 작업 필요 여부
                    "progress_rate": float, # 특정 방향으로의 권취 작업률 입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다.
                                            # 증가할 수록 작업 진행률이 높아 양끝에 가까워 진 상태입니다.
                    "segmentation_contours": {
                            "left_drum": (None, np.ndarray)  # 왼쪽 드럼에 대한 contour (N, 2) 입니다.
                            "right_drum": (None, np.ndarray)  # 오른쪽 드럼에 대한 contour (N, 2) 입니다.
                            "winding_cable": (None, np.ndarray)  # 권취 중인 전선에 대한 contour (N, 2) 입니다.
                            "drum_body": (None, np.ndarray)  # 드럼 몸통에 권취 완료된 전선에 대한 contour (N, 2) 입니다.
                    },
                    "drum_shifting_score": float,  # 드럼을 이동시킬지 여부에 대한 점수입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다. 증가할 수록 드럼 이동 시점에 가까워진 상태입니다.
                    "tightening_score": float,  # 드럼 사이드 부에 도달하여 극단 밀착을 위한 이동에 대한 점수입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다.
                    "drum_swing_score": float,  # 두번째 바퀴 로직을 위한 이동 점수입니다. -1.0 또는 [0.0, 1.0] 범위의 실수 값입니다.
                    "shift_strength": int,  # 드럼이 한 번에 이동하는 시간 (밀리초)입니다.
                                           # FirstSwing/SecondSwing 상태에서는 default_swing_shift_strength로 고정 출력됩니다.
                    "cable_shifting_box": (None, list[int]), # 권취 중인 전선이 위치할 수 있는 유효 범위를 나타내는 박스 좌표 [x_top, y_left, width, height]
                    "cable_moving_direction": (None, str), # 권취 중인 전선이 움직이는 방향입니다. ["left_to_right", "right_to_left"]
                    "drum_moving_direction": (None, str), # 드럼이 움직이는 방향입니다. ["left_to_right", "right_to_left"]
                                                        # cable_moving_direction의 반대 방향에 해당합니다.
                    "rotation_angle": (None, float), # 드럼의 회전 각도입니다. 단위는 도(degree)입니다.
                                                     # 드럼 회전각도를 측정하기 시작할 때부터 누적된 각도입니다.
                                                     # process_state가 "waiting_tightening", "waiting_for_first_swing", "swing_term"일 때만 측정됩니다.
                                                     # 회전 각도를 구하지 않을 때는 None입니다.
                    "reference_angle": (None, float), # 도달해야할 회전 각도입니다. 단위는 도(degree)입니다.
                                                      # process_state가 "waiting_tightening", "waiting_for_first_swing", "swing_term"일 때만 값이 나오고
                                                      # 그 외의 상태에서는 None입니다.
                    "cumulative_angle": (None, float), # 누적 회전 각도입니다. 단위는 도(degree)입니다.
                                                       # process state에 상관없이 출력됩니다.
                                                       # 연구 및 분석 용도로 사용됩니다.
                    "time": (None, dict), # 각 단계별 수행 시간입니다.
                                                        # "segmentation_time": segmentation 수행 시간 (ms)
                                                        # "direction_monitoring_time": 권취 방향 및 진행률 예측 시간 (ms)
                                                        # "anomaly_monitoring_time": 권취 불량 예측 시간 (ms)
                                                        # "shifting_monitoring_time": 드럼 이동 여부 예측 시간 (ms)
                                                        # "total_time": 전체 수행 시간 (ms)
                    "error_message": (None, str), # API 오류 발생 시 오류 메시지입니다. 정상 작동 시 None입니다.
                }
                ```
        """
        logging.call_start()

        try:
            segmentation_results, side_segmentation_results = (
                self.winding_engine.infer_segmentation_handler(frame, side_view_frame)
            )
            result = self.winding_engine.monitor(
                segmentation_results, side_segmentation_results, side_view_frame
            )
            logging.dump(
                frame=frame,
                side_view_frame=side_view_frame,
                segmentation_results=segmentation_results,
                side_segmentation_results=side_segmentation_results,
                results=result,
            )
        except Exception as e:
            logging.log(f"API error occurred: {e}")
            result = {
                "state_index": 5,  # API 오류 상태
                "process_state": ProcessState.WINDING.value,
                "is_ng": False,
                "ng_boxes": None,
                "require_sponge_wrapping": False,
                "progress_rate": DirectionSettings.default_progress_rate,
                "segmentation_contours": segmentation_results.segmentation_contours,
                "drum_shifting_score": ShiftSettings.default_shifting_score,
                "tightening_score": ShiftSettings.default_tightening_score,
                "drum_swing_score": ShiftSettings.default_tightening_score,
                "shift_strength": self.winding_engine.shift_monitor.shift_strength,
                "cable_shifting_box": None,
                "cable_moving_direction": None,
                "drum_moving_direction": None,
                "rotation_angle": None,
                "reference_angle": None,
                "cumulative_angle": None,
                "time": None,
                "error_message": str(e),
            }

        logging.call_end()

        return result

    @error_handler
    def reset(self) -> None:
        """
        WindingMonitoringEngine을 새롭게 빌드하지 않고, 새로운 뷰에서 모니터링을 할 경우에 `process_frame`에 앞서 호출해야 합니다.

        Returns:
            None: None
        """
        return self.winding_engine.reset()

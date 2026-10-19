from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

# 개소별 파라미터 관리
SITE_CONFIGS = {
    "taping4": {
        # RotationSettings
        "angle_scale_factor": 0.6,
        "drum_center_point_range": (2250, 700, 2300, 2100),
        # ShiftSettings
        "shift_alarm_threshold": 0.95,
        "progress_rate_threshold": 0.998,
        "tightening_delay_angle": 60.0,
        "tightening_range_factor": 1.0,
        "swing_term_angles": [20.0, 50.0],
        "swing_excess_factor": 0.5,
        "swing_score_thresholds": [0.2, 0.2],
    },
    "taping7": {
        # RotationSettings
        "angle_scale_factor": 0.54,
        "drum_center_point_range": (1650, 700, 1700, 1700),
        # ShiftSettings
        "shift_alarm_threshold": 0.9,
        "progress_rate_threshold": 0.995,
        "tightening_delay_angle": 65.0,
        "tightening_range_factor": 1.2,
        "swing_term_angles": [20.0, 30.0],
        "swing_excess_factor": 0.4,
        "swing_score_thresholds": [0.2, 0.25],
    },
}

# 현재 개소 설정
CURRENT_SITE = "taping4"


@dataclass
class CommonSettings:
    num_empty_sequences_to_reset: int = 30  # 엔진 초기화를 위한 드럼 몸통 미검출 시퀀스 수
    body_height_ratio: float = 0.6  # 첫번째 바퀴 권취 여부 판단을 위한 드럼 몸통 높이 비율
    roi_narrowing_width: int = 40  # 드럼 몸통 ROI(관심영역) 좌우 범위 좁히기

    timer_enabled: bool = True  # 타이머 활성화 여부
    state_index: ClassVar[dict[str, int]] = {
        "winding": 1,  # 정상적으로 작동 중인 기본 상태입니다.
        "shift_drum": 2,  # 드럼 이동이 필요한 상태입니다.
        "change_direction": 3,  # 드럼 사이드 부에 도달하여 방향 전환이 필요한 상태입니다.
        "no_winding_cable": 4,  # 프레임에서 권취 전선이 검출되지 않은 상태입니다.
        "error": 5,  # API 오류가 발생한 상태입니다.
    }

    shift_strength: int = 3000  # 드럼이 한 번에 이동하는 시간 (밀리초). 기본값은 3초.
    downscale_factor: int = 4  # 다운스케일 팩터

    edge_region_ratio: float = 0.2  # 엣지 영역 비율

    # cable thickness
    cable_thickness_lookup_table: ClassVar[list[int]] = [50, 75, 85, 95, 105]

    # angle calibration
    max_calibration_ratio: float = 0.00003  # 각도당 진행률 변화량의 최대값


@dataclass
class AnomalySettings:
    slope_threshold: float = 0.8  # 권취 불량 기울기 기본값
    anomaly_sensitivity: int = 50  # 권취 불량을 감지하기 위한 [1, 100] 사이의 민감도 기본값
    sensitivity_scaling_slope_limit: float = 0.3  # 민감도 조정 시, 기울기 임계값 범위의 크기 조정 비율
    ng_box_y_padding: int = 30  # NG 박스 상하 여유 패딩
    gradient_smoothing_lambda: float = 2.0  # 기울기 스무딩을 위한 람다 값
    overlap_iou_threshold: float = 0.3  # 권취 불량 마스킹 IOU 임계값


@dataclass
class DirectionSettings:
    # direction
    first_layer_width_ratio: float = 0.9  # 첫번째 층 권취 여부를 판단하기 위한 드럼 몸통 너비 비율
    direction_confirmation_frames: int = 5  # 진행 방향을 결정하기 위한 연속 프레임 수
    direction_change_threshold: float = 0.9  # 방향 전환 임계값
    min_pixel_for_protrusion: int = 1  # 방향 판단을 위한 극단 돌출부 높이 차이 임계값 (픽셀)

    # progress
    progress_rate_history_length: int = 300  # 진행률 히스토리 최대 길이
    progress_rate_max_change: float = 1000  # 진행률 히스토리에서 최대 변화량
    progress_rate_ema_alpha: float = 0.5  # 진행률 스무딩을 위한 EMA 알파 값
    default_progress_rate: float = -1.0  # 진행률이 계산되지 않았을 때의 기본값
    left_drum_corner_angle_range: ClassVar[list[float]] = [110, 160]  # 왼쪽 드럼 코너 포인트의 각도 범위
    right_drum_corner_angle_range: ClassVar[list[float]] = [
        20,
        70,
    ]  # 오른쪽 드럼 코너 포인트의 각도 범위
    correct_progress_rate_start: float = 0.9  # 진행률 보정 시작 임계값


@dataclass
class SpongeWrappingSettings:
    sponge_wrapping_threshold: int = 90  # 스펀지 보양 알림 진행률


@dataclass
class ShiftSettings:
    # winding shift
    shift_sensitivity: int = 50  # 드럼 이동을 감지하기 위한 [1, 100] 사이의 민감도.
    shifting_box_ratio: float = 1.35  # 드럼 이동 유효 영역 박스의 크기 조정 비율.
    right_over_left_expansion_ratio: float = 2  # 좌/우 여유 폭 가중치 비율 (left : right = 1 : ratio)
    sensitivity_scaling_ratio_limit: float = 0.2  # 민감도 조정 시, 드럼 이동 감지 영역의 크기 조정 비율.
    shift_alarm_interval: int = 5  # 드럼 이동 알람 주기
    default_shifting_score: float = 0.0  # 드럼 이동 점수 초기값
    shift_alarm_threshold: float = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["shift_alarm_threshold"]
    )  # 드럼 이동 점수 임계값
    min_turns_for_shifting_box: float = 2  # 유효 영역 생성을 위한 최소 권취 바퀴 수
    is_mid_start_progress_threshold: float = (
        0.10  # 드럼 이동 유효 영역 박스 설정 시, 이미 케이블이 감기는 도중에 시작한 것으로 판단하는 진행률 임계값
    )
    default_swing_shift_strength: int = 1100  # 스윙 단계에서 출력할 shift_strength 기본값 (밀리초)

    # tightening
    progress_rate_threshold: float = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["progress_rate_threshold"]
    )  # 양극단 드럼 이동 점수 계산을 위한 권취 진행률 임계값
    default_tightening_score: float = -1.0  # 극단 이동 점수 초기값
    tightening_delay_angle: float = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["tightening_delay_angle"]
    )  # 밀착 대기 각도 임계값 (도)
    tightening_range_factor: float = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["tightening_range_factor"]
    )  # 조임 범위 조정 비율
    tightening_score_thresholds: ClassVar[list[float]] = [
        0.05,
        0.05,
    ]  # 측면 이동 완료 점수 임계값 [left, right]
    tightening_ready_distance_factor: float = (
        1.5  # 밀착 준비 판단을 위한 거리 임계값 (distance_ema <= cable_width * factor)
    )
    tightening_early_trigger_distance_factor: float = (
        1.3  # 밀착 조기 시작을 위한 거리 임계값 (distance_ema <= cable_width * factor)
    )

    # protrusion
    protrusion_confirmation_window: int = 15  # 극단 돌출부 결정을 위한 최근 프레임 수
    protrusion_confirmation_frames: int = 12  # 극단 돌출부 결정을 위한 돌출 프레임 수
    min_pixel_for_protrusion: int = 6  # 돌출 판단을 위한 극단 돌출부 높이 차이 임계값 (픽셀)

    angle_thresholds: ClassVar[list[float]] = [
        35.0,
        25.0,
    ]  # 극단 한바퀴 권취 로직 종료를 판단하기 위한 각도 임계값 [left degree, right degree]

    # swing shift
    swing_score_thresholds: list[float] = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["swing_score_thresholds"]
    )  # 두번째 바퀴 이동 완료 점수 임계값 [left, right]
    target_swing_score: float = 0.2  # 목표 스윙 점수
    swing_shift_alarm_interval: int = 2  # 드럼 스윙 이동 알람 주기
    swing_range_factors: list[float] = field(
        default_factory=lambda: [1.0, 1.2]
    )  # 드럼 스윙 범위 조정 비율 [left, right]. * cable_width
    swing_term_angles: list[float] = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["swing_term_angles"]
    )  # 스윙 대기 각도 임계값 (도). [left degree, right degree]
    swing_excess_factor: float = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["swing_excess_factor"]
    )  # 스윙 범위 초과 이동 방지
    reset_delay_factor: int = (
        3  # "second tightening -> winding" / "second swing -> winding" 리셋 대기시간 조정 비율 (shift_strength * factor). 기본값은 2배.
    )
    reset_delay_angles: ClassVar[list[float]] = [
        360.0,
        60.0,
    ]  # 스윙 복귀 후, winding 상태로 리셋되기 위한 최소 회전 각도 (도). [left degree, right degree]. 기본값은 [180도, 60도]
    protrusion_angle_thresholds: ClassVar[list[float]] = [
        99999.0,
        99999.0,
    ]  # 돌출 감지가 제대로 되지 않았을 때 강제로 넘어가기 위한 각도 임계값 [left degree, right degree]


@dataclass
class RotationSettings:
    camera_distortion_coeffs: ClassVar[tuple[float, float, float, float, float]] = (
        -0.7,
        0.0,
        0.0,
        0.0,
        0.0,
    )  # 이미지 왜곡을 보정하기 위한 카메라 계수 (k1, k2, p1, p2, k3)
    drum_center_point: ClassVar[tuple[int, int]] | None = (
        None  # 드럼 중심 좌표 (픽셀) (x, y) # XXX: 십자가 없는 드럼을 위해 추가됨. taping7 기준값: (1755, 1100)
    )
    # "이미지 왜곡이 보정된 상태에서" 드럼 중심이 위치할 수 있는 대략적인 범위를 나타내는 직선 (x1, y1) - (x2, y2)
    drum_center_point_range: tuple[int, int, int, int] = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["drum_center_point_range"]
    )  # 드럼 중심 범위
    angle_scale_factor: float = field(
        default_factory=lambda: SITE_CONFIGS[CURRENT_SITE]["angle_scale_factor"]
    )  # 회전 각도 스케일링 팩터
    angle_threshold: float = 20.0  # 원 중심에 대한 방사선 각도 임계값 (도)
    min_line_length: float = 100.0  # 선 검출을 위한 최소 선 길이 (픽셀)
    max_line_length: float = 3000.0  # 선 검출을 위한 최대 선 길이 (픽셀)
    min_line_angle: float = 130.0  # 선 검출을 위한 최소 각도 (도)
    max_line_angle: float = 220.0  # 선 검출을 위한 최대 각도 (도) # 태핑 7호기 큰드럼이면: 230.0
    enable_line_merging: bool = True  # 인접한 선 병합 활성화
    merge_angle_threshold: float = 10.0  # 선 병합을 위한 각도 차이 임계값 (도)
    merge_distance_threshold: float = 50.0  # 선 병합을 위한 거리 임계값 (픽셀)
    merge_collinearity_threshold: float = 10.0  # 선 병합을 위한 공선성 임계값
    rotation_direction: str = "clockwise"  # 회전 방향 ("clockwise", "counterclockwise")
    max_merge_iterations: int = 3  # 선 병합 최대 반복 횟수


@dataclass
class LoggingSettings:
    log_dir: str | None = None  # "D:\\Saige\\Dump"  # None인 경우, 로깅 기능 비활성화
    image_resolution: tuple[int, int] = (1280, 960)
    image_quality: int = 80
    enable_state_archiving: bool = True  # 상태 기반 자동 압축 기능 활성화 여부
    archive_buffer_size: int = 15  # 압축 시 앞뒤로 포함할 프레임 수


@dataclass
class MonitoringConfig:
    """모니터링 설정

    개소별 설정은 CURRENT_SITE를 변경하여 조정합니다.
    실제 엔진 사용 시 test_utils.configure_site_parameters()를 통해 동적 설정 가능합니다.
    """

    common: CommonSettings = field(default_factory=CommonSettings)
    anomaly: AnomalySettings = field(default_factory=AnomalySettings)
    direction: DirectionSettings = field(default_factory=DirectionSettings)
    sponge: SpongeWrappingSettings = field(default_factory=SpongeWrappingSettings)
    shift: ShiftSettings = field(default_factory=ShiftSettings)
    rotation: RotationSettings = field(default_factory=RotationSettings)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("개소별 설정(Site Configuration) 테스트")
    print("=" * 70)

    # 현재 개소(CURRENT_SITE) 기본 설정 확인
    print(f"\n현재 개소: CURRENT_SITE = '{CURRENT_SITE}'")
    print("-" * 70)
    config = MonitoringConfig()
    print(f"  Angle Scale Factor: {config.rotation.angle_scale_factor}")
    print(f"  Drum Center Range: {config.rotation.drum_center_point_range}")
    print(f"  Shift Alarm Threshold: {config.shift.shift_alarm_threshold}")
    print(f"  Progress Rate Threshold: {config.shift.progress_rate_threshold}")
    print(f"  Tightening Delay Angle: {config.shift.tightening_delay_angle}")
    print(f"  Tightening Range Factor: {config.shift.tightening_range_factor}")
    print(f"  Swing Term Angles: {config.shift.swing_term_angles}")
    print(f"  Swing Score Thresholds: {config.shift.swing_score_thresholds}")

    # 사용 가능한 개소 목록
    print(f"\n사용 가능한 개소 목록:")
    print("-" * 70)
    for site in SITE_CONFIGS.keys():
        indicator = " (현재)" if site == CURRENT_SITE else ""
        print(f"  - {site}{indicator}")

    # 개소별 설정 비교
    print(f"\n개소별 파라미터 비교:")
    print("-" * 70)
    print(f"{'파라미터':<30} {'taping4':<20} {'taping7':<20}")
    print("-" * 70)
    params = [
        "angle_scale_factor",
        "shift_alarm_threshold",
        "progress_rate_threshold",
        "tightening_delay_angle",
        "tightening_range_factor",
        "swing_term_angles",
        "swing_score_thresholds",
    ]
    for param in params:
        t4_val = str(SITE_CONFIGS["taping4"][param])
        t7_val = str(SITE_CONFIGS["taping7"][param])
        print(f"{param:<30} {t4_val:<20} {t7_val:<20}")

    # CURRENT_SITE 변경 방법 안내
    print(f"\n개소 변경 방법:")
    print("-" * 70)
    print("  1. config.py 상단의 'CURRENT_SITE' 변경")
    print("  2. 변경 후 MonitoringConfig()의 기본값이 자동으로 해당 개소 설정 적용")
    print("  3. 테스트 코드에서는 build_engine_from_config(site=SITE_NAME) 사용")

    print("\n" + "=" * 70)

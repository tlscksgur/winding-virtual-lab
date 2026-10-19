from enum import Enum


class ProcessState(Enum):
    # 현재 상태를 나타냄.
    # UI 로그를 위한 용도. (이를 위해서 세분화되었음)

    WINDING = "winding"
    WAITING_TIGHTENING = "waiting_tightening"
    TIGHTENING_START = "tightening_start"
    TIGHTENING = "tightening"
    TIGHTENING_END = "tightening_end"
    PROTRUSION_DETECTION_START = "protrusion_detection_start"
    PROTRUSION_DETECTION = "protrusion_detection"
    PROTRUSION_DETECTION_END = "protrusion_detection_end"
    WAITING_FOR_FIRST_SWING = "waiting_for_first_swing"
    FIRST_SWING_START = "first_swing_start"
    FIRST_SWING = "first_swing"
    FIRST_SWING_END = "first_swing_end"
    SWING_TERM = "swing_term"
    SECOND_SWING_START = "second_swing_start"
    SECOND_SWING = "second_swing"
    SECOND_SWING_END = "second_swing_end"
    AUTO_SWING_SKIP_RESET = "auto_swing_skip_reset"

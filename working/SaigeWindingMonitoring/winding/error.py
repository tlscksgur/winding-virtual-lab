from SaigeToolkit.error.base import SaigeWindingMonitoringError


class SegmentationModelBuildError(SaigeWindingMonitoringError):
    code = "001"
    message = "Error building segmentation model"


class SegmentationModelInferenceError(SaigeWindingMonitoringError):
    code = "002"
    message = "Error running segmentation model inference & postprocessing"


class DirectionMonitoringError(SaigeWindingMonitoringError):
    code = "003"
    message = "Unexpected error in direction monitoring"


class AnomalyMonitoringError(SaigeWindingMonitoringError):
    code = "004"
    message = "Unexpected error in anomaly monitoring"


class ShiftMonitoringError(SaigeWindingMonitoringError):
    code = "005"
    message = "Unexpected error in shift monitoring"


class SpongeMonitoringError(SaigeWindingMonitoringError):
    code = "006"
    message = "Unexpected error in sponge monitoring"


class AngleMonitoringError(SaigeWindingMonitoringError):
    code = "007"
    message = "Unexpected error in angle monitoring"


class MonitoringOptionSpongeWrappingThresholdTypeError(SaigeWindingMonitoringError):
    code = "100"
    message = "Invalid type for monitoring option 'sponge_wrapping_threshold'"


class MonitoringOptionSpongeWrappingThresholdValueError(SaigeWindingMonitoringError):
    code = "101"
    message = "Invalid value for monitoring option 'sponge_wrapping_threshold'"


class MonitoringOptionShiftSensitivityTypeError(SaigeWindingMonitoringError):
    code = "102"
    message = "Invalid type for monitoring option 'shift_sensitivity'"


class MonitoringOptionShiftSensitivityValueError(SaigeWindingMonitoringError):
    code = "103"
    message = "Invalid value for monitoring option 'shift_sensitivity'"


class MonitoringOptionAnomalySensitivityTypeError(SaigeWindingMonitoringError):
    code = "104"
    message = "Invalid type for monitoring option 'anomaly_sensitivity'"


class MonitoringOptionAnomalySensitivityValueError(SaigeWindingMonitoringError):
    code = "105"
    message = "Invalid value for monitoring option 'anomaly_sensitivity'"


class MonitoringOptionShiftAlarmIntervalTypeError(SaigeWindingMonitoringError):
    code = "106"
    message = "Invalid type for monitoring option 'shift_alarm_interval'"


class MonitoringOptionShiftAlarmIntervalValueError(SaigeWindingMonitoringError):
    code = "107"
    message = "Invalid value for monitoring option 'shift_alarm_interval'"


class MissingSideViewFrameError(SaigeWindingMonitoringError):
    code = "200"
    message = "Side view frame is required but not provided."

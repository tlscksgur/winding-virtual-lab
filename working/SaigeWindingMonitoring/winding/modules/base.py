from typing import Any
from abc import ABC, abstractmethod

from SaigeToolkit.util.config import get_tree_node, set_tree_node

from ..engine.config import MonitoringConfig


class BaseMonitor(ABC):
    def __init__(
        self,
        monitoring_config: MonitoringConfig,
    ) -> None:
        self.common_config = monitoring_config.common
        self.direction_config = monitoring_config.direction
        self.anomaly_config = monitoring_config.anomaly
        self.shift_config = monitoring_config.shift
        self.sponge_config = monitoring_config.sponge
        self.rotation_config = monitoring_config.rotation

    def reset(self):
        pass

    @abstractmethod
    def process_frame(self, segmentation_results: dict) -> dict:
        """
        Main monitoring interface. Must be implemented by subclasses.
        """
        pass

    @property
    def monitoring_options(self) -> dict:
        return {}

    @monitoring_options.setter
    def monitoring_options(self, options: dict) -> None:
        pass

    @property
    def default_monitoring_options(self) -> dict:
        return {}

    def get_default_monitoring_option(self, key: str) -> Any:
        monitoring_options = self.default_monitoring_options
        return get_tree_node(monitoring_options, key)

    def get_monitoring_option(self, key: str) -> Any:
        monitoring_options = self.monitoring_options
        return get_tree_node(monitoring_options, key)

    def set_monitoring_option(self, key: str, value: Any) -> None:
        monitoring_options = self.monitoring_options
        set_tree_node(monitoring_options, key, value)
        self.monitoring_options = monitoring_options

    def update_config_from_params(self, params: dict) -> None:
        """params 딕셔너리의 값으로 config 속성들을 업데이트합니다.

        모든 config 객체들(common_config, direction_config, anomaly_config,
        shift_config, sponge_config, rotation_config)을 순회하며, params에 있는 키와
        일치하는 속성을 찾아 값을 업데이트합니다.

        단, 다음 파라미터들은 monitoring options의 scaling factor 계산에 기준값으로
        사용되므로 직접 업데이트에서 제외됩니다 (이 값들은 제품에서 직접 변경됩니다):
        - shift_sensitivity: shift_config.shifting_box_ratio 계산의 기준값
        - shifting_box_ratio: scaling_factor를 통해 동적으로 계산됨
        - anomaly_sensitivity: anomaly_config.slope_threshold 계산의 기준값
        - slope_threshold: scaling_factor를 통해 동적으로 계산됨
        - sensitivity_scaling_ratio_limit: scaling_factor 계산에 사용되는 한계값
        - sensitivity_scaling_slope_limit: scaling_factor 계산에 사용되는 한계값

        Args:
            params: 업데이트할 파라미터 이름과 값의 딕셔너리
        """
        # Parameters that should not be directly updated via config
        # These are either managed via properties or calculated dynamically
        excluded_params = {
            "shift_sensitivity",
            "shifting_box_ratio",
            "anomaly_sensitivity",
            "slope_threshold",
            "sensitivity_scaling_ratio_limit",
            "sensitivity_scaling_slope_limit",
        }

        config_attrs = [
            "common_config",
            "direction_config",
            "anomaly_config",
            "shift_config",
            "sponge_config",
            "rotation_config",
        ]

        for param_key, param_value in params.items():
            # Skip excluded parameters
            if param_key in excluded_params:
                continue

            # Try to find and update the parameter in any of the config objects
            for config_attr in config_attrs:
                config_obj = getattr(self, config_attr, None)
                if config_obj is not None and hasattr(config_obj, param_key):
                    setattr(config_obj, param_key, param_value)
                    break

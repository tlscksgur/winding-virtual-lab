from typing import TYPE_CHECKING  # 런타임에서는 False이고 타입 체커가 동작할때만 True입니다.
import os

from .version import __VERSION__

if TYPE_CHECKING:
    # Cythonize & Type Checking
    # XXX: Cythonize 하기 위해서 import 구문이 실제로 포함되어 있어야 하고 개발할 때 자동완성을 위해서도 필요합니다.
    from .engine.api import WindingMonitoringEngine

else:
    # Runtime
    # XXX: 런타임에서는 lazy loading으로 동작합니다.
    # 무거운 모듈들이 실제로 사용되기 전까지는 import되지 않습니다.
    from SaigeToolkit.util.lazy_loader import lazy_class, lazy_module

    WindingMonitoringEngine = lazy_class("WindingMonitoringEngine", ".engine.api", globals=globals())

# XXX: 무겁지 않은 모듈(torch 등을 사용하지 않는)은 여기서 import해도 무방합니다.
from SaigeToolkit.error.api import get_error_message


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

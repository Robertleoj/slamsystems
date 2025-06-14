"""

Bindings to the foundation.
---------------------------

"""

from __future__ import annotations

from . import oak_slam, spatial, utils

__all__ = ["oak_slam", "set_spdlog_level", "spatial", "utils"]

def set_spdlog_level(arg0: str) -> None:
    """
    Set spd log level. Supported levels are: trace, debug, info, warn, error, critical, off.
    """

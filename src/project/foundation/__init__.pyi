"""

Bindings to the foundation.
---------------------------

"""

from __future__ import annotations

from . import depth_slam, oak_slam, spatial, symforce_exercises, utils

__all__ = ["depth_slam", "oak_slam", "set_spdlog_level", "spatial", "symforce_exercises", "utils"]

def set_spdlog_level(arg0: str) -> None:
    """
    Set spd log level. Supported levels are: trace, debug, info, warn, error, critical, off.
    """

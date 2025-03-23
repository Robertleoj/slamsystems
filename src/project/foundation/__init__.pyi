"""

Bindings to the foundation.
---------------------------

"""

from __future__ import annotations

from . import dbow, depth_slam, oak_slam, pose_graph, spatial, tag_slam, utils

__all__ = ["dbow", "depth_slam", "oak_slam", "pose_graph", "set_spdlog_level", "spatial", "tag_slam", "utils"]

def set_spdlog_level(arg0: str) -> None:
    """
    Set spd log level. Supported levels are: trace, debug, info, warn, error, critical, off.
    """

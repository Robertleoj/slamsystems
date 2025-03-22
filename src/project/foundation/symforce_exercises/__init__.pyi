"""
Symforce Exercises
"""

from __future__ import annotations

import numpy

from . import depth_slam, tag_slam

__all__ = ["PoseGraphEdge", "PoseGraphVertex", "depth_slam", "pose_graph_ba", "tag_slam"]

class PoseGraphEdge:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, v1_id: int, v2_id: int, v1_to_v2: numpy.ndarray) -> None: ...

class PoseGraphVertex:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, id: int, pose: numpy.ndarray) -> None: ...

def pose_graph_ba(vertices: list[PoseGraphVertex], edges: list[PoseGraphEdge]) -> list[numpy.ndarray]:
    """
    Pose graph BA with symforce
    """

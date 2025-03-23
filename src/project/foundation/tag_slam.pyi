from __future__ import annotations

import numpy

__all__ = ["CameraPose", "Tag", "TagObservation", "tag_slam_ba"]

class CameraPose:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, frame_id: int, camera_pose: numpy.ndarray) -> None: ...
    @property
    def frame_id(self) -> int: ...
    @property
    def pose(self) -> numpy.ndarray: ...

class Tag:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, tag_id: int, tag_pose: numpy.ndarray) -> None: ...
    @property
    def pose(self) -> numpy.ndarray: ...
    @property
    def tag_id(self) -> int: ...

class TagObservation:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, frame_id: int, tag_id: int, observation: numpy.ndarray) -> None: ...

def tag_slam_ba(
    fxfycxcy: numpy.ndarray,
    camera_poses: list[CameraPose],
    observations: list[TagObservation],
    tags: list[Tag],
    tag_side_length: float,
) -> tuple[list[CameraPose], list[Tag]]:
    """
    Tag slam bundle adjustment
    """

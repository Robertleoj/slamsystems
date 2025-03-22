from __future__ import annotations

import numpy

__all__ = ["CameraPose", "Landmark", "LandmarkProjectionObservation", "depth_slam_ba"]

class CameraPose:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, frame_id: int, camera_pose: numpy.ndarray) -> None: ...
    @property
    def camera_pose(self) -> numpy.ndarray: ...
    @property
    def frame_id(self) -> int: ...

class Landmark:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, id: int, loc: numpy.ndarray) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def loc(self) -> numpy.ndarray: ...

class LandmarkProjectionObservation:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self, frame_id: int, landmark_id: int, observation: numpy.ndarray, depth: float | None) -> None: ...

def depth_slam_ba(
    camera_cal: ...,
    camera_poses: list[CameraPose],
    observations: list[LandmarkProjectionObservation],
    landmarks: list[Landmark],
) -> tuple[list[CameraPose], list[Landmark]]:
    """
    Depth slam bundle adjustment
    """

"""
Oak slam broseph
"""

from __future__ import annotations

import foundation.utils
import numpy

__all__ = ["OakSlam"]

class OakSlam:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(
        self,
        color_intrinsics: foundation.utils.CameraParams,
        left_intrinsics: foundation.utils.CameraParams,
        right_intrinsics: foundation.utils.CameraParams,
        center_to_left: numpy.ndarray,
        center_to_right: numpy.ndarray,
    ) -> None: ...
    def process_frame(
        self, center_color: numpy.ndarray, left_mono: numpy.ndarray, right_mono: numpy.ndarray
    ) -> None: ...

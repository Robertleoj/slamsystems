"""
Utilities
"""

from __future__ import annotations

import typing

import numpy

__all__ = ["CameraParams"]

class CameraParams:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    @typing.overload
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None: ...
    @typing.overload
    def __init__(self, K: numpy.ndarray, dist_coeffs: numpy.ndarray, width: int, height: int) -> None: ...
    def __repr__(self) -> str: ...

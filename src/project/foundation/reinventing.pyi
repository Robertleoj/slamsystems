"""
Reinventing the wheel
"""

from __future__ import annotations

import numpy

__all__ = ["fit_poly_ceres", "icp_ceres", "solve_pnp_ceres"]

def fit_poly_ceres(data_x: list[float], data_y: list[float], poly_order: int) -> list[float]:
    """
    Fit a polynomial to data with Ceres
    """

def icp_ceres(
    points1: numpy.ndarray, points2: numpy.ndarray, initial_rvec: numpy.ndarray, initial_tvec: numpy.ndarray
) -> numpy.ndarray: ...
def solve_pnp_ceres(
    image_points: numpy.ndarray,
    object_points: numpy.ndarray,
    K: numpy.ndarray,
    rvec_init: numpy.ndarray,
    tvec_init: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Solve PnP problem using Ceres
    """

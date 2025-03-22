from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    @staticmethod
    def from_arr(arr: np.ndarray) -> Vec2:
        arr = arr.squeeze()
        assert arr.shape == (2,)

        return Vec2(arr[0], arr[1])

    def to_arr(self, dtype: type = np.float32) -> np.ndarray:
        return np.array([self.x, self.y], dtype=dtype)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    @staticmethod
    def from_arr(arr: np.ndarray) -> Vec3:
        arr = arr.squeeze()
        assert arr.shape() == (3,)

        return Vec3(*arr)

    def to_arr(self, dtype: type = np.float32) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)

    @staticmethod
    def zero() -> Vec3:
        return Vec3(0.0, 0.0, 0.0)

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from project.utils.spatial.pose import Pose


@dataclass
class Landmark:
    id: int
    loc: np.ndarray

    def __post_init__(self):
        assert self.loc.shape == (3,)


@dataclass
class Map:
    landmarks: dict[int, Landmark] = field(default_factory=dict)

    def next_landmark_id(self):
        if len(self.landmarks) == 0:
            return 0
        return max(self.landmarks.keys()) + 1

    def add_landmark(self, point3D: np.ndarray) -> int:
        landmark_id = self.next_landmark_id()
        landmark = Landmark(landmark_id, point3D)

        self.landmarks[landmark_id] = landmark

        return landmark_id


@dataclass
class Feature:
    landmark_id: int
    pixel: np.ndarray
    depth: float | None = None


@dataclass
class Frame:
    # mapping from landmark id to feature
    features: dict[int, Feature]
    camera_pose: Pose

    @property
    def feature_pixels(self) -> np.ndarray:
        return np.array([f.pixel for f in self.features.values()])

    @property
    def landmark_ids(self) -> list[int]:
        return [f.landmark_id for f in self.features.values()]


def get_features_3d(map: Map, frame: Frame) -> np.ndarray:
    features_3d = []
    for feature in frame.features.values():
        landmark = map.landmarks[feature.landmark_id]
        features_3d.append(landmark.loc)

    return np.array(features_3d)


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray

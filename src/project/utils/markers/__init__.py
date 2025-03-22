"""Utilities for feduciary markers.

Corner convention is [top left, top right, bottom right, bottom left].
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

import cv2
import dt_apriltags as april
import numpy as np

from project.utils.camera.camera_params import Intrinsics
from project.utils.image import to_greyscale
from project.utils.pnp import estimate_pose_pnp
from project.utils.spatial.pose import Pose


@dataclass(frozen=True)
class DetectedTag:
    tag_id: int
    corners: np.ndarray


@dataclass(frozen=True)
class Tag3D:
    tag_id: int
    pose: Pose
    side_length: float

    @property
    def corners(self):
        return self.pose.apply(get_corners_in_tag(self.side_length))

    def transform(self, pose) -> Tag3D:
        return replace(self, pose=pose @ self.pose)


def get_corners_in_tag(tag_side_length: float) -> np.ndarray:
    return np.array(
        [
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
    ) * (tag_side_length / 2)


def estimate_3d_tag(detected: DetectedTag, intrinsics: Intrinsics, tag_side_length: float) -> Tag3D:
    world_points = get_corners_in_tag(tag_side_length)

    cam_in_tag, _ = estimate_pose_pnp(intrinsics, world_points, detected.corners, method=cv2.SOLVEPNP_IPPE_SQUARE)

    return Tag3D(detected.tag_id, cam_in_tag.inv, tag_side_length)


@cache
def get_tag_detector() -> april.Detector:
    return april.Detector(families="tagStandard52h13", nthreads=16)


def detect_tags(img: np.ndarray, detector: april.Detector) -> dict[int, DetectedTag]:
    grey = to_greyscale(img)

    april_res = detector.detect(grey)
    detected_tags = {}

    for tag_april in april_res:
        tag_id = tag_april.tag_id
        corners = np.array(tag_april.corners)

        detected_tags[tag_id] = DetectedTag(tag_id, corners)

    return detected_tags

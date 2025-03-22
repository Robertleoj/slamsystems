import logging

import cv2
import numpy as np

from project.utils.camera.view import View
from project.utils.features import (
    draw_matches,
    get_good_matches,
    get_matched_points,
    match_feature_descriptors,
    orb_detect_and_compute,
)
from project.visual_odometry.motion_estimation import estimate_motion_from_matches

LOG = logging.getLogger(__name__)


def estimate_camera_movement(
    view1: View,
    view2: View,
    orb: cv2.ORB,
    flann_matcher: cv2.FlannBasedMatcher,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the motion between two views.

    see https://stackoverflow.com/questions/77522308/understanding-cv2-recoverposes-coordinate-frame-transformations
    for description of R and t. If you want to make a pose, here are the equations
    cam_1_R_cam_2 = R.T
    cam_1_t_cam_2 = - (R.T @ (t.reshape(3, 1))).reshape(3)
    cam_1_T_cam_2 = Pose.from_rotmat_trans(cam_1_R_cam_2, cam_1_t_cam_2)

    Returns:
        R: the rotation
        t: the translation
    """

    keypoints1, descriptors1 = orb_detect_and_compute(view1.color, orb=orb)
    keypoints2, descriptors2 = orb_detect_and_compute(view2.color, orb=orb)

    matches = match_feature_descriptors(descriptors1, descriptors2, flann_matcher)

    good_matches = get_good_matches(matches)

    cv2.imshow("matches", draw_matches(view1.color, keypoints1, view2.color, keypoints2, good_matches))
    cv2.waitKey(100)

    pixels1, pixels2 = get_matched_points(keypoints1, keypoints2, good_matches)

    return estimate_motion_from_matches(pixels1, pixels2, view1.intrinsics)

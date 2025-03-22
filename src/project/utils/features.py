from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from project.utils.colors import Color
from project.utils.image import to_greyscale
from project.utils.spatial.vec import Vec2

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Keypoint:
    """Class that mirrors OpenCV's keypoint class"""

    angle: float
    point: Vec2

    # how strong the keypoint is
    response: float

    # diameter of the meaningful keypoint neighborhood
    size: float

    # pyramid layer from which the keypoint was extracted
    octave: int

    # if the keypoints are clustered, this is their class ID
    class_id: int | None

    @staticmethod
    def from_opencv(keypoint: cv2.KeyPoint) -> Keypoint:
        return Keypoint(
            angle=keypoint.angle,
            point=Vec2(keypoint.pt[0], keypoint.pt[1]),
            response=keypoint.response,
            size=keypoint.size,
            octave=keypoint.octave,
            class_id=keypoint.class_id if keypoint.class_id != -1 else None,
        )

    def to_opencv(self) -> cv2.KeyPoint:
        return cv2.KeyPoint(
            x=self.point.x,
            y=self.point.y,
            size=self.size,
            angle=self.angle,
            response=self.response,
            octave=self.octave,
            class_id=self.class_id if self.class_id is not None else -1,
        )


def keypoints_from_opencv(keypoints: list[cv2.KeyPoint]) -> list[Keypoint]:
    return [Keypoint.from_opencv(kp) for kp in keypoints]


def keypoints_to_opencv(keypoints: list[Keypoint]) -> list[cv2.KeyPoint]:
    return [kp.to_opencv() for kp in keypoints]


@dataclass(frozen=True, kw_only=True)
class DescriptorMatch:
    # the distance between the two descriptors
    distance: float

    # index of the query descriptor (first set of descriptors)
    query_idx: int

    # index of the target descriptors (second set)
    target_idx: int

    # index of the image in the target set
    image_idx: int | None

    @staticmethod
    def from_opencv(match: cv2.DMatch) -> DescriptorMatch:
        return DescriptorMatch(
            distance=match.distance,
            image_idx=match.imgIdx if match.imgIdx != -1 else None,
            target_idx=match.trainIdx,
            query_idx=match.queryIdx,
        )

    def to_opencv(self) -> cv2.DMatch:
        return cv2.DMatch(
            _distance=self.distance,
            _trainIdx=self.target_idx,
            _queryIdx=self.query_idx,
            _imgIdx=self.image_idx if self.image_idx is not None else -1,
        )


def descriptor_matches_to_opencv(matches: list[DescriptorMatch]) -> list[cv2.DMatch]:
    return [m.to_opencv() for m in matches]


def descriptor_matches_from_opencv(matches: list[cv2.DMatch]) -> list[DescriptorMatch]:
    return [DescriptorMatch.from_opencv(m) for m in matches]


def orb_detect_and_compute(img: np.ndarray, orb: cv2.ORB) -> tuple[list[Keypoint], np.ndarray]:
    """Detect orb features

    Args:
        img: a color (H x W x 3) or greyscale (H x W) image. If color, it will be
            converted to greyscale
        orb: the orb detector. If None, a new one is created.
    """

    if len(img.shape) == 3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    keypoints_opencv, descriptors = orb.detectAndCompute(grey, None)  # type: ignore

    keypoints = keypoints_from_opencv(keypoints_opencv)

    assert descriptors.shape[0] == len(keypoints)

    return keypoints, descriptors


def orb_detect(img: np.ndarray, *, mask: np.ndarray | None = None, orb: cv2.ORB) -> list[Keypoint]:
    grey = to_greyscale(img)

    if mask:
        assert mask.shape == grey.shape
        mask = mask.astype(np.uint8)

    keypoints_opencv = list(orb.detect(img, mask))

    return keypoints_from_opencv(keypoints_opencv)


def fast_detect(img: np.ndarray, fast: cv2.FastFeatureDetector):
    img = to_greyscale(img)

    kp = list(fast.detect(img))

    return keypoints_from_opencv(kp)


def orb_compute(img: np.ndarray, keypoints: list[Keypoint], orb: cv2.ORB) -> tuple[list[Keypoint], np.ndarray]:
    cv_keypoints = keypoints_to_opencv(keypoints)

    grey = to_greyscale(img)

    kp_cv, descriptors = orb.compute(grey, cv_keypoints)

    new_keypoints = keypoints_from_opencv(list(kp_cv))

    return new_keypoints, descriptors


def draw_keypoints(img: np.ndarray, keypoints: list[Keypoint], color: Color | None = None) -> np.ndarray:
    return cv2.drawKeypoints(
        img,
        keypoints_to_opencv(keypoints),
        None,  # type: ignore
        color=color.rgb_int() if color is not None else None,  # type: ignore
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )


def match_feature_descriptors(
    descriptors_1: np.ndarray, descriptors_2: np.ndarray, matcher: cv2.DescriptorMatcher
) -> list[DescriptorMatch]:
    matches_cv = matcher.match(descriptors_1, descriptors_2)

    matches = descriptor_matches_from_opencv(list(matches_cv))

    return matches


def draw_matches(
    im1: np.ndarray, kp_1: list[Keypoint], im2: np.ndarray, kp_2: list[Keypoint], matches: list[DescriptorMatch]
) -> np.ndarray:
    cv_kp_1 = keypoints_to_opencv(kp_1)
    cv_kp_2 = keypoints_to_opencv(kp_2)

    cv_matches = descriptor_matches_to_opencv(matches)

    visualized = cv2.drawMatches(
        im1,
        cv_kp_1,
        im2,
        cv_kp_2,
        cv_matches,
        None,  # type: ignore
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )  # type: ignore

    return visualized


def get_flann_matcher_for_orb() -> cv2.FlannBasedMatcher:
    # see https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

    return cv2.FlannBasedMatcher(
        dict(
            algorithm=6,  # lsh index
            trees=5,
        ),
        dict(checks=50),
    )


def get_good_matches(matches: list[DescriptorMatch], min_num_matches: int = 8) -> list[DescriptorMatch]:
    matches = sorted(matches, key=lambda m: m.distance)

    dists = np.array([m.distance for m in matches])
    min_dist = dists.min()

    good_matches = []

    for i, match in enumerate(matches):
        if i >= min_num_matches and match.distance > 2 * min_dist:
            break

        good_matches.append(match)

    return good_matches


def get_matched_points(
    keypoints1: list[Keypoint], keypoints2: list[Keypoint], matches: list[DescriptorMatch]
) -> tuple[np.ndarray, np.ndarray]:
    points1_lis = []
    points2_lis = []
    for match in matches:
        point1_idx = match.query_idx
        point1 = keypoints1[point1_idx].point
        points1_lis.append(point1.to_arr())

        point2_idx = match.target_idx
        point2 = keypoints2[point2_idx].point
        points2_lis.append(point2.to_arr())

    points_1 = np.array(points1_lis)
    points_2 = np.array(points2_lis)

    return points_1, points_2


@dataclass(frozen=True)
class FeatureMatchResult:
    matched_points_1: np.ndarray
    matched_points_2: np.ndarray

    keypoints_1: list[Keypoint]
    keypoints_2: list[Keypoint]

    matches: list[DescriptorMatch]


def orb_feature_detect_and_match(
    img1: np.ndarray, img2: np.ndarray, orb: cv2.ORB, matcher: cv2.DescriptorMatcher, min_num_matches: int = 8
) -> FeatureMatchResult:
    """Detect and match features with ORB"""

    kp1, des1 = orb_detect_and_compute(img1, orb)
    kp2, des2 = orb_detect_and_compute(img2, orb)

    matches = match_feature_descriptors(des1, des2, matcher)

    good_matches = get_good_matches(matches, min_num_matches=min_num_matches)

    matched_points_1, matched_points_2 = get_matched_points(kp1, kp2, good_matches)

    return FeatureMatchResult(
        matched_points_1=matched_points_1,
        matched_points_2=matched_points_2,
        keypoints_1=kp1,
        keypoints_2=kp2,
        matches=good_matches,
    )


def get_good_features_to_track(
    grey_img: np.ndarray,
    max_features: int = 100,
    mask: np.ndarray | None = None,
    quality_level: float = 0.3,
    min_distance: float = 7.0,
) -> np.ndarray:
    grey_img = grey_img.squeeze()

    if len(grey_img.shape) == 3:
        grey_img = to_greyscale(grey_img)

    if mask is not None:
        mask = mask.astype(np.uint8)

    return cv2.goodFeaturesToTrack(grey_img, max_features, quality_level, min_distance, mask=mask).reshape(-1, 2)

from dataclasses import dataclass
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np

from project.utils.camera.camera_params import Intrinsics
from project.utils.spatial.pose import Pose

ORIG_COLOR_RESOLUTION = (1920, 1080)
COLOR_RESOLUTION = (640, 400)
MONO_RESOLUTION = (640, 400)
FPS = 5


@dataclass(frozen=True)
class MonoView:
    img: np.ndarray
    intrinsics: Intrinsics
    center_to_cam: Pose


@dataclass(frozen=True)
class ColorView:
    img: np.ndarray
    intrinsics: Intrinsics


@dataclass(frozen=True)
class OakFrame:
    center_color: ColorView
    right_mono: MonoView
    left_mono: MonoView


def _intrinsics_from_oak(cam_matrix: list[list[float]], dist_coeffs: list[float], resolution: tuple[int, int]):
    return Intrinsics(
        width=resolution[0],
        height=resolution[1],
        camera_matrix=np.array(cam_matrix),
        distortion_parameters=np.array(dist_coeffs),
    )


class OakReader:
    def __init__(
        self,
    ):
        self.pipeline = dai.Pipeline()

        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(FPS)

        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_right.setFps(FPS)
        mono_left.setFps(FPS)

        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        sync = self.pipeline.create(dai.node.Sync)

        sync.setSyncThreshold(timedelta(milliseconds=50))
        mono_left.out.link(sync.inputs["mono_left"])
        mono_right.out.link(sync.inputs["mono_right"])
        cam_rgb.video.link(sync.inputs["cam_rgb"])

        xout_grp = self.pipeline.create(dai.node.XLinkOut)
        xout_grp.setStreamName("xout")

        sync.out.link(xout_grp.input)

        self.device = dai.Device(self.pipeline)

        self.queue = self.device.getOutputQueue("xout", 10, False)

        calib_data = self.device.readCalibration()

        # Get intrinsics for each camera
        orig_color_intrinsics = _intrinsics_from_oak(
            calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB, *ORIG_COLOR_RESOLUTION),
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB),
            ORIG_COLOR_RESOLUTION,
        )
        self.color_intrinsics = orig_color_intrinsics.resized(width=COLOR_RESOLUTION[0], height=COLOR_RESOLUTION[1])

        self.left_intrinsics = _intrinsics_from_oak(
            calib_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, *MONO_RESOLUTION),
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.LEFT),
            MONO_RESOLUTION,
        )

        self.right_intrinsics = _intrinsics_from_oak(
            calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, *MONO_RESOLUTION),
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT),
            MONO_RESOLUTION,
        )

        # these are in centimeters, so we convert to millimeters
        self.center_to_left = (
            Pose(np.array(calib_data.getCameraExtrinsics(dai.CameraBoardSocket.RGB, dai.CameraBoardSocket.LEFT)))
            .inv.snap()
            .scale_translation(10.0)
        )
        self.center_to_right = (
            Pose(np.array(calib_data.getCameraExtrinsics(dai.CameraBoardSocket.RGB, dai.CameraBoardSocket.RIGHT)))
            .inv.snap()
            .scale_translation(10.0)
        )

    def get_frame(self):
        out = self.queue.get()  # type: ignore
        print(out)

        color_img = out["cam_rgb"].getCvFrame()  # type: ignore
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, COLOR_RESOLUTION)

        left_img: np.ndarray = out["mono_left"].getCvFrame()  # type: ignore
        right_img: np.ndarray = out["mono_right"].getCvFrame()  # type: ignore

        left_view = MonoView(left_img, self.left_intrinsics, self.center_to_left)

        right_view = MonoView(right_img, self.right_intrinsics, self.center_to_right)

        color_view = ColorView(color_img, self.color_intrinsics)

        assert color_img.shape[:2][::-1] == COLOR_RESOLUTION, f"expected {COLOR_RESOLUTION}, got {color_img.shape}"
        assert left_img.shape[:2][::-1] == MONO_RESOLUTION
        assert right_img.shape[:2][::-1] == MONO_RESOLUTION

        return OakFrame(color_view, right_view, left_view)

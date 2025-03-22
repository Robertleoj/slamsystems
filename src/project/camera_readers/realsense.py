import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs

from project.utils.camera.camera_params import Intrinsics
from project.utils.camera.view import View


class RealSenseCamera:
    dead: bool

    def __init__(self, width=640, height=480, fps=30):
        """Initialize the RealSense camera pipeline."""
        self.dead = False
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure the stream
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.align = rs.align(rs.stream.color)
        config_in_use = self.pipeline.start(self.config)
        depth_sensor = config_in_use.get_device().first_depth_sensor()

        self.intrinsics = self.get_intrinsics()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_view(self) -> View:
        """Fetch aligned depth & color frames."""
        if self.dead:
            raise RuntimeError("Camera not started! Call start() first.")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("No frames!")

        # Convert frames to numpy
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        # always convert to millimeters
        depth_image = depth_image * self.depth_scale * 1000.0

        color_image = np.asanyarray(color_frame.get_data()).copy()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return View(depth=depth_image, color=color_image, intrinsics=self.intrinsics)

    def get_intrinsics(self) -> Intrinsics:
        """Retrieve factory calibration intrinsics for depth & color cameras."""
        profile = self.pipeline.get_active_profile()

        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        return Intrinsics.from_pinhole_params(
            width=color_intrinsics.width,
            height=color_intrinsics.height,
            fx=color_intrinsics.fx,
            fy=color_intrinsics.fy,
            cx=color_intrinsics.ppx,
            cy=color_intrinsics.ppy,
            distortion_params=np.array(color_intrinsics.coeffs),
        )

    def stop(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()
        self.dead = True

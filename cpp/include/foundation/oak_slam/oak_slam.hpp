#pragma once
#include <Remotery.h>

#include <foundation/oak_slam/map.hpp>
#include <foundation/oak_slam/tracker.hpp>
#include <foundation/oak_slam/type_defs.hpp>
#include <foundation/utils/camera.hpp>
#include <foundation/utils/numpy.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

namespace foundation {
namespace oak_slam {

class OakSlam {
 public:
  OakSlam(CameraParams& color_intrinsics,
          CameraParams& left_intrinsics,
          CameraParams& right_intrinsics,
          py::EigenDRef<Eigen::Matrix4d> center_to_left,
          py::EigenDRef<Eigen::Matrix4d> center_to_right)
      : cam_calibration(OakCameraCalibration{color_intrinsics, left_intrinsics,
                                             right_intrinsics, center_to_left,
                                             center_to_right}),
        tracker(cam_calibration, &map) {

    rmtSettings* settings = rmt_Settings();
    settings->port = 17815;  // Change to your desired port

    rmt_CreateGlobalInstance(&rmt);
  }

  void process_frame(OakFrame& frame) {
    tracker.process_frame(frame);
  }

  void process_frame(img_array center_color,
                     img_array left_mono,
                     img_array right_mono) {
    OakFrame frame{img_numpy_to_mat_uint8(center_color).clone(),
                   img_numpy_to_mat_uint8(left_mono).clone(),
                   img_numpy_to_mat_uint8(right_mono).clone()};

    process_frame(frame);
  }

 private:
  Map map;
  OakCameraCalibration cam_calibration;

  Tracker tracker;

  Remotery* rmt;
};

}  // namespace oak_slam
}  // namespace foundation
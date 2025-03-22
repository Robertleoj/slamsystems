#pragma once

#include <foundation/oak_slam/orb.hpp>
#include <foundation/oak_slam/type_defs.hpp>
#include <foundation/utils/rerun.hpp>
#include <rerun.hpp>

namespace foundation {
namespace oak_slam {

class RerunLogger {
 public:
  RerunLogger(std::string stream_name);

  void log_calibration(OakCameraCalibration& cam_calibration);
  void log_frame(OakFrame& frame);

  void log_triple_match(OakFrame* frame, TripleMatch* triple_match);

  void log_maybe_triple_triangulated(
      std::vector<std::optional<Eigen::Vector3d>>* maybe_landmarks,
      OakCameraCalibration* cam_calibration);

 private:
  rerun::RecordingStream stream;
  std::mutex stream_lock;
};
}  // namespace oak_slam
}  // namespace foundation
#include <Remotery.h>
#include <spdlog/spdlog.h>
#include <ranges>

#include <format>
#include <foundation/oak_slam/orb.hpp>
#include <foundation/oak_slam/triangulation.hpp>
#include <foundation/oak_slam/type_defs.hpp>
#include <foundation/utils/thread_queue.hpp>
#include <opencv2/opencv.hpp>
#include <rerun.hpp>
#include <thread>

namespace foundation {
namespace oak_slam {

enum class TrackerStage { INITIALIZE, TRACKING, TRACKING_LOST };

class Tracker {
 public:
  Tracker(OakCameraCalibration& cam_calibration,
          RerunLogger* rerun_logger,
          Map* map)
      : cam_calibration(cam_calibration),
        stage(TrackerStage::INITIALIZE),
        rerun_logger(rerun_logger),
        frame_queue(2),
        map(map) {
    worker = std::thread(&Tracker::work, this);
  }

  void process_frame(OakFrame& frame) { frame_queue.push(frame); }

 private:
  OakCameraCalibration cam_calibration;
  TrackerStage stage;

  ThreadQueue<OakFrame> frame_queue;
  std::thread worker;
  OrbDetector orb_detector;
  OrbMatcher orb_matcher;
  RerunLogger* rerun_logger;
  Map* map;

  bool should_stop = false;

  void work() {
    spdlog::info("Tracker started");

    while (!should_stop) {
      auto frame = frame_queue.wait_and_pop();
      // spdlog::info(fmt::format("queue size: {}", frame_queue.size()));
      spdlog::info(std::format("queue size: {}", frame_queue.size()));

      switch (stage) {
        case TrackerStage::INITIALIZE: {
          handle_initialize(frame);
          break;
        }
        case TrackerStage::TRACKING: {
          handle_tracking(frame);
          break;
        }

        case TrackerStage::TRACKING_LOST: {
          handle_tracking_lost(frame);
          break;
        }
      }
    }
  }

  void handle_initialize(OakFrame frame) {
    // here we initialize the map
    // we'll detect ORB features in all three frames
    // and triangulate them into 3D, and put them into the map.

    // This frame will be added as a keyframe.

    // 1. Detect orb in all cameras
    rmt_ScopedCPUSample(Initialize, 0);
    rmt_LogText("Initializing");

    spdlog::debug("Detecting orb");

    rmt_BeginCPUSample(detect_all, 0);
    auto center_orb = orb_detector.detect(frame.center_color);
    auto left_orb = orb_detector.detect(frame.left_mono);
    auto right_orb = orb_detector.detect(frame.right_mono);
    rmt_EndCPUSample();

    spdlog::debug("making triple match");
    // 2. Find orb matches between all the cameras
    auto triple_match =
        orb_matcher.make_triple_match(center_orb, left_orb, right_orb);

    spdlog::debug("logging triple match");
    rerun_logger->log_triple_match(&frame, &triple_match);

    if (triple_match.num_matches() < 5) {
      spdlog::debug("not enough matches to triangulate");
      return;
    }

    spdlog::debug("triangulating triple matches");
    // 3. Triangulate those points in 3D
    auto triangulated =
        triangulate_triple_match(&triple_match, &cam_calibration, 5.0);

    spdlog::debug("logging triangulated triple matches");
    rerun_logger->log_maybe_triple_triangulated(&triangulated,
                                                &cam_calibration);

    spdlog::debug("Adding to map");
    Keyframe keyframe{};

    // Add all the points to the map
    for (int i = 0; i < triple_match.num_matches(); i++) {
      auto triangulated_location_opt = triangulated[i];

      if (!triangulated_location_opt.has_value()) {
        continue;
      }
      auto triangulated_location = triangulated_location_opt.value();

      Landmark landmark{};
      landmark.location = triangulated_location;

      for (auto cam_loc : OakCamLoc::all()) {
        Feature feature(landmark.id);

        landmark.add_feature(feature.id, keyframe.id, cam_loc);

        keyframe.add_feature(cam_loc, std::move(feature));
      }
    }

    map->add_keyframe(std::move(keyframe));

    // Add keyframe with all the features

    // 5. Signal backend to optimize
  }

  void handle_tracking(OakFrame& frame) {}

  void handle_tracking_lost(OakFrame& frame) {}
};

}  // namespace oak_slam

}  // namespace foundation
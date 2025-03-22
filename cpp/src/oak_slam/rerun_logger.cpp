#include <Remotery.h>
#include <spdlog/spdlog.h>
#include <format>

#include <foundation/oak_slam/rerun_logger.hpp>

namespace foundation {
namespace oak_slam {

struct ScaledImage {
  cv::Mat img;
  double scaleX, scaleY;
};

// Resize images to the tallest one while keeping aspect ratio
std::vector<ScaledImage> resize_to_max_height(std::vector<cv::Mat>& images) {
  int maxHeight = 0;
  std::vector<ScaledImage> scaledImages;

  // Find max height
  for (const auto& img : images) {
    maxHeight = std::max(maxHeight, img.rows);
  }

  // Resize images and store scale factors
  for (auto& img : images) {
    double scale = static_cast<double>(maxHeight) / img.rows;
    int newWidth = static_cast<int>(img.cols * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newWidth, maxHeight));

    scaledImages.push_back({resized, scale, scale});
  }

  return scaledImages;
}

// Draws lines using dynamically scaled points
void draw_lines(cv::Mat& stacked,
                const std::vector<int>& widths,
                const std::vector<ScaledImage>& scaledImages,
                const std::vector<cv::Point>& va,
                const std::vector<cv::Point>& vb,
                const std::vector<cv::Point>& vc) {
  for (size_t i = 0; i < va.size(); i++) {
    // Scale points on-the-fly based on their image's scale factor
    cv::Point pA = cv::Point(va[i].x * scaledImages[0].scaleX,
                             va[i].y * scaledImages[0].scaleY);

    cv::Point pB = cv::Point(vb[i].x * scaledImages[1].scaleX,
                             vb[i].y * scaledImages[1].scaleY) +
                   cv::Point(widths[0], 0);

    cv::Point pC = cv::Point(vc[i].x * scaledImages[2].scaleX,
                             vc[i].y * scaledImages[2].scaleY) +
                   cv::Point(widths[0] + widths[1], 0);

    cv::line(stacked, pA, pB, cv::Scalar(0, 255, 0), 2);  // Green line
    cv::line(stacked, pB, pC, cv::Scalar(0, 255, 0), 2);
  }
}

cv::Mat draw_triple_match(OakFrame* frame, TripleMatch* match) {
  // convert all to color
  cv::Mat left_color;
  cv::cvtColor(frame->left_mono, left_color, cv::COLOR_GRAY2RGB);

  cv::Mat right_color;
  cv::cvtColor(frame->right_mono, right_color, cv::COLOR_GRAY2RGB);

  std::vector<cv::Mat> images = {left_color, frame->center_color, right_color};

  auto resizedImages = resize_to_max_height(images);

  // Extract resized images and their scale factors
  std::vector<cv::Mat> resizedMats;
  std::vector<int> widths;
  for (const auto& img : resizedImages) {
    resizedMats.push_back(img.img);
    widths.push_back(img.img.cols);
  }

  std::vector<cv::Point> va;
  std::vector<cv::Point> vb;
  std::vector<cv::Point> vc;

  for (int i = 0; i < match->num_matches(); i++) {
    va.push_back(match->match[OakCamLoc::LEFT].keypoints[i].pt);
    vb.push_back(match->match[OakCamLoc::CENTER].keypoints[i].pt);
    vc.push_back(match->match[OakCamLoc::RIGHT].keypoints[i].pt);
  }

  // Stack images horizontally
  cv::Mat stacked;
  cv::hconcat(resizedMats, stacked);

  // Draw connecting lines (scaling applied only while drawing)
  draw_lines(stacked, widths, resizedImages, va, vb, vc);

  return stacked;
}

RerunLogger::RerunLogger(std::string stream_name) : stream(stream_name) {
  stream.spawn().exit_on_failure();
}

void RerunLogger::log_calibration(OakCameraCalibration& cam_calibration) {
  std::scoped_lock l(stream_lock);

  auto color_cam = rerun_utils::camera(cam_calibration.color_intrinsics)
                       .with_image_plane_distance(50.0);
  auto left_cam = rerun_utils::camera(cam_calibration.left_intrinsics)
                      .with_image_plane_distance(25.0);
  auto right_cam = rerun_utils::camera(cam_calibration.right_intrinsics)
                       .with_image_plane_distance(25.0);

  stream.log_static("/cameras/color/cam", color_cam);

  stream.log_static("/cameras/left/",
                    rerun_utils::transform(cam_calibration.center_to_left));
  stream.log_static("/cameras/left/cam", left_cam);

  stream.log_static("/cameras/right/",
                    rerun_utils::transform(cam_calibration.center_to_right));
  stream.log_static("/cameras/right/cam", right_cam);
}

void RerunLogger::log_frame(OakFrame& frame) {
  std::scoped_lock l(stream_lock);

  stream.log_static("/cameras/color/cam/image",
                    rerun_utils::rgb_image(frame.center_color));

  stream.log_static("/cameras/left/cam/image",
                    rerun_utils::grey_image(frame.left_mono));

  stream.log_static("/cameras/right/cam/image",
                    rerun_utils::grey_image(frame.right_mono));
}

void RerunLogger::log_triple_match(OakFrame* frame, TripleMatch* triple_match) {
  rmt_ScopedCPUSample(log_triple_match, 0);
  rmt_LogText("Logging Triple Match");

  std::scoped_lock l(stream_lock);

  auto drawn = draw_triple_match(frame, triple_match);
  stream.log_static("/tripe_matches/vis", rerun_utils::rgb_image(drawn));

  // std::vector<rerun::components::Position2D> color_pts;
  // for (auto& pt : triple_match.center_keypoints) {
  // color_pts.push_back(rerun::components::Position2D(pt.pt.x, pt.pt.y));
  // }
  // stream.log_static("/cameras/color/cam/image/points",
  // rerun::Points2D().with_positions(color_pts));
}

void RerunLogger::log_maybe_triple_triangulated(
    std::vector<std::optional<Eigen::Vector3d>>* maybe_landmarks,
    OakCameraCalibration* cam_calibration) {
  std::vector<rerun::components::Position3D> landmarks;

  std::vector<rerun::components::Vector3D> vectors;
  std::vector<rerun::components::Position3D> origins;

  for (auto& maybe_landmark : *maybe_landmarks) {
    if (!maybe_landmark.has_value()) {
      continue;
    }
    auto& landmark = maybe_landmark.value();
    auto position =
        rerun::components::Position3D(landmark.x(), landmark.y(), landmark.z());

    landmarks.push_back(position);

    auto vec =
        rerun::components::Vector3D(landmark.x(), landmark.y(), landmark.z());

    auto origin1 =
        rerun::components::Position3D(cam_calibration->center_to_left(0, 3),
                                      cam_calibration->center_to_left(1, 3),
                                      cam_calibration->center_to_left(2, 3));
    vectors.push_back(rerun::components::Vector3D(
        vec.x() - origin1.x(), vec.y() - origin1.y(), vec.z() - origin1.z()));

    origins.push_back(origin1);

    auto origin2 = rerun::components::Position3D(0, 0, 0);
    vectors.push_back(vec);
    origins.push_back(origin2);

    auto origin3 =
        rerun::components::Position3D(cam_calibration->center_to_right(0, 3),
                                      cam_calibration->center_to_right(1, 3),
                                      cam_calibration->center_to_right(2, 3));
    origins.push_back(origin3);
    vectors.push_back(rerun::components::Vector3D(
        vec.x() - origin3.x(), vec.y() - origin3.y(), vec.z() - origin3.z()));
  }

  std::scoped_lock l(stream_lock);

  rerun::Arrows3D();

  stream.log_static(
      "/triangulated_triple_match/points",
      rerun::Points3D().with_positions(landmarks).with_radii(20.0));
  stream.log_static(
      "/triangulated_triple_match/arrows",
      rerun::Arrows3D().with_vectors(vectors).with_origins(origins));
}

}  // namespace oak_slam

}  // namespace foundation
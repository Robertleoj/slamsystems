#pragma once
#include <Eigen/Core>
#include <foundation/utils/camera.hpp>
#include <opencv2/opencv.hpp>
#include <rerun.hpp>

namespace foundation {
namespace rerun_utils {

template <typename T, size_t N>
std::array<T, N> ptr_to_array(const float* ptr) {
  std::array<T, N> arr;
  std::copy(ptr, ptr + N, arr.begin());
  return arr;
}

rerun::Pinhole camera(CameraParams& cam);
rerun::Transform3D transform(Eigen::Matrix4d homo);
rerun::Image grey_image(cv::Mat& grey_img);
rerun::Image rgb_image(cv::Mat& rgb_img);

}  // namespace rerun_utils
}  // namespace foundation

// Adapters so we can borrow an OpenCV image easily into Rerun images without
// copying:
template <>
struct rerun::CollectionAdapter<uint8_t, cv::Mat> {
  /// Borrow for non-temporary.
  Collection<uint8_t> operator()(const cv::Mat& img) {
    assert("OpenCV matrix was expected have bit depth CV_U8" &&
           CV_MAT_DEPTH(img.type()) == CV_8U);

    return Collection<uint8_t>::borrow(img.data, img.total() * img.channels());
  }

  // Do a full copy for temporaries (otherwise the data might be deleted when
  // the temporary is destroyed).
  Collection<uint8_t> operator()(cv::Mat&& img) {
    assert("OpenCV matrix was expected have bit depth CV_U8" &&
           CV_MAT_DEPTH(img.type()) == CV_8U);

    std::vector<uint8_t> img_vec(img.total() * img.channels());
    img_vec.assign(img.data, img.data + img.total() * img.channels());
    return Collection<uint8_t>::take_ownership(std::move(img_vec));
  }
};
#include <foundation/utils/rerun.hpp>
#include <sophus/se3.hpp>

namespace foundation {
namespace rerun_utils {

rerun::Pinhole camera(CameraParams& cam) {
  Eigen::Matrix3f K = cam.K.cast<float>();

  auto pinhole_proj =
      rerun::components::PinholeProjection::from_mat3x3(K.data());

  return rerun::Pinhole()
      .with_image_from_camera(pinhole_proj)
      .with_resolution(static_cast<float>(cam.width),
                       static_cast<float>(cam.height));
}

rerun::Transform3D transform(Eigen::Matrix4d homo) {
  Sophus::SE3f trans = Sophus::SE3f(homo.cast<float>());
  auto translation =
      rerun::components::Translation3D(trans.translation().data());

  auto rotation_array = ptr_to_array<float, 9>(trans.rotationMatrix().data());

  auto rotation = rerun::components::TransformMat3x3(rotation_array);

  return rerun::Transform3D::from_translation_mat3x3(translation, rotation);
}

rerun::Image grey_image(cv::Mat& grey_img) {
  return rerun::Image::from_greyscale8(
      grey_img, {static_cast<unsigned int>(grey_img.cols),
                 static_cast<unsigned int>(grey_img.rows)});
}

rerun::Image rgb_image(cv::Mat& rgb_img) {
  return rerun::Image::from_rgb24(rgb_img,
                                  {static_cast<unsigned int>(rgb_img.cols),
                                   static_cast<unsigned int>(rgb_img.rows)});
}

}  // namespace rerun_utils
}  // namespace foundation

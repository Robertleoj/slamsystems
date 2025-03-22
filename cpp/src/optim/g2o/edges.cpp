#include <foundation/optim/g2o/edges.hpp>
#include <sophus/so3.hpp>

namespace foundation {

/**
 * Reprojection error where the 3D point is fixed
 */
Fixed3DReprojectionErrorEdge::Fixed3DReprojectionErrorEdge(
    Eigen::Vector3d world_point, CameraParams cam)
    : BaseUnaryEdge(), _world_point(world_point), _cam(cam) {}

void Fixed3DReprojectionErrorEdge::computeError() {
  auto point_in_cam = get_point_in_camera();
  Eigen::Vector3d point_in_image = _cam.K * point_in_cam;

  point_in_image /= point_in_image[2];

  _error = _measurement - point_in_image.head<2>();
}

void Fixed3DReprojectionErrorEdge::linearizeOplus() {
  auto point_in_cam = get_point_in_camera();

  auto X = point_in_cam[0];
  auto Y = point_in_cam[1];
  auto Z = point_in_cam[2];

  auto fx = _cam.fx();
  auto fy = _cam.fy();
  auto cx = _cam.cx();
  auto cy = _cam.cy();

  _jacobianOplusXi << fx / Z, 0, -(fx * X) / (Z * Z), -(fx * X * Y) / (Z * Z),
      fx + (fx * X * X) / (Z * Z), -(fx * Y) / Z, 0, fy / Z,
      -(fy * Y) / (Z * Z), -fy - (fy * Y * Y) / (Z * Z), (fy * X * Y) / (Z * Z),
      (fy * X) / Z;

  _jacobianOplusXi = -_jacobianOplusXi;
}

bool Fixed3DReprojectionErrorEdge::read(std::istream &in) { return false; }

bool Fixed3DReprojectionErrorEdge::write(std::ostream &out) const {
  return false;
}

Eigen::Vector3d Fixed3DReprojectionErrorEdge::get_point_in_camera() const {
  const CameraPose *v = static_cast<const CameraPose *>(_vertices[0]);

  Sophus::SE3d cam_pose_estimate = v->estimate();

  Eigen::Vector3d point_in_cam = cam_pose_estimate * _world_point;
  return point_in_cam;
}

/**
 * Transforming a 3D point to match another
 */

Two3DPointsAlignmentErrorEdge::Two3DPointsAlignmentErrorEdge(
    Eigen::Vector3d point2)
    : _point2(point2) {}

Eigen::Vector3d Two3DPointsAlignmentErrorEdge::get_transformed_point_2() {
  return get_cam_pose() * _point2;
}

Sophus::SE3d Two3DPointsAlignmentErrorEdge::get_cam_pose() {
  const CameraPose *v = static_cast<const CameraPose *>(_vertices[0]);

  Sophus::SE3d cam_pose_estimate = v->estimate();
  return cam_pose_estimate;
}

void Two3DPointsAlignmentErrorEdge::computeError() {
  _error = _measurement - get_transformed_point_2();
}

void Two3DPointsAlignmentErrorEdge::linearizeOplus() {
  _jacobianOplusXi.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

  _jacobianOplusXi.block(0, 3, 3, 3) =
      -Sophus::SO3d::hat(get_transformed_point_2());

  _jacobianOplusXi = -_jacobianOplusXi;
}
bool Two3DPointsAlignmentErrorEdge::read(std::istream &in) { return false; }

bool Two3DPointsAlignmentErrorEdge::write(std::ostream &out) const {
  return false;
}

}  // namespace foundation
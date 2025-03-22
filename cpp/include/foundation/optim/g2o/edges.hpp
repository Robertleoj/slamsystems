#include <g2o/core/base_unary_edge.h>

#include <foundation/optim/g2o/vertices.hpp>
#include <foundation/utils/camera.hpp>

namespace foundation {
class Fixed3DReprojectionErrorEdge
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, CameraPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Fixed3DReprojectionErrorEdge(Eigen::Vector3d world_point, CameraParams cam);

  void computeError() override;
  void linearizeOplus() override;
  bool read(std::istream &in) override;
  bool write(std::ostream &out) const override;

 private:
  Eigen::Vector3d _world_point;
  CameraParams _cam;

  Eigen::Vector3d get_point_in_camera() const;
};

class Two3DPointsAlignmentErrorEdge
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, CameraPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Two3DPointsAlignmentErrorEdge(Eigen::Vector3d point2);

  void computeError() override;
  void linearizeOplus() override;
  bool read(std::istream &in) override;
  bool write(std::ostream &out) const override;

 private:
  Eigen::Vector3d _point2;

  Eigen::Vector3d get_transformed_point_2();

  Sophus::SE3d get_cam_pose();
};
}  // namespace foundation
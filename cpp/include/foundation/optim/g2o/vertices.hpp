#pragma once
#include <g2o/core/base_vertex.h>

#include <sophus/se3.hpp>

namespace foundation {
class CameraPose : public g2o::BaseVertex<6, Sophus::SE3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setToOriginImpl() override;
  void oplusImpl(const double *update) override;
  bool read(std::istream &in) override;
  bool write(std::ostream &out) const override;
};

}  // namespace foundation
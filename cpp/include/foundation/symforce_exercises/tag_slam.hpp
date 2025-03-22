#pragma once

#include <foundation/symforce_generated/tag_slam/cpp/symforce/tag_slam_factors/tag_reprojection_error_factor.h>
#include <sym/linear_camera_cal.h>
#include <sym/pose3.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include <Eigen/Core>
#include <foundation/optim/symforce/conversion.hpp>
#include <foundation/optim/symforce/default_opt_params.hpp>
#include <foundation/types.hpp>
#include <vector>

namespace py = pybind11;

namespace foundation {
namespace tag_slam {

enum Var : char {
  CAMERA_CALIBRATION = 'c',
  CAM_POSE = 'p',
  TAG_POSE = 't',
  OBSERVATION = 'o',
  EPSILON = 'e',
  TAG_SIZE = 'z'
};

struct CameraPose {
  int frame_id;
  Eigen::Matrix4d camera_pose;

  CameraPose(int frame_id, py::EigenDRef<Eigen::Matrix4d> camera_pose)
      : frame_id(frame_id), camera_pose(camera_pose) {}
};

struct Tag {
  int tag_id;
  Eigen::Matrix4d tag_pose;

  Tag(int tag_id, py::EigenDRef<Eigen::Matrix4d> tag_pose)
      : tag_id(tag_id), tag_pose(tag_pose) {}
};

struct TagObservation {
  int frame_id;
  int tag_id;
  Eigen::Matrix<double, 8, 1> observation;

  TagObservation(int frame_id, int tag_id,
                 py::EigenDRef<Eigen::Matrix<double, 8, 1>> observation)
      : frame_id(frame_id), tag_id(tag_id), observation(observation) {}
};

std::tuple<std::vector<CameraPose>, std::vector<Tag>> tag_slam_ba(
    py::EigenDRef<Eigen::Vector4d> fxfycxcy,
    std::vector<CameraPose> &camera_poses,
    std::vector<TagObservation> &observations, std::vector<Tag> &tags,
    double tag_side_length) {
  std::vector<sym::Factord> factors;

  for (auto &o : observations) {
    factors.push_back(sym::Factord::Hessian(
        tag_slam_factors::TagReprojectionErrorFactor<double>,
        {{Var::CAMERA_CALIBRATION},
         {Var::CAM_POSE, o.frame_id},
         {Var::TAG_POSE, o.tag_id},
         {Var::OBSERVATION, o.frame_id, o.tag_id},
         {Var::TAG_SIZE},
         {Var::EPSILON}},
        {{Var::CAM_POSE, o.frame_id}, {Var::TAG_POSE, o.tag_id}}));
  }

  sym::Valuesd values;

  const double epsilon = 1e-10;
  values.Set({Var::EPSILON}, epsilon);

  sym::LinearCameraCald cal(fxfycxcy);
  values.Set({Var::CAMERA_CALIBRATION}, cal);

  values.Set({Var::TAG_SIZE}, tag_side_length);

  for (auto &o : observations) {
    values.Set({Var::OBSERVATION, o.frame_id, o.tag_id},
               sym::Matrix81d(o.observation));
  }

  for (auto &c : camera_poses) {
    values.Set({Var::CAM_POSE, c.frame_id}, pose_from_homo_mat(c.camera_pose));
  }

  for (auto &t : tags) {
    values.Set({Var::TAG_POSE, t.tag_id}, pose_from_homo_mat(t.tag_pose));
  }

  const auto optimizer_params = default_optimizer_params();

  sym::Optimizerd optimizer(optimizer_params, factors, "TagSlamOptimizer", {},
                            epsilon);

  auto stats = optimizer.Optimize(values);

  std::vector<CameraPose> out_cams;
  std::vector<Tag> out_tags;

  for (auto &c : camera_poses) {
    Eigen::Matrix4d pose = values.At<sym::Pose3d>({Var::CAM_POSE, c.frame_id})
                               .ToHomogenousMatrix();
    CameraPose p = {c.frame_id, pose};
    out_cams.push_back(std::move(p));
  }

  for (auto &t : tags) {
    Eigen::Matrix4d tag_pose =
        values.At<sym::Pose3d>({Var::TAG_POSE, t.tag_id}).ToHomogenousMatrix();
    Tag tag = {t.tag_id, tag_pose};
    out_tags.push_back(std::move(tag));
  }

  return std::make_tuple(out_cams, out_tags);
}

}  // namespace tag_slam
}  // namespace foundation
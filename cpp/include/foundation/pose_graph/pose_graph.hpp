#pragma once

#include <foundation/symforce_generated/circle_pose_graph/cpp/symforce/circle_pose_graph/pose_diff_factor.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
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

enum Var : char { POSE = 'p', POSE_DIFF = 'd', EPSILON = 'e' };

struct PoseGraphVertex {
  int id;
  Eigen::Matrix4d pose;

  PoseGraphVertex(int id, py::EigenDRef<Eigen::Matrix4d> pose)
      : id(id), pose(pose) {}
};

struct PoseGraphEdge {
  int v1_id;
  int v2_id;
  Eigen::Matrix4d v1_to_v2;

  PoseGraphEdge(int v1_id, int v2_id, py::EigenDRef<Eigen::Matrix4d> v1_to_v2)
      : v1_id(v1_id), v2_id(v2_id), v1_to_v2(v1_to_v2) {}
};

std::vector<Eigen::Matrix4d> pose_graph_ba(
    std::vector<PoseGraphVertex> vertices,
    std::vector<PoseGraphEdge> edges) {
  std::vector<sym::Factor<double>> factors;

  int num_vertices = vertices.size();
  int num_edges = edges.size();

  for (auto& e : edges) {
    factors.push_back(
        sym::Factord::Hessian(circle_pose_graph::PoseDiffFactor<double>,
                              {{Var::POSE, e.v1_id},
                               {Var::POSE, e.v2_id},
                               {Var::POSE_DIFF, e.v1_id, e.v2_id},
                               {Var::EPSILON}},
                              {{Var::POSE, e.v1_id}, {Var::POSE, e.v2_id}}));
  }

  sym::Valuesd values;

  const double epsilon = 1e-10;
  values.Set({Var::EPSILON}, epsilon);
  for (auto& v : vertices) {
    values.Set({Var::POSE, v.id}, pose_from_homo_mat(v.pose));
  }
  for (auto& e : edges) {
    values.Set({Var::POSE_DIFF, e.v1_id, e.v2_id},
               pose_from_homo_mat(e.v1_to_v2));
  }

  const auto optimizer_params = default_optimizer_params();

  sym::Optimizerd optimizer(optimizer_params, factors, "PoseGraphOptimizer", {},
                            epsilon);

  auto stats = optimizer.Optimize(values);

  std::vector<Eigen::Matrix4d> out;

  for (int i = 0; i < num_vertices; i++) {
    out.push_back(values.At<sym::Pose3d>({Var::POSE, i}).ToHomogenousMatrix());
  }

  return out;
}

}  // namespace foundation
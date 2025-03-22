
#include <DBow3/DBoW3.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <format>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <symforce/opt/optimizer.h>

#include <foundation/oak_slam/python.hpp>
#include <foundation/reinventing/icp_ceres.hpp>
#include <foundation/reinventing/icp_g2o.hpp>
#include <foundation/reinventing/poly_fit_ceres.hpp>
#include <foundation/reinventing/poly_fit_g2o.hpp>
#include <foundation/reinventing/solve_pnp_ceres.hpp>
#include <foundation/reinventing/solve_pnp_g2o.hpp>
#include <foundation/spatial/lie_algebra.hpp>
#include <foundation/symforce_exercises/depth_slam.hpp>
#include <foundation/symforce_exercises/pose_graph.hpp>
#include <foundation/symforce_exercises/tag_slam.hpp>
#include <foundation/utils/camera.hpp>
#include <foundation/utils/fbow.hpp>

// NOTE: This sets compile time level. In addition, you need to set the
// runtime level low enough to show these (e.g. trace for everything)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include <spdlog/spdlog.h>

#include <fstream>

namespace py = pybind11;

using namespace foundation;

template <typename T>
std::string info(T &obj) {
  std::stringstream s;
  s << obj;
  return s.str();
}

void init_reinventing(py::module_ &m) {
  m.def("solve_pnp_ceres", &solve_pnp_ceres_pywrapper,
        "Solve PnP problem using Ceres", py::arg("image_points"),
        py::arg("object_points"), py::arg("K"), py::arg("rvec_init"),
        py::arg("tvec_init"));

  m.def("solve_pnp_g2o", &solve_pnp_g2o_pywrapper, "Solve PnP with g2o",
        py::arg("image_points"), py::arg("object_points"), py::arg("K"),
        py::arg("rvec_init"), py::arg("tvec_init"));

  m.def("fit_poly_ceres", &fit_poly_ceres,
        "Fit a polynomial to data with Ceres", py::arg("data_x"),
        py::arg("data_y"), py::arg("poly_order"));

  m.def("fit_poly_g2o", &fit_poly_g2o, "Fit a polynomial to data with g2o",
        py::arg("data_x"), py::arg("data_y"), py::arg("poly_order"));

  m.def("icp_g2o", &icp_g2o_pywrapper, py::arg("points1"), py::arg("points2"),
        py::arg("initial_rvec"), py::arg("initial_tvec"));

  m.def("icp_ceres", &icp_ceres_pywrapper, py::arg("points1"),
        py::arg("points2"), py::arg("initial_rvec"), py::arg("initial_tvec"));
}

void init_spatial(py::module_ &m) {
  m.def("SE3_log", &SE3_log, "Logarithm mapping from SE3 to se3",
        py::arg("mat"));

  m.def("se3_exp", &se3_exp, "Exponential mapping from se3 to SE3",
        py::arg("se3"));
}

void init_tag_slam(py::module_ &m) {
  py::class_<tag_slam::CameraPose>(m, "CameraPose")
      .def(py::init<int, py::EigenDRef<Eigen::Matrix4d>>(), py::arg("frame_id"),
           py::arg("camera_pose"))
      .def_readonly("frame_id", &tag_slam::CameraPose::frame_id)
      .def_readonly("pose", &tag_slam::CameraPose::camera_pose);

  py::class_<tag_slam::Tag>(m, "Tag")
      .def(py::init<int, py::EigenDRef<Eigen::Matrix4d>>(), py::arg("tag_id"),
           py::arg("tag_pose"))
      .def_readonly("tag_id", &tag_slam::Tag::tag_id)
      .def_readonly("pose", &tag_slam::Tag::tag_pose);

  py::class_<tag_slam::TagObservation>(m, "TagObservation")
      .def(py::init<int, int, py::EigenDRef<Eigen::Matrix<double, 8, 1>>>(),
           py::arg("frame_id"), py::arg("tag_id"), py::arg("observation"));

  m.def("tag_slam_ba", &tag_slam::tag_slam_ba, "Tag slam bundle adjustment",
        py::arg("fxfycxcy"), py::arg("camera_poses"), py::arg("observations"),
        py::arg("tags"), py::arg("tag_side_length"));
}

void init_depth_slam(py::module_ &m) {
  py::class_<depth_slam::CameraPose>(m, "CameraPose")
      .def(py::init<int, py::EigenDRef<Eigen::Matrix4d>>(), py::arg("frame_id"),
           py::arg("camera_pose"))
      .def_readonly("frame_id", &depth_slam::CameraPose::frame_id)
      .def_readonly("camera_pose", &depth_slam::CameraPose::camera_pose);

  py::class_<depth_slam::Landmark>(m, "Landmark")
      .def(py::init<int, py::EigenDRef<Eigen::Vector3d>>(), py::arg("id"),
           py::arg("loc"))
      .def_readonly("id", &depth_slam::Landmark::id)
      .def_readonly("loc", &depth_slam::Landmark::loc);

  py::class_<depth_slam::LandmarkProjectionObservation>(
      m, "LandmarkProjectionObservation")
      .def(py::init<int, int, py::EigenDRef<Eigen::Vector2d>,
                    std::optional<double>>(),
           py::arg("frame_id"), py::arg("landmark_id"), py::arg("observation"),
           py::arg("depth"));

  m.def("depth_slam_ba", &depth_slam::depth_slam_ba,
        "Depth slam bundle adjustment", py::arg("camera_cal"),
        py::arg("camera_poses"), py::arg("observations"), py::arg("landmarks"));
}

void init_symforce_exercises(py::module_ &m) {
  py::class_<PoseGraphEdge>(m, "PoseGraphEdge")
      .def(py::init<int, int, py::EigenDRef<Eigen::Matrix4d>>(),
           py::arg("v1_id"), py::arg("v2_id"), py::arg("v1_to_v2"));
  py::class_<PoseGraphVertex>(m, "PoseGraphVertex")
      .def(py::init<int, py::EigenDRef<Eigen::Matrix4d>>(), py::arg("id"),
           py::arg("pose"));

  m.def("pose_graph_ba", &pose_graph_ba, "Pose graph BA with symforce",
        py::arg("vertices"), py::arg("edges"));

  auto tag_slam_module = m.def_submodule("tag_slam");
  init_tag_slam(tag_slam_module);

  auto depth_slam_module = m.def_submodule("depth_slam");
  init_depth_slam(depth_slam_module);
}

void init_utils(py::module_ &m) {
  py::class_<DBoW3::BowVector>(m, "BowVector");

  py::class_<DBoW3::Result>(m, "BowResult")
      .def_readonly("entry_id", &DBoW3::Result::Id)
      .def_readonly("score", &DBoW3::Result::Score)
      .def("__str__", &info<DBoW3::Result>)
      .def("__repr__", &info<DBoW3::Result>);

  py::class_<DBoW3::Vocabulary>(m, "BowVocabulary")
      .def(py::init<>())
      .def("create", &vocab_create, py::arg("descriptors"))
      .def("save", &vocab_save, py::arg("path"))
      .def("load", &vocab_load, py::arg("path"))
      .def("info", [](DBoW3::Vocabulary &vocab) { return info(vocab); })
      .def("transform", &vocab_transform, py::arg("descriptor"))
      .def("score", &DBoW3::Vocabulary::score, py::arg("v1"), py::arg("v2"))
      .def("__str__", &info<DBoW3::Vocabulary>)
      .def("__repr__", &info<DBoW3::Vocabulary>);

  py::class_<DBoW3::Database>(m, "BowDatabase")
      .def(py::init<DBoW3::Vocabulary &, bool, int>(), py::arg("vocab"),
           py::arg("use_direct_index") = true,
           py::arg("direct_index_levels") = 0)
      .def("add", &db_add, py::arg("desc"))
      .def("info", [](DBoW3::Database &obj) { return info(obj); })
      .def("query", &db_query, py::arg("descriptors"), py::arg("max_results"))
      .def("__str__", &info<DBoW3::Database>)
      .def("__repr__", &info<DBoW3::Database>);

  py::class_<CameraParams>(m, "CameraParams")
      .def(py::init<double, double, double, double>(), py::arg("fx"),
           py::arg("fy"), py::arg("cx"), py::arg("cy"))
      .def(py::init<py::EigenDRef<Eigen::Matrix3d>,
                    py::EigenDRef<Eigen::Matrix<double, 14, 1>>, unsigned int,
                    unsigned int>(),
           py::arg("K"), py::arg("dist_coeffs"), py::arg("width"),
           py::arg("height"))
      .def("__repr__", [](CameraParams &cam) {
        return std::format(
            "CameraParams(fx={}, fy={}, cx={}, cy={}, width={}, height={})",
            cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.width, cam.height);
      });
}

PYBIND11_MODULE(foundation, m) {
  m.doc() = R"pbdoc(
        Bindings to the foundation.
        ---------------------------
    )pbdoc";

  m.def(
      "set_spdlog_level",
      [](const std::string &level) {
        spdlog::set_level(spdlog::level::from_str(level));
      },
      "Set spd log level. Supported levels are: trace, debug, info, warn, "
      "error, critical, off.");

  auto reinventing = m.def_submodule("reinventing", "Reinventing the wheel");
  init_reinventing(reinventing);

  auto spatial = m.def_submodule("spatial", "Spatial stuff");
  init_spatial(spatial);

  auto symforce_exercises =
      m.def_submodule("symforce_exercises", "Symforce Exercises");
  init_symforce_exercises(symforce_exercises);

  auto utils = m.def_submodule("utils", "Utilities");
  init_utils(utils);

  auto oak_slam = m.def_submodule("oak_slam", "Oak slam broseph");
  oak_slam::init_oak_slam(oak_slam);
}

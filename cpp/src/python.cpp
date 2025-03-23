#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <foundation/dbow/python.hpp>
#include <foundation/depth_slam/python.hpp>
#include <foundation/oak_slam/python.hpp>
#include <foundation/pose_graph/python.hpp>
#include <foundation/spatial/python.hpp>
#include <foundation/tag_slam/python.hpp>
#include <foundation/utils/python.hpp>

// NOTE: This sets compile time level. In addition, you need to set the
// runtime level low enough to show these (e.g. trace for everything)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include <spdlog/spdlog.h>

#include <fstream>

namespace py = pybind11;

using namespace foundation;

PYBIND11_MODULE(foundation, m) {
  m.doc() = R"pbdoc(
        Bindings to the foundation.
        ---------------------------
    )pbdoc";

  m.def(
      "set_spdlog_level",
      [](const std::string& level) {
        spdlog::set_level(spdlog::level::from_str(level));
      },
      "Set spd log level. Supported levels are: trace, debug, info, warn, "
      "error, critical, off.");

  auto spatial = m.def_submodule("spatial", "Spatial stuff");
  init_spatial(spatial);

  auto utils = m.def_submodule("utils", "Utilities");
  init_utils(utils);

  auto oak_slam = m.def_submodule("oak_slam", "Oak slam broseph");
  oak_slam::init_oak_slam(oak_slam);

  auto depth_slam_module = m.def_submodule("depth_slam");
  init_depth_slam(depth_slam_module);

  auto tag_slam_module = m.def_submodule("tag_slam");
  init_tag_slam(tag_slam_module);

  auto dbow_module = m.def_submodule("dbow");
  init_dbow(dbow_module);

  auto pose_graph_module = m.def_submodule("pose_graph");
  init_pose_graph(pose_graph_module);
}

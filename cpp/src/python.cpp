#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <foundation/dbow/python.hpp>
#include <foundation/oak_slam/python.hpp>
#include <foundation/spatial/python.hpp>
#include <foundation/utils/python.hpp>

// NOTE: This sets compile time level. In addition, you need to set the
// runtime level low enough to show these (e.g. trace for everything)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include <spdlog/spdlog.h>

namespace py = pybind11;

using namespace foundation;

PYBIND11_MODULE(
    foundation,
    m
) {
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
        "error, critical, off."
    );

    auto spatial = m.def_submodule("spatial", "Spatial stuff");
    init_spatial(spatial);

    auto utils = m.def_submodule("utils", "Utilities");
    init_utils(utils);

    auto oak_slam = m.def_submodule("oak_slam", "Oak slam broseph");
    oak_slam::init_oak_slam(oak_slam);

    auto dbow_module = m.def_submodule("dbow");
    init_dbow(dbow_module);
}

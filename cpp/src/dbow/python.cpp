#include <DBow3/DBoW3.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <foundation/dbow/dbow.hpp>
#include <foundation/utils/python_utils.hpp>

namespace foundation {

void init_dbow(py::module_& m) {
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
      .def("info", [](DBoW3::Vocabulary& vocab) { return info(vocab); })
      .def("transform", &vocab_transform, py::arg("descriptor"))
      .def("score", &DBoW3::Vocabulary::score, py::arg("v1"), py::arg("v2"))
      .def("__str__", &info<DBoW3::Vocabulary>)
      .def("__repr__", &info<DBoW3::Vocabulary>);

  py::class_<DBoW3::Database>(m, "BowDatabase")
      .def(py::init<DBoW3::Vocabulary&, bool, int>(), py::arg("vocab"),
           py::arg("use_direct_index") = true,
           py::arg("direct_index_levels") = 0)
      .def("add", &db_add, py::arg("desc"))
      .def("info", [](DBoW3::Database& obj) { return info(obj); })
      .def("query", &db_query, py::arg("descriptors"), py::arg("max_results"))
      .def("__str__", &info<DBoW3::Database>)
      .def("__repr__", &info<DBoW3::Database>);
}
}  // namespace foundation
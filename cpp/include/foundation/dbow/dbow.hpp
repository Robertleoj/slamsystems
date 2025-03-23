#pragma once
#include <DBow3/DBoW3.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <filesystem>
#include <foundation/utils/numpy.hpp>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

namespace foundation {

cv::Mat desc_array_to_cv(
    py::array_t<float, py::array::c_style>& desc
) {
    auto buf = desc.request();
    std::vector<int> shape(buf.shape.begin(), buf.shape.end());
    return cv::Mat(shape, CV_32FC1, buf.ptr);
}

void vocab_create(
    DBoW3::Vocabulary& vocab,
    std::vector<py::array_t<float, py::array::c_style>>& descriptors
) {
    std::vector<cv::Mat> descriptors_mat;

    for (auto& desc : descriptors) {
        descriptors_mat.push_back(desc_array_to_cv(desc));
    }

    vocab.create(descriptors_mat);
}
void vocab_save(
    DBoW3::Vocabulary& vocab,
    std::filesystem::path& path
) {
    vocab.save(path.string());
}
void vocab_load(
    DBoW3::Vocabulary& vocab,
    std::filesystem::path& path
) {
    vocab.load(path.string());
}

DBoW3::BowVector vocab_transform(
    DBoW3::Vocabulary& vocab,
    py::array_t<float, py::array::c_style> descriptor
) {
    auto descriptor_cv = desc_array_to_cv(descriptor);

    DBoW3::BowVector vec;
    vocab.transform(descriptor_cv, vec);

    return vec;
}
void db_add(
    DBoW3::Database& db,
    py::array_t<float, py::array::c_style>& desc
) {
    auto desc_cv = desc_array_to_cv(desc);

    db.add(desc_cv);
}

std::vector<DBoW3::Result> db_query(
    DBoW3::Database& db,
    py::array_t<float, py::array::c_style>& desc,
    int max_results
) {
    auto desc_cv = desc_array_to_cv(desc);

    DBoW3::QueryResults qr;

    db.query(desc_cv, qr, max_results);

    return qr;
}

}  // namespace foundation
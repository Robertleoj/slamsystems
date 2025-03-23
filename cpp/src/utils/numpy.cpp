#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <foundation/types.hpp>
#include <foundation/utils/numpy.hpp>
#include <vector>

namespace py = pybind11;

namespace foundation {

std::vector<Eigen::Vector3d> array_to_3d_points(
    double_array arr
) {
    arr = arr.squeeze();

    if (!is_shape(arr, {std::nullopt, 3})) {
        throw std::runtime_error("Must have shape (N, 3)");
    }

    std::vector<Eigen::Vector3d> out;
    for (int n = 0; n < arr.shape(0); n++) {
        Eigen::Vector3d vec;
        for (int d = 0; d < 3; d++) {
            vec(d) = arr.at(n, d);
        }
        out.push_back(std::move(vec));
    }
    return out;
}

Eigen::Vector3d array_to_3d_point(
    double_array arr
) {
    arr = arr.squeeze();

    if (!is_shape(
            arr,
            {
                3,
            }
        )) {
        throw std::runtime_error("Must have shape (3,)");
    }

    Eigen::Vector3d out;
    for (int d = 0; d < arr.shape(0); d++) {
        out(d) = arr.at(d);
    }
    return out;
}

std::vector<Eigen::Vector2d> array_to_2d_points(
    double_array arr
) {
    arr = arr.squeeze();

    if (!is_shape(arr, {std::nullopt, 2})) {
        throw std::runtime_error("Must have shape (N, 2)");
    }

    std::vector<Eigen::Vector2d> out;
    for (int n = 0; n < arr.shape(0); n++) {
        Eigen::Vector2d vec;
        for (int d = 0; d < 2; d++) {
            vec(d) = arr.at(n, d);
        }
        out.push_back(std::move(vec));
    }

    return out;
}

Eigen::Matrix3d array_to_3mat(
    double_array arr
) {
    arr = arr.squeeze();

    if (!is_shape(arr, {3, 3})) {
        throw std::runtime_error("Must have shape (3, 3)");
    }

    Eigen::Matrix3d out;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            out(r, c) = arr.at(r, c);
        }
    }
    return out;
}
cv::Mat img_numpy_to_mat_uint8(
    py::array_t<uint8_t, py::array::c_style> input
) {
    py::buffer_info buf = input.request();

    if (buf.ndim < 2 || buf.ndim > 3) {
        throw std::runtime_error(
            "Input should have 2 (grayscale) or 3 (color) dimensions"
        );
    }
    if (buf.ndim == 3 && buf.shape[2] != 3) {
        throw std::runtime_error("Color image should have three channels");
    }

    int type = (buf.ndim == 2) ? CV_8UC1 : CV_8UC3;

    return cv::Mat(buf.shape[0], buf.shape[1], type, buf.ptr);
}

py::array_t<uint8_t> mat_to_numpy_uint8(
    const cv::Mat& mat
) {
    if (mat.empty()) {
        throw std::runtime_error("Input cv::Mat is empty!");
    }

    // Check if mat type is 8-bit unsigned int
    if (mat.depth() != CV_8U) {
        throw std::runtime_error(
            "Unsupported Mat type. Only CV_8U (8-bit unsigned) is supported."
        );
    }

    // Determine the shape based on the number of channels
    std::vector<std::size_t> shape;

    if (mat.channels() == 1) {
        shape = {
            static_cast<std::size_t>(mat.rows),
            static_cast<std::size_t>(mat.cols)
        };
    } else if (mat.channels() == 3) {
        shape = {
            static_cast<std::size_t>(mat.rows),
            static_cast<std::size_t>(mat.cols),
            3
        };
    } else {
        throw std::runtime_error(
            "Unsupported number of channels. Only CV_8UC1 and CV_8UC3 are "
            "supported."
        );
    }

    return py::array_t<uint8_t>(
        shape,    // Shape of the numpy array
        mat.data  // Pointer to the data
    );
}

}  // namespace foundation

#pragma once

#include <opencv2/opencv.hpp>

namespace foundation {
namespace img_utils {

cv::Mat to_greyscale(cv::Mat& img);

}  // namespace img_utils
}  // namespace foundation
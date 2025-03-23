#include <foundation/utils/image.hpp>

namespace foundation {
namespace img_utils {

cv::Mat to_greyscale(
    cv::Mat& img
) {
    if (img.channels() == 1) {
        // already greyscale
        return img;
    }

    cv::Mat grey_img;
    cv::cvtColor(img, grey_img, cv::COLOR_RGB2GRAY);

    return grey_img;
}

}  // namespace img_utils
}  // namespace foundation
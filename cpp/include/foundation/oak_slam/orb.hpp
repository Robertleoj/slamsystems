#pragma once

#include <foundation/oak_slam/type_defs.hpp>
#include <foundation/utils/image.hpp>
#include <opencv2/opencv.hpp>

namespace foundation {
namespace oak_slam {

struct OrbDetection {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    OrbDetection(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors);

    // Move constructor
    OrbDetection(OrbDetection&& other);
};

class OrbDetector {
   public:
    OrbDetector();

    OrbDetection detect(cv::Mat img);

   private:
    // detectors for the center color and side mono cameras
    cv::Ptr<cv::ORB> detector;
};

struct FeatureMatchFrameData {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    FeatureMatchFrameData() {}

    FeatureMatchFrameData(
        FeatureMatchFrameData&& other
    ) noexcept
        : keypoints(std::move(other.keypoints)),
          descriptors(std::move(other.descriptors)) {}
};

struct TripleMatch {
    std::map<OakCamLoc::Enum, FeatureMatchFrameData> match;

    int num_matches() { return match.begin()->second.keypoints.size(); }
};

class OrbMatcher {
   public:
    OrbMatcher();
    std::vector<cv::DMatch>
    good_matches(cv::Mat des1, cv::Mat des2, int min_num_matches = 10);

    TripleMatch make_triple_match(
        OrbDetection& center_orb,
        OrbDetection& left_orb,
        OrbDetection& right_orb
    );

   private:
    cv::FlannBasedMatcher matcher;

    static cv::FlannBasedMatcher make_matcher();
};

}  // namespace oak_slam
}  // namespace foundation
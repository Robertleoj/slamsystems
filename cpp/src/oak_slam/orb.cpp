#include <Remotery.h>
#include <spdlog/spdlog.h>
#include <format>

#include <foundation/oak_slam/orb.hpp>
#include <foundation/utils/camera.hpp>

namespace foundation {
namespace oak_slam {

OrbDetection::OrbDetection(
    std::vector<cv::KeyPoint> keypoints,
    cv::Mat descriptors
)
    : keypoints(keypoints),
      descriptors(descriptors) {}

// Move constructor
OrbDetection::OrbDetection(
    OrbDetection&& other
)
    : keypoints(std::move(other.keypoints)),
      descriptors(std::move(other.descriptors)) {}

OrbDetector::OrbDetector() {
    detector = cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE);
}

OrbDetection OrbDetector::detect(
    cv::Mat img
) {
    auto grey = img_utils::to_greyscale(img);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    rmt_BeginCPUSample(detect_and_compute, 0);
    detector->detectAndCompute(grey, cv::noArray(), keypoints, descriptors);
    rmt_EndCPUSample();

    return OrbDetection(std::move(keypoints), descriptors);
}

OrbMatcher::OrbMatcher()
    : matcher(OrbMatcher::make_matcher()) {}

cv::FlannBasedMatcher OrbMatcher::make_matcher() {
    return cv::FlannBasedMatcher(
        cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)
    );
}

std::vector<cv::DMatch> OrbMatcher::good_matches(
    cv::Mat descriptors1,
    cv::Mat descriptors2,
    int min_num_matches
) {
    std::vector<cv::DMatch> matches;
    spdlog::debug(std::format(
        "Making good matches from {} and {} descriptors",
        descriptors1.rows,
        descriptors2.rows
    ));

    if (descriptors1.rows == 0 || descriptors2.rows == 0) {
        return std::vector<cv::DMatch>();
    }
    matcher.match(descriptors1, descriptors2, matches);

    std::sort(
        matches.begin(),
        matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        }
    );

    if (matches.empty()) {
        return {};
    }

    float minDist = matches.front().distance;
    std::vector<cv::DMatch> good_matches;

    for (size_t i = 0; i < matches.size(); ++i) {
        if (i >= static_cast<size_t>(min_num_matches) &&
            matches[i].distance > 2 * minDist) {
            break;
        }
        good_matches.push_back(matches[i]);
    }

    return good_matches;
}

TripleMatch OrbMatcher::make_triple_match(
    OrbDetection& center_orb,
    OrbDetection& left_orb,
    OrbDetection& right_orb
) {
    rmt_ScopedCPUSample(make_triple_match, 0);
    spdlog::debug("getting good matches");
    std::vector<cv::DMatch> center_to_left_matches =
        good_matches(center_orb.descriptors, left_orb.descriptors, 50);

    std::vector<cv::DMatch> center_to_right_matches =
        good_matches(center_orb.descriptors, right_orb.descriptors, 50);

    // filter out matches that don't appear on both
    // sides
    std::vector<cv::KeyPoint> center_keypoints;
    std::vector<cv::Mat> center_descriptors;

    std::vector<cv::KeyPoint> left_keypoints;
    std::vector<cv::Mat> left_descriptors;

    std::vector<cv::KeyPoint> right_keypoints;
    std::vector<cv::Mat> right_descriptors;

    spdlog::debug("finding triple matches");
    for (int cl_idx = 0; cl_idx < center_to_left_matches.size(); cl_idx++) {
        auto& cl_match = center_to_left_matches[cl_idx];
        auto center_idx = cl_match.queryIdx;
        auto left_idx = cl_match.trainIdx;

        for (int cr_idx = 0; cr_idx < center_to_right_matches.size();
             cr_idx++) {
            auto& cr_match = center_to_right_matches[cr_idx];

            if (cr_match.queryIdx == center_idx) {
                auto right_idx = cr_match.trainIdx;

                // found a match
                center_keypoints.push_back(center_orb.keypoints[center_idx]);
                center_descriptors.push_back(
                    center_orb.descriptors.row(center_idx)
                );

                left_keypoints.push_back(left_orb.keypoints[left_idx]);
                left_descriptors.push_back(left_orb.descriptors.row(left_idx));

                right_keypoints.push_back(right_orb.keypoints[right_idx]);
                right_descriptors.push_back(right_orb.descriptors.row(right_idx)
                );
            }
        }
    }
    spdlog::debug(
        std::format("Found {} triple matches", center_keypoints.size())
    );

    TripleMatch triple_match;

    FeatureMatchFrameData center_match_data{};
    center_match_data.keypoints = std::move(center_keypoints);
    cv::vconcat(center_descriptors, center_match_data.descriptors);

    triple_match.match.emplace(OakCamLoc::CENTER, std::move(center_match_data));

    FeatureMatchFrameData left_match_data{};
    left_match_data.keypoints = std::move(left_keypoints);
    cv::vconcat(left_descriptors, left_match_data.descriptors);

    triple_match.match.emplace(OakCamLoc::LEFT, std::move(left_match_data));

    FeatureMatchFrameData right_match_data{};
    right_match_data.keypoints = std::move(right_keypoints);
    cv::vconcat(right_descriptors, right_match_data.descriptors);

    triple_match.match.emplace(OakCamLoc::RIGHT, std::move(right_match_data));

    return triple_match;
}

}  // namespace oak_slam
}  // namespace foundation
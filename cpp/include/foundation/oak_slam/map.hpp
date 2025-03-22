#include <Eigen/Core>
#include <foundation/oak_slam/ID.hpp>
#include <foundation/oak_slam/type_defs.hpp>
#include <map>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace foundation {
namespace oak_slam {

// ID definitions
struct FeatureTag {};
struct LandmarkTag {};
struct KeyframeTag {};

using FeatureID = ID<FeatureTag>;
using LandmarkID = ID<LandmarkTag>;
using KeyframeID = ID<KeyframeTag>;

/**
 *  an ORB feature detected in keyframes.
 */
class Feature {
 public:
  Feature(std::optional<LandmarkID> landmark_id = std::nullopt)
      : id(FeatureID::next()), landmark_id(landmark_id) {}

  FeatureID id;
  std::optional<LandmarkID> landmark_id;
};

/**
 * used to reference both the keyframe and the feature a landmark points to
 */
struct FeatureRef {
  KeyframeID keyframe_id;
  OakCamLoc::Enum cam;
  // invariant: this feature must be contained in the keyframe, or not exist
  FeatureID feature_id;
};
class Landmark {
 public:
  Landmark() : id(LandmarkID::next()) {}
  LandmarkID id;
  Eigen::Vector3d location;
  // all features that detected this landmark
  std::map<FeatureID, FeatureRef> features;

  void add_feature(FeatureID feature_id,
                   KeyframeID keyframe_id,
                   OakCamLoc::Enum cam) {
    FeatureRef ref{
        .keyframe_id = keyframe_id, .cam = cam, .feature_id = feature_id};

    features.insert({feature_id, ref});
  }
};

class Keyframe {
 public:
  KeyframeID id;
  std::map<OakCamLoc::Enum, std::map<FeatureID, Feature>> features;

  Keyframe() : id(KeyframeID::next()) {}

  void add_feature(OakCamLoc::Enum which, Feature&& feature) {
    features[which].insert({feature.id, std::move(feature)});
  }
};

class Map {
 public:
  std::map<KeyframeID, Keyframe> keyframes;
  std::map<LandmarkID, Landmark> landmarks;

  void add_keyframe(Keyframe&& keyframe) {
    keyframes.insert({keyframe.id, std::move(keyframe)});
  }
};

}  // namespace oak_slam
}  // namespace foundation

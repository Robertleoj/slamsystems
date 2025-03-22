#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace foundation {

template <typename T, int R, int C>
cv::Mat eigen_to_cv_mat(Eigen::Matrix<T, R, C> eig) {
  cv::Mat m;
  cv::eigen2cv(eig, m);
  return m;
}

template <typename T>
cv::Point_<T> eigen_vec2_to_cv(Eigen::Vector2<T> eigen_vec) {
  return cv::Point_<T>(eigen_vec.x(), eigen_vec.y());
}

template <typename T>
std::vector<cv::Point_<T>> eigen_vecs2_to_cv(
    std::vector<Eigen::Vector2<T>> eigen_vecs) {
  std::vector<cv::Point_<T>> cv_vecs;

  for (auto& eigen_vec : eigen_vecs) {
    cv_vecs.push_back(eigen_vec2_to_cv(eigen_vec));
  }
  return cv_vecs;
}

template <typename T>
Eigen::Vector2<T> cv_vec2_to_eigen(cv::Point_<T> cv_vec) {
  return Eigen::Vector2<T>(cv_vec.x, cv_vec.y);
}

template <typename T>
std::vector<Eigen::Vector2<T>> cv_vecs2_to_eigen(
    std::vector<cv::Point_<T>> cv_vecs) {
  std::vector<Eigen::Vector2<T>> eig_vecs;
  for (auto& cv_vec : cv_vecs) {
    eig_vecs.push_back(cv_vec2_to_eigen(cv_vec));
  }
  return eig_vecs;
}

template <typename T>
cv::Point3_<T> eigen_vec3_to_cv(Eigen::Vector3<T> eigen_vec) {
  return cv::Point3_<T>(eigen_vec.x(), eigen_vec.y(), eigen_vec.z());
}

template <typename T>
std::vector<cv::Point3_<T>> eigen_vecs3_to_cv(
    std::vector<Eigen::Vector3<T>> eigen_vecs) {
  std::vector<cv::Point3_<T>> cv_vecs;

  for (auto& eigen_vec : eigen_vecs) {
    cv_vecs.push_back(eigen_vec3_to_cv(eigen_vec));
  }
  return cv_vecs;
}

template <typename T>
Eigen::Vector3<T> cv_vec3_to_eigen(cv::Point3_<T> cv_vec) {
  return Eigen::Vector3<T>(cv_vec.x, cv_vec.y, cv_vec.z);
}

template <typename T>
std::vector<Eigen::Vector3<T>> cv_vecs3_to_eigen(
    std::vector<cv::Point3_<T>> cv_vecs) {
  std::vector<Eigen::Vector3<T>> eig_vecs;
  for (auto& cv_vec : cv_vecs) {
    eig_vecs.push_back(cv_vec3_to_eigen(cv_vec));
  }
  return eig_vecs;
}

}  // namespace foundation
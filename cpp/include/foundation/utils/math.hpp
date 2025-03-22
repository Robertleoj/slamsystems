#include <optional>
#include <vector>

namespace foundation {

template <typename T>
std::optional<T> get_median(std::vector<T> &nums) {
  if (nums.empty()) return std::nullopt;  // Return NaN if empty

  std::sort(nums.begin(), nums.end());  // Step 1: Sort

  size_t mid = nums.size() / 2;  // Step 2: Find middle index

  // Step 3: Handle even/odd cases
  if (nums.size() % 2 == 0) {
    return (nums[mid - 1] + nums[mid]) / 2.0;  // Average of middle two
  }

  return nums[mid];  // Middle element
}

}  // namespace foundation
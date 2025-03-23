#pragma once
#include <string>

namespace foundation {
template <typename T>
std::string info(T& obj) {
  std::stringstream s;
  s << obj;
  return s.str();
}

}  // namespace foundation
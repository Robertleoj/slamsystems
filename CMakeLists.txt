cmake_minimum_required(VERSION 3.22)
project(project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux")

set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "")


message(STATUS "${CMAKE_TOOLCHAIN_FILE}")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

find_package(pybind11 REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
# set(OpenCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux/opencv4/opencv2")
find_package(OpenCV REQUIRED)

set(LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

set(INSTALLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_installs")
include_directories(${INSTALLS_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/external/Remotery/lib)
link_directories(${INSTALLS_DIR}/lib)

add_subdirectory(external/DBow3)
execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/external/DBow3/src/ ${INSTALLS_DIR}/include/DBow3)

add_subdirectory(cpp/src)

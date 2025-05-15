cmake_minimum_required(VERSION 3.22)
project(project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Vcpkg setup
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux")
set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "")
message(STATUS "${CMAKE_TOOLCHAIN_FILE}")

# ccache for the win
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()


# 🧱 External deps
find_package(pybind11 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

# Optional local includes
set(LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
include_directories(
    ${LOCAL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/Remotery/lib
)

# Optional FetchContent dep (slam_dunk)
# include(FetchContent)
# FetchContent_Declare(slamd
#     SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/slam_dunk
#     SOURCE_SUBDIR slamd
# )
# FetchContent_MakeAvailable(slamd)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/slam_dunk/slamd)

# Your actual code
add_subdirectory(cpp/src)

target_link_libraries(foundation 

PRIVATE
    spdlog::spdlog 
    Eigen3::Eigen 
    Sophus::Sophus 
    opencv_core 
    slamd::slamd
)

target_include_directories(foundation PRIVATE ${LOCAL_INCLUDE_DIR} ${CPP_INSTALLS_DIR}/include)

# Optional linker flags
target_link_options(foundation PRIVATE 
    "-Wl,--no-as-needed"
)

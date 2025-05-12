cmake_minimum_required(VERSION 3.22)
project(project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux")

set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "")

message(STATUS "${CMAKE_TOOLCHAIN_FILE}")

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

include(ExternalProject)

ExternalProject_Add(symforce_ext
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/symforce  # 🔥 Local path to the SymForce repo
  BINARY_DIR ${CMAKE_BINARY_DIR}/symforce-build
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/symforce-install
    -DSYMFORCE_BUILD_OPT=ON
    -DSYMFORCE_BUILD_CC_SYM=OFF         # 🔥 No Python bindings
    -DSYMFORCE_ADD_PYTHON_TESTS=OFF     # 🔥 No Python tests
    -DSYMFORCE_BUILD_SYMENGINE=OFF      # 🔥 No symenginepy junk
    -DSYMFORCE_BUILD_EXAMPLES=OFF
    -DSYMFORCE_BUILD_TESTS=OFF
    -DSYMFORCE_GENERATE_MANIFEST=OFF
  INSTALL_DIR ${CMAKE_BINARY_DIR}/symforce-install
  INSTALL_COMMAND cmake --install . --prefix <INSTALL_DIR>
)


include_directories(${CMAKE_BINARY_DIR}/symforce-install/include)
link_directories(${CMAKE_BINARY_DIR}/symforce-install/lib)

find_package(pybind11 REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

set(LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/external/Remotery/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DBow3/src
)
# link_directories(${INSTALLS_DIR}/lib)

set(SYMFORCE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SYMFORCE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# set(SYMFORCE_BUILD_OPT ON CACHE BOOL "" FORCE)

add_subdirectory(external/DBow3)
add_subdirectory(cpp/src)

target_link_options(foundation PRIVATE 
  "-Wl,--no-as-needed"
)
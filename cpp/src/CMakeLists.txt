set(FILES
    spatial/lie_algebra.cpp
    spatial/python.cpp

    utils/camera.cpp
    utils/numpy.cpp
    utils/image.cpp
    utils/python.cpp

    oak_slam/orb.cpp
    oak_slam/python.cpp
    oak_slam/triangulation.cpp

    depth_slam/python.cpp

    pose_graph/python.cpp

    tag_slam/python.cpp

    optim/symforce/conversion.cpp

    dbow/python.cpp

    ../../external/Remotery/lib/Remotery.c
)

pybind11_add_module(foundation python.cpp ${FILES})

target_link_libraries(foundation PRIVATE 
    spdlog::spdlog 
    Eigen3::Eigen 
    Sophus::Sophus 
    opencv_core 
    symforce_gen 
    symforce_opt 
    symforce_cholesky 
    symforce_slam 
    DBoW3
    ${Pangolin_LIBRARIES}
)

target_include_directories(foundation PRIVATE ${LOCAL_INCLUDE_DIR})

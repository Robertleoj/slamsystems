set(FILES
    reinventing/solve_pnp_g2o.cpp
    reinventing/poly_fit_ceres.cpp
    reinventing/poly_fit_g2o.cpp
    reinventing/solve_pnp_ceres.cpp
    reinventing/icp_g2o.cpp
    reinventing/icp_ceres.cpp
    spatial/lie_algebra.cpp
    utils/camera.cpp
    utils/numpy.cpp
    utils/rerun.cpp
    utils/image.cpp
    optim/g2o/vertices.cpp
    optim/g2o/edges.cpp
    oak_slam/rerun_logger.cpp
    oak_slam/orb.cpp
    oak_slam/python.cpp
    oak_slam/triangulation.cpp
    ../../external/Remotery/lib/Remotery.c
)

pybind11_add_module(foundation python.cpp ${FILES})

target_link_libraries(foundation PRIVATE spdlog::spdlog Eigen3::Eigen Ceres::ceres g2o::core g2o::stuff g2o::solver_dense g2o::types_slam2d Sophus::Sophus ${OpenCV_LIBS} symforce_gen symforce_opt symforce_cholesky symforce_slam DBoW3 rerun_sdk)

target_include_directories(foundation PRIVATE ${LOCAL_INCLUDE_DIR})

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

    ../../external/Remotery/lib/Remotery.c
)

set_source_files_properties(
    ../../external/Remotery/lib/Remotery.c
    PROPERTIES
    COMPILE_FLAGS "-Wno-nonnull -Wno-error"
)


pybind11_add_module(foundation python.cpp ${FILES})

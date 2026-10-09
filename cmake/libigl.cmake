if(TARGET igl::core)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.6.4
)
FetchContent_MakeAvailable(libigl)

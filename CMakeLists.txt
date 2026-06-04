cmake_minimum_required(VERSION 3.21)


# CUDA setup
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)


set(BUILD_SHARED_LIBS OFF)

project(NewtonBioMorphX LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "--extended-lambda")

find_package(gmsh REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale )
pkg_check_modules(GMP REQUIRED gmp)
pkg_check_modules(MPFR REQUIRED mpfr)
# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(SDL2 REQUIRED QUIET)
find_package(Freetype REQUIRED)

find_package(VTK REQUIRED COMPONENTS
        CommonCore
        CommonDataModel
        IOXML
        IOLegacy
)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_OPENGL "Use OpenGL" ON)
option(LIBIGL_GLFW   "Use GLFW"   ON)
option(LIBIGL_IMGUI  "Use ImGui"  ON)
option(LIBIGL_STB    "Use STB"    ON)
include(libigl)

#include_directories(${FREETYPE_INCLUDE_DIRS})
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIBRARY_DIRS})
# ---------------------------------------------------------
# ✅ Executable
# ---------------------------------------------------------
add_executable(${PROJECT_NAME}
        src/Main.cu
        src/MeshIO.cu
        src/Kernels.cu
        src/DeviceData.cu
        src/MyGUI.cu
        src/Visualisation.cu
        src/Octree.cu
        src/TextIO.cu
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/imgui-src/backends/imgui_impl_sdl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/imgui-src/backends/imgui_impl_opengl3.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_INCLUDE_PATH="/usr/local/cuda/include")
target_include_directories(${PROJECT_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${gmsh_SOURCE_DIR}/api ${gmsh_BINARY_DIR}/api)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions(${PROJECT_NAME} PRIVATE IGL_VIEWER_VIEWER_QUIET)

# ---------------------------------------------------------
# ✅ Linking
# ---------------------------------------------------------
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        igl::core
        igl::glfw
        igl::imgui
        igl::stb
        Eigen3::Eigen
        SDL2
        glad
        stdc++fs
        ${FFMPEG_LIBRARIES}
        ${VTK_LIBRARIES}
        gmp
        mpfr
        gmsh
        CUDA::nvrtc
        CUDA::cuda_driver
)

set(CMAKE_INSTALL_PREFIX "/home/jahanbak/applications/GrowthSimulator")

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin)
file(GLOB NVRTC_KERNELS "${CMAKE_SOURCE_DIR}/src/nvrtc_kernels/*.cu")
install(FILES ${NVRTC_KERNELS}
        DESTINATION src/nvrtc_kernels)
install(FILES src/Typedefs.h src/Primitives.h src/DeviceDataPtr.h src/SVD3Cuda.h
        DESTINATION src)
install(DIRECTORY share/
        DESTINATION share)
install(DIRECTORY examples/
        DESTINATION examples)

install(CODE " execute_process( COMMAND chmod -R 755 \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}\" ) ")
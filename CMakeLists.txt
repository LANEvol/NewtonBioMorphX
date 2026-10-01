cmake_minimum_required(VERSION 3.21)

# =========================================================
# NewtonBioMorphX -- one build for Windows and Linux
#
#   Shared settings live in this file.
#   Platform-specific parts live in:
#       cmake/PlatformWindows.cmake   (DLLs, icon resource, ZIP/NSIS via CPack)
#       cmake/PlatformLinux.cmake     (static NVRTC, rpath, .so bundling, AppImage)
#   Each platform file defines three hooks that are called below:
#       nbmx_platform_setup()     before dependencies are searched
#       nbmx_platform_target()    after the executable is created
#       nbmx_platform_package()   at the very end (packaging must be the last install step)
#
#   Build:    cmake -S . -B build
#             cmake --build build --config RelWithDebInfo
#   Package:  cmake --install build --config RelWithDebInfo
#             Windows -> install/  + install/packages/*.zip (+ installer)
#             Linux   -> install/AppDir/usr + install/*.AppImage
# =========================================================

# ---------------------------------------------------------
# CUDA architectures (must be set BEFORE project())
#   Real code for Turing..Hopper + PTX (90-virtual) so newer GPUs JIT-compile it.
#   Faster dev builds: -DCMAKE_CUDA_ARCHITECTURES=75-real
#   Volta (70) is left out: CUDA 13 no longer supports it. With CUDA 12 you can add
#   it back with -DCMAKE_CUDA_ARCHITECTURES="70-real;75-real;...".
# ---------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;90-virtual")
    set(CMAKE_CUDA_ARCHITECTURES "89-real")
endif()

project(NewtonBioMorphX VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Icon files (both optional, see the platform files for what they are used for)
set(APP_ICON_ICO "${CMAKE_SOURCE_DIR}/share/icons/${PROJECT_NAME}.ico")
set(APP_ICON_PNG "${CMAKE_SOURCE_DIR}/share/icons/${PROJECT_NAME}.png")

# ---------------------------------------------------------
# Platform file
# ---------------------------------------------------------
if(WIN32)
    include(PlatformWindows)
elseif(UNIX AND NOT APPLE)
    include(PlatformLinux)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

nbmx_platform_setup()

# ---------------------------------------------------------
# Shared dependencies
# ---------------------------------------------------------
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED COMPONENTS
        CommonCore
        CommonDataModel
        IOXML
        IOLegacy
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP  REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)

# libigl
option(LIBIGL_OPENGL "Use OpenGL" ON)
option(LIBIGL_GLFW   "Use GLFW"   ON)
option(LIBIGL_IMGUI  "Use ImGui"  ON)
option(LIBIGL_STB    "Use STB"    ON)
include(libigl)

# ---------------------------------------------------------
# Executable
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

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(${PROJECT_NAME} PRIVATE IGL_VIEWER_VIEWER_QUIET)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        igl::core igl::glfw igl::imgui igl::stb
        Eigen3::Eigen
        glad
        ${VTK_LIBRARIES}
        PkgConfig::GMP PkgConfig::MPFR
        CUDA::cuda_driver            # libcuda / nvcuda come from the user's NVIDIA driver
        Threads::Threads)

# SDL2, FFmpeg, gmsh, NVRTC, platform link flags, icon, build-folder DLLs,
# and installing the executable with its dependencies
nbmx_platform_target()

# ---------------------------------------------------------
# Shared install layout (relative to the install prefix)
#   bin/           executable (+ DLLs on Windows)     lib/ (.so on Linux)
#   src/           NVRTC kernel sources and headers
#   include/cuda/  CUDA headers for NVRTC (users have no CUDA Toolkit)
#   share/         fonts, icons
#   examples/      presets
# ---------------------------------------------------------
file(GLOB NVRTC_KERNELS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/nvrtc_kernels/*.cu")
install(FILES ${NVRTC_KERNELS}
        DESTINATION src/nvrtc_kernels)
install(FILES src/Typedefs.h src/Primitives.h src/DeviceDataPtr.h src/SVD3Cuda.h
        DESTINATION src)
install(DIRECTORY share/
        DESTINATION share)
install(DIRECTORY examples/
        DESTINATION examples)
install(DIRECTORY "${CUDAToolkit_TARGET_DIR}/include/"
        DESTINATION include/cuda)

# ---------------------------------------------------------
# Packaging -- must stay the LAST thing in this file
# ---------------------------------------------------------
nbmx_platform_package()
# NewtonBioMorphX:
NewtonBioMorphX is a **C++17/CUDA** application for GPU-accelerated biomechanical morphogenesis simulation built with **CMake**.

## Requirements

### Operating System

The recommended environment is:

- Ubuntu 22.04
- NVIDIA CUDA 12.4
- NVIDIA GPU with CUDA support

The provided Docker environment is based on:
```
nvidia/cuda:12.4.0-devel-ubuntu22.04
```

---

## Hardware Requirements

### NVIDIA GPU

A CUDA-capable NVIDIA GPU is required.

The project is currently configured for CUDA architecture:
```
75
``` 

This corresponds to compute capability 7.5, commonly used by NVIDIA Turing GPUs such as RTX 20-series and GTX 16-series cards.

If your GPU uses a different compute capability, update the CUDA architecture in `CMakeLists.txt`.

Examples:
```
cmake set(CMAKE_CUDA_ARCHITECTURES 86)
```
for many RTX 30-series GPUs.
```
cmake set(CMAKE_CUDA_ARCHITECTURES 89)
```
for many RTX 40-series GPUs.
You can check your GPU with:
```
nvidia-smi
```

---

## Software Dependencies

The application requires:

- CMake 3.21 or newer
- C++17 compiler
- CUDA Toolkit
- NVIDIA CUDA driver
- Git
- GDB
- pkg-config
- OpenGL / GLX / X11 libraries
- Gmsh
- Eigen3
- SDL2
- Freetype
- VTK
- FFmpeg development libraries
- GMP
- MPFR
- Python 3 and pip

---

## Build and Run Instructions

The application can be run in two modes:

- **GUI mode**: run without arguments.
- **No-GUI / batch mode**: pass a project/settings file as an argument.

Example:

```
./NewtonBioMorphX ../examples/sphere_growth.txt
```

---

### 1. Linux Docker Build and Run

The provided Docker setup is based on NVIDIA CUDA Ubuntu images and is intended for GPU-enabled systems.

Docker GPU support requires the **NVIDIA Container Toolkit**.

---

### 1.1 Verify Docker GPU Support

After installing Docker and NVIDIA Container Toolkit, test GPU access:

```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this prints your GPU information, Docker can access the GPU.

### 1.2 Build the Docker Image

From the project root:

```
docker build -t newton-biomorphx .
```

### 1.3 Run the container

#### GUI mode
Allow the container to access your X11 display:
```
xhost +local:docker
```
Run the container:
```
docker run --rm -it --gpus all -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD/:/workspace/" newton-biomorphx
```
#### No-GUI / Batch mode
Run the container without X11 access:
```
docker run --rm --gpus all -v "$PWD/:/workspace/"   newton-biomorphx
```
---
### 1.4 Build the Application for Docker Packaging

Build it first:
```
cmake -S . -B cmake-build-release-docker_gpu -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release-docker_gpu -j
```

Check that the binary exists:
```
ls cmake-build-release-docker_gpu/NewtonBioMorphX
```

You can run NewtonBioMorphX with or without GUI. After finishing, revoke the permission:
```
xhost -local:docker
```

---

## 2. Linux Native Build and Run

These instructions are intended for Ubuntu 22.04 or a similar Debian-based Linux distribution.

---

### 2.1 Install System Requirements

Install the required build tools and libraries:

```
sudo apt update
sudo apt install -y
build-essential
cmake
git
pkg-config
python3
python3-pip
python-is-python3
mesa-utils
libgl1-mesa-glx
libglx-mesa0
libglu1-mesa
libx11-6
libxext6
libxrender1
libxrandr2
libxcursor1
libxft2
libxinerama1
zenity
ffmpeg
libavcodec-dev
libavutil-dev
libavformat-dev
libswscale-dev
libgmp-dev
libmpfr-dev
libeigen3-dev
libsdl2-dev
libfreetype-dev
libvtk9-dev
```

Install Gmsh:
```
python3 -m pip install --user --force-reinstall --no-cache-dir gmsh==4.15.2
```

If needed, expose the Gmsh shared library to the linker:
```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH sudo ldconfig
```

---

### 2.2 Check CUDA

Verify that the NVIDIA driver and CUDA compiler are available:
```
nvidia-smi nvcc --version
```

If `nvcc` is not found, add CUDA to your environment:
```
export PATH=/usr/local/cuda/bin:PATH export LD_LIBRARY_PATH=/usr/local/cuda/lib64:LD_LIBRARY_PATH
```

---
### 2.3 Configure and Build

From the project root:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release cmake --build build -j
```

The executable should be created at:

```
build/NewtonBioMorphX
```

---

### 2.4 Run on Linux

#### GUI Mode

From the project root:

```
./build/NewtonBioMorphX
```

#### No-GUI / Batch Mode
Run using a project/settings file:
```
./build/NewtonBioMorphX examples/sphere_growth.txt
```

Other examples:
```
./build/NewtonBioMorphX examples/box_growth.txt
./build/NewtonBioMorphX examples/cylinder_theta_growth.txt
./build/NewtonBioMorphX examples/torus.txt
```

---
### 4.5 Optional Install Step

You can install the application using:
```
cmake --install build
```

To choose a custom install directory, configure CMake with `CMAKE_INSTALL_PREFIX`:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/applications/NewtonBioMorphX"
cmake --build build -j cmake --install build
```

Then run:
```
$HOME/applications/NewtonBioMorphX/bin/NewtonBioMorphX
```

---

## 3. Windows Build Options

There are two practical ways to use NewtonBioMorphX on Windows:

1. **Recommended:** WSL2 with Ubuntu and NVIDIA CUDA support.
2. **Native Windows build:** Possible, but requires manually installing and configuring all dependencies.

The WSL2 method is usually easier because it is closer to the Linux build environment.

---

## 3A. Windows Recommended Method: WSL2 + Ubuntu

---

### 3A.1 Install Prerequisites on Windows

Install:

- Windows 10/11 with WSL2 support
- Latest NVIDIA Windows driver with CUDA support for WSL
- Ubuntu 22.04 from Microsoft Store
- Optional: Windows Terminal

Inside WSL, check GPU access:
```
nvidia-smi
```

If this works, CUDA GPU access is available inside WSL.

---
### 3A.2 Installing Build Dependencies inside WSL

Inside Ubuntu/WSL:

```
sudo apt update
sudo apt install -y
build-essential
cmake
git
pkg-config
python3
python3-pip
python-is-python3
mesa-utils
libgl1-mesa-glx
libglx-mesa0
libglu1-mesa
libx11-6
libxext6
libxrender1
libxrandr2
libxcursor1
libxft2
libxinerama1
zenity
ffmpeg
libavcodec-dev
libavutil-dev
libavformat-dev
libswscale-dev
libgmp-dev
libmpfr-dev
libeigen3-dev
libsdl2-dev
libfreetype-dev
libvtk9-dev
```

Install Gmsh:

```
python3 -m pip install --user --force-reinstall --no-cache-dir gmsh==4.15.2
```

Check CUDA:
```
nvcc --version nvidia-smi
```

---
### 3A.3 Build inside WSL

From the project directory inside WSL:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release cmake --build build -j
```

---
### 3A.4 Run inside WSL

#### GUI Mode
```
./build/NewtonBioMorphX
```

On Windows 11, WSLg normally handles Linux GUI windows automatically.

#### No-GUI / Batch Mode
```
./build/NewtonBioMorphX examples/sphere_growth.txt
```

---

## 3B. Native Windows Build

Native Windows compilation is possible, but it is more fragile because all dependencies must be available to CMake in Windows-compatible form.

---

### 3B.1 Install Required Tools

Install:

- Visual Studio 2022
  - Desktop development with C++
  - MSVC compiler
  - Windows SDK
- CMake 3.21 or newer
- NVIDIA CUDA Toolkit
- Git
- vcpkg

---

### 3B.2 Install Dependencies with vcpkg

From PowerShell:
```
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg C:\vcpkg\bootstrap-vcpkg.bat
```

Install dependencies:

```
C:\vcpkg\vcpkg install eigen3:x64-windows
C:\vcpkg\vcpkg install sdl2:x64-windows
C:\vcpkg\vcpkg install freetype:x64-windows
C:\vcpkg\vcpkg install ffmpeg:x64-windows
C:\vcpkg\vcpkg install gmp:x64-windows
C:\vcpkg\vcpkg install mpfr:x64-windows
C:\vcpkg\vcpkg install vtk:x64-windows
C:\vcpkg\vcpkg install gmsh:x64-windows
```

Depending on the vcpkg version, some packages may require additional features or manual configuration.

---
### 3B.3 Configure with CMake
From the project root:
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```
Build:
```
cmake --build build --config Release
```

The executable should be generated in one of the following locations:
```
build\Release\NewtonBioMorphX.exe
```

or:
```
build\NewtonBioMorphX.exe
```

---

### 3B.4 Run on Native Windows
#### GUI Mode
```
.\build\Release\NewtonBioMorphX.exe
```
#### No-GUI / Batch Mode
```
.\build\Release\NewtonBioMorphX.exe .\examples\sphere_growth.txt
```

---

## 4. CUDA Architecture Note

The project is configured for a specific CUDA architecture. If your GPU has a different compute capability, update the CUDA architecture before building.

Common values:

| GPU Generation | CUDA Architecture |
|---|---:|
| Turing | 75 |
| Ampere | 86 |
| Ada Lovelace | 89 |
| Hopper | 90 |

For example, for an RTX 30-series GPU, use architecture `86`.

After changing the architecture, rebuild from a clean build directory.

Linux:

```
rm -rf build cmake -S . -B build -DCMAKE_BUILD_TYPE=Release cmake --build build -j
```

Windows PowerShell:
```
Remove-Item -Recurse -Force build cmake -S . -B build -G "Visual Studio 17 2022" -A x64 cmake --build build --config Release
```

---
## 5. Common Commands Summary

### Docker GUI on Linux
```
xhost +local:docker
docker run --rm -it --gpus all -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD/:/workspace/" newton-biomorphx ./cmake-build-release-docker_gpu/NewtonBioMorphX
xhost -local:docker
```

### Docker No-GUI
```
docker run --rm --gpus all -v "$PWD/:/workspace/" newton-biomorphx ./cmake-build-release-docker_gpu/NewtonBioMorphX
```

### Linux GUI
```
./build/NewtonBioMorphX
```
### Linux No-GUI Example
```
./build/NewtonBioMorphX examples/sphere_growth.txt
```

### Windows WSL2 GUI
```
./build/NewtonBioMorphX
```

### Windows WSL2 No-GUI
```
./build/NewtonBioMorphX examples/sphere_growth.txt
```

### Native Windows GUI
```
.\build\Release\NewtonBioMorphX.exe
```

### Native Windows No-GUI
```
.\build\Release\NewtonBioMorphX.exe .\examples\sphere_growth.txt
```

## License

This project is distributed under the GNU General Public License v3.0 or later.

See the `LICENSE` file for details.
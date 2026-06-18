# NewtonBioMorphX:
NewtonBioMorphX is a GPU‑accelerated finite‑element framework for simulating hyperelastic, plastic,
and viscous tissues with growth, anisotropic passive fibers, and active stresses. It integrates a 
broad set of functionalities into a single platform with mesh generation, material property
assignment, boundary‑conditions, visualization tools, and efficient GPU solvers, enabling researchers
to model diverse morphogenetic processes through an accessible graphical interface.

---

## Operating system, Hardware and Software Dependencies

The recommended environment is:

- Ubuntu 22.04 or WSL2 (Windows Subsystem for Linux)
- NVIDIA CUDA 12.4
- NVIDIA GPU with CUDA support

The project is tested successfully for CUDA architecture:
```
75, 89, 90
``` 
This corresponds to compute capabilities 7.5, 8.9, and 9.0.
- 7.5 is used by NVIDIA Turing GPUs (RTX 20‑series, GTX 16‑series).
- 8.9 is used by NVIDIA Ada Lovelace GPUs (RTX 40‑series and 50-series).
- 9.0 is used by NVIDIA Hopper GPUs (H100 and related data‑center GPUs).

⚠️ Running the application on pre‑7.0 CUDA architectures may cause significant performance degradation.

This application relies on the following libraries:

- `GCC` https://gcc.gnu.org/
- `CUDA Toolkit` https://developer.nvidia.com/cuda/toolkit
- `libigl` https://libigl.github.io/
- `Eigen` https://eigen.tuxfamily.org
- `Dear ImGui` https://github.com/ocornut/imgui
- `zenity` https://gitlab.gnome.org/GNOME/zenity 
- `OpenGL/GLX` https://www.opengl.org/ 
- `GLFW` https://www.glfw.org/
- `Xlib` https://www.x.org/
- `SDL2` https://www.libsdl.org/
- `Gmsh` https://gmsh.info/
- `Freetype` https://freetype.org/
- `VTK` https://vtk.org/
- `FFmpeg` https://www.ffmpeg.org/
- `GMP` https://gmplib.org/
- `MPFR` https://www.mpfr.org/
- `Python 3` https://www.python.org
- `CMake` https://cmake.org/ 
- `pkgconf` https://github.com/pkgconf/pkgconf
---
## 🐳 Running the Application with Docker

Running the application using Docker is the recommended method for users who prefer a simplified setup or do not wish to manage dependencies manually.

---

### 📦 Installing the requirements

#### Docker Installation

Follow the official Docker documentation for installation instructions:

- **Docker Engine (Linux):**  
  https://docs.docker.com/engine/install/

- **Docker Desktop (Windows):**  
  https://docs.docker.com/desktop/windows/install/

#### Windows‑specific requirements

Windows users must ensure the following:

- **WSL2 is installed and enabled**
- **Docker Desktop is configured to use the WSL2 backend**

---

### ▶️ Running the application

After installing Docker, you can run the application by starting the container followed by the application name:
```
docker run --rm --gpus all \
           -e DISPLAY=$DISPLAY \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v "$PWD/:/workspace/" jahanbak/newtonbiomorphx:latest NewtonBioMorphX
```
For Windows users, the WSL terminal can be accessed by pressing the Windows key and typing `wsl`.
The application can be run also in No-GUI / batch mode. For that purpose, first start the container in interactive mode:
```
docker run --rm -it \
       --gpus all \
       -v "$PWD/:/workspace/" jahanbak/newtonbiomorphx:latest
```
Then run the application by passing the project file path as an argument:
```
NewtonBioMorphX /usr/examples/sphere_growth.txt
```

⚠️ For GUI mode, you may need to allow the container to access your X11 display. To do so, run the following command in the terminal before starting the container:
```
xhost +local:docker
```
And after finishing, revoke the permission:
```
xhost -local:docker
```

---

## ⚙️ Building the Application with Docker (for Developers)

The following instructions are for developers who wish to build the application from source.
The provided Docker setup is based on NVIDIA CUDA Ubuntu images and is intended for GPU-enabled systems.
Docker GPU support requires the **NVIDIA Container Toolkit**.

---

### 1. Verify docker GPU support

After installing Docker and NVIDIA Container Toolkit, test GPU access:
```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
If this prints your GPU information, Docker can access the GPU.

### 2. Build the Docker image and run the container

From the project root:
```
docker build -t newtonbiomorphx .
```
Run the container in interactive mode:
```
docker run --rm  -it \
       --gpus all \
       -v "$PWD/:/workspace/" newtonbiomorphx
```
---
### 3. Build the application
Build the application using CMake:
```
cmake -S . \
      -B cmake-build-release-docker_gpu \
      -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release-docker_gpu -j
```
### 4. CUDA Architecture Note

The project is configured for a specific CUDA architecture. If your GPU has a different compute capability, update the CUDA architecture before building.

Common values:

| GPU Generation | CUDA Architecture |
|----------------|------------------:|
| Volta          |                70 |
| Turing         |                75 |
| Ampere         |                86 |
| Ada Lovelace   |                89 |
| Hopper         |                90 |

For example, for an RTX 30-series GPU, use:
```
set(CMAKE_CUDA_ARCHITECTURES "86")
```

After changing the architecture, rebuild from a clean build directory.
If you want to run the application with a GUI, start the container with the required X11 and GPU settings:

```
docker run --rm -it \
       --gpus all -e DISPLAY=$DISPLAY \
       -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v /tmp/.X11-unix:/tmp/.X11-unix \ 
       -v "$PWD/:/workspace/" newtonbiomorphx
```

---

## Contact

For questions or support, reach us at:
- ebrahim.jahanbakhsh [at] unige [dot] ch
- michel.milinkovitch [at] unige [dot] ch

---

## License

This project is distributed under the GNU General Public License v3.0 or later.

See the `LICENSE` file for details.
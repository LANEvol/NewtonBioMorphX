FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# -------------------------------------------------------
# Basic development tools (CLion expects these)
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gdb \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# OpenGL + X11 runtime libraries
# These allow GPU-accelerated OpenGL apps to run and display
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    mesa-utils \
    libgl1-mesa-glx \
    libglx-mesa0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# Gmsh library
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-pip \
    python-is-python3 \
    libglu1-mesa \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --force-reinstall --no-cache-dir gmsh==4.15.2
RUN ln /usr/local/lib/libgmsh.so.4.15 /usr/local/lib/libgmsh.so.4.15.2
RUN ln /usr/local/lib/libgmsh.so.4.15 /usr/local/lib/libgmsh.so
RUN ldconfig -v

# -------------------------------------------------------
# GMsh libraries
# These allow GPU-accelerated OpenGL apps to run and display
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    zenity \
    ffmpeg \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libgmp-dev \
    libmpfr-dev \
    libeigen3-dev \
    libsdl2-dev \
    libfreetype-dev \
    && rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------
# VTK librarary
# -------------------------------------------------------
# Skip interactive geographic area prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    libvtk9-dev \
    && rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------
# Workspace for CLion
# -------------------------------------------------------
WORKDIR /workspace

#ADD cmake-build-release-docker_gpu/NewtonBioMorphX ./
ADD examples ../examples
ADD share ../share
ADD src/nvrtc_kernels ../src/nvrtc_kernels
COPY src/Typedefs.h ../src/
COPY src/Primitives.h ../src/
COPY src/DeviceDataPtr.h ../src/
COPY src/SVD3Cuda.h ../src/
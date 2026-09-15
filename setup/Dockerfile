# HuRo unified environment as a container: the image carries the fully-built python stack in a
# venv at /opt/venv, no conda inside. Build needs no GPU: the CUDA extensions compile
# ahead-of-time.
# Neither the code nor the model weights are baked in. setup/docker_run.sh runs the image with
# the checkout's parent directory mounted at /workspace and the GPU, Vulkan and OptiX pieces
# stage 9 needs (setup/README.md).
#
#   docker build -f setup/Dockerfile -t huro .
#   ./setup/docker_run.sh
#
# First commands inside the container: ./setup/setup.sh dev && ./setup/setup.sh check
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System libs: build tooling, and the GL/Vulkan runtime Isaac Sim renders through.
# The NVIDIA driver userspace itself is injected at `docker run --gpus`, not baked here.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates unzip \
      build-essential software-properties-common \
      ffmpeg \
      libvulkan1 vulkan-tools \
      libosmesa6-dev libgl1 libegl1 libgles2 mesa-utils \
      libxt6 libglu1-mesa libxkbcommon0 && \
    rm -rf /var/lib/apt/lists/*

# Python 3.11 (not default on Ubuntu 22.04, but the version the wheel set is built for), in a
# venv so pip never collides with apt's dist-packages
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install -U pip
ENV PATH=/opt/venv/bin:$PATH

# NVIDIA EGL vendor manifest (the library it names arrives with the runtime injection)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# NVIDIA Vulkan ICD manifest, used when the host has none to inject. Under /usr/local
# because the container runtime bind-mounts /usr/share/vulkan/icd.d over anything there.
RUN mkdir -p /usr/local/share/vulkan/icd.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.242"}}\n' \
      > /usr/local/share/vulkan/icd.d/nvidia_icd.json

ENV CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1

# Build-time copy only. The image carries the environment, not the code. Mount a checkout
# to run (see setup/docker_run.sh). Its submodules must be fetched first.
WORKDIR /build/huro
COPY . .

# Compute capabilities the extensions are compiled for. Unset, setup/setup.sh's own default
# covers 7.5 through 12.0. Pass only the target GPU's capability to cut the build time
# (docker build -f setup/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=8.9 -t huro .). Declared as
# ARG, not ENV, so it applies to this build only and nothing leaks into the image.
ARG TORCH_CUDA_ARCH_LIST

# patches + python stack + CUDA extensions, from the same script the conda path uses.
# Everything installs non-editable, so nothing keeps a path into this build directory.
RUN ./setup/setup.sh pip && ./setup/setup.sh build && cd / && rm -rf /build

WORKDIR /workspace
CMD ["/bin/bash"]

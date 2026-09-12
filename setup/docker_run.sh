#!/bin/bash
# Attach to the HuRo container, creating it on first use.
#
# The container is named `huro` and mounts this checkout's parent directory at /workspace, so the
# repo is at /workspace/<checkout-name>. It keeps running after a detach (Ctrl-P Ctrl-Q) and
# after a reboot.
#
#   ./setup/docker_run.sh  # from the host
#   docker attach huro     # same thing
#   docker rm -f huro      # start over
#
# Stage 10 needs OMNI_KIT_ACCEPT_EULA=Y, which accepts NVIDIA's Omniverse licence. This
# script forwards the value from the calling shell rather than accepting the licence
# itself. The container bakes the value in at creation, so export it
# BEFORE the first run:  export OMNI_KIT_ACCEPT_EULA=Y && ./setup/docker_run.sh
# If the container already exists without it, `docker rm -f huro` and run this again.
#
# Do not mount anything at /usr/share/vulkan/icd.d. The container runtime bind-mounts the
# host's copy there, which may be empty. run_pipeline.sh picks the Vulkan manifest at run
# time, falling back to the one baked into the image, so VK_ICD_FILENAMES is not set here.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"

if ! docker container inspect huro >/dev/null 2>&1; then
  docker create -it --name huro --restart unless-stopped \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/dri \
    -v "$PARENT:/workspace" \
    -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
    -v /usr/share/vulkan/implicit_layer.d:/usr/share/vulkan/implicit_layer.d:ro \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e OMNI_KIT_ACCEPT_EULA \
    -e HF_HOME=/workspace/.hf_cache \
    -w "/workspace/$(basename "$ROOT")" \
    huro bash >/dev/null
fi

docker start huro >/dev/null
exec docker attach huro

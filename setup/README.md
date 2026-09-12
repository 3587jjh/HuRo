# Setup

The pipeline requires Linux, an NVIDIA GPU with at least 24 GB of VRAM and a driver supporting
CUDA 12.8, and git. Installation takes one of two paths, **A (conda)** or **B (docker)**,
both from this checkout:

```bash
git clone https://github.com/3587jjh/HuRo HuRo && cd HuRo
git submodule update --init --recursive
```

Either path takes several hours, most of it spent compiling the CUDA extensions. The default build
covers all eight architectures from 6.0 to 9.0, so the result runs on any GPU in that range.

`TORCH_CUDA_ARCH_LIST` narrows the build to one architecture, which shortens it markedly.
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` reports the compute capability of the
GPUs present, which is 8.9 for an RTX 4090 or L40S, 9.0 for an H100 and 8.0 for an A100.
Extensions built this way run on that architecture alone, so keep the default when one environment
or image has to work on machines with different GPUs.

## 📌 Robot overlay requirements

Every part of the pipeline except the robot overlay runs on any NVIDIA GPU the driver above
supports. Isaac Sim renders the overlay through the RTX raytracer, which requires **RT
cores**. The GPUs built without them cannot render it, among them the **V100** (compute
capability 7.0), **A100** (8.0) and **H100** (9.0). The overlay then renders as black frames, or
the render does not terminate, instead of failing with a clear error.

Any GeForce RTX or RTX-series professional GPU carries the cores, among them the RTX 4090, L40S,
A40 and RTX 6000 Ada. When no such GPU is available, `LAST_STAGE=9` in `run_pipeline.sh` stops the
run after the retargeting, and the overlay follows later on a machine that has one.

The overlay also requires `OMNI_KIT_ACCEPT_EULA=Y`. Setting that variable accepts NVIDIA's
Omniverse licence, which the install puts at `site-packages/isaacsim/LICENSE.txt`. Nothing in this
repository sets it, and the overlay does not run without it. Read the licence, then set it in the
environment the pipeline runs in:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

## A: conda

```bash
./setup/setup.sh          # env `huro` + CUDA extensions + import checks, several hours
./setup/download.sh       # model weights (below)
conda activate huro       # before every run
```

`TORCH_CUDA_ARCH_LIST=8.9 ./setup/setup.sh` builds for that architecture alone.

## B: docker

The image holds the environment and not the repository. `setup/docker_run.sh` mounts the
checkout's parent directory at `/workspace`, so the repository sits at `/workspace/HuRo`. Keep the
input videos under that parent directory, or add a mount for their directory to
`setup/docker_run.sh` before its first run.

```bash
docker build -f setup/Dockerfile -t huro .  # several hours: the extensions compile ahead of time
# export OMNI_KIT_ACCEPT_EULA=Y here (licence above), since the container fixes it at creation
./setup/docker_run.sh                       # creates the container `huro`, then attaches
```

`docker build -f setup/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=8.9 -t huro .` builds for that
architecture alone. It is a build argument and does not survive into the image, so
`setup/setup.sh dev` below recompiles detectron2 and sam2 for all eight architectures unless
`TORCH_CUDA_ARCH_LIST` is exported in the container too.

On first entry, in this order:

```bash
./setup/setup.sh dev       # patch the submodules in this checkout + editable re-install
./setup/setup.sh check     # verify every stage stack imports on this GPU
./setup/download.sh        # model weights (below)
```

`setup/docker_run.sh` also sets `HF_HOME` to `/workspace/.hf_cache`. On the host that is
`.hf_cache` beside the checkout, so the stage-7 VLM is downloaded once and outlives the container.

## Model weights

`setup/download.sh` fetches every checkpoint the stages need, and a re-run fetches only what is
missing. The one exception is **MANO**, which requires registration at
<https://mano.is.tue.mpg.de>. Download `mano_v*_*.zip` and place its two models at

```
submodules/hawor/_DATA/data/mano/MANO_RIGHT.pkl
submodules/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

The stage-7 Qwen3.5-9B VLM (~18 GB) is not among them. It downloads itself on first use into
`$HF_HOME`, which defaults to `~/.cache/huggingface`. Run `hf download Qwen/Qwen3.5-9B` to
fetch it in advance.

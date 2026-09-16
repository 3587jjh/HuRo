# Setup

The pipeline requires Linux, an NVIDIA GPU with at least 24 GB of VRAM and a driver supporting
CUDA 12.8, and git. Installation takes one of two paths, **A (conda)** or **B (docker)**,
both from this checkout:

```bash
git clone https://github.com/3587jjh/HuRo HuRo && cd HuRo
git submodule update --init --recursive
```

Either path takes several hours, most of it spent compiling the CUDA extensions. The default build
covers seven GPU architectures from compute capability 7.5 to 12.0.

`TORCH_CUDA_ARCH_LIST` narrows the build to one architecture, which shortens it markedly.
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` reports the compute capability of the
GPUs present, which is 8.9 for an RTX 4090 or L40S, 9.0 for an H100 and 8.0 for an A100.
Extensions built this way run on that architecture alone, so keep the default when one environment
or image has to work on machines with different GPUs.

## 📌 Robot overlay requirements

Isaac Sim renders the robot overlay through the RTX raytracer, which requires **RT cores**.
The GPUs built without them cannot render it, among them the **A100**
(compute capability 8.0) and **H100** (9.0).

Any GeForce RTX or RTX-series professional GPU carries the cores, among them the RTX 4090, L40S,
A40 and RTX 6000 Ada. Without one, set `LAST_STAGE=8` in `run_pipeline.sh` and run the
overlay on another machine.

NVIDIA tested Isaac Sim 5.1 on driver 580.65.06, and a branch newer than R580 breaks the
overlay. On 595 or 610 the RTX renderer crashes at startup and the stage exits with a
segmentation fault.

After changing the driver, delete `/tmp/huro_ov_cache_*` and rerun `./setup/setup.sh check`.
The overlay caches its compiled shaders there, and they were built for the old driver.

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
architecture alone. The build argument does not survive into the image. Export the same
`TORCH_CUDA_ARCH_LIST` in the container before `setup/setup.sh dev` below, or that step recompiles
detectron2 and sam2 for all seven architectures.

On first entry, in this order:

```bash
./setup/setup.sh dev       # patch the submodules in this checkout + editable re-install
./setup/setup.sh check     # verify every stage stack imports on this GPU
./setup/download.sh        # model weights (below)
```

`setup/docker_run.sh` also sets `HF_HOME` to `/workspace/.hf_cache`. On the host that is
`.hf_cache` beside the checkout, so the stage-6 VLM (below) is downloaded
once and outlives the container.

## Model weights

`setup/download.sh` fetches every checkpoint the stages need except two, and a re-run fetches
only what is missing. The first exception is **MANO**, which requires registration
at <https://mano.is.tue.mpg.de>. Download `mano_v*_*.zip`
and place its two models at these paths:

```
submodules/hawor/_DATA/data/mano/MANO_RIGHT.pkl
submodules/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

The second exception is the stage-6 Qwen3.5-9B VLM (~18 GB). It downloads itself on first use
into `$HF_HOME`, which defaults to `~/.cache/huggingface`. Run `hf download Qwen/Qwen3.5-9B`
to fetch it in advance.

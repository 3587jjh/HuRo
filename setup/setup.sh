#!/bin/bash
# Build the `huro` environment (all pipeline stages) from a fresh checkout.
#
#   ./setup/setup.sh        # everything: conda env + CUDA extensions + checks (internet + GPU)
#   ./setup/setup.sh env    # conda env + python stack + source clones (internet, no GPU needed)
#   ./setup/setup.sh pip    # the python stack only, without creating a conda env.
#                           #   The Dockerfile runs it in /opt/venv
#   ./setup/setup.sh build  # CUDA extensions (nvcc, builds fine without a GPU)
#   ./setup/setup.sh check  # import checks for every stage stack (run where there is a GPU)
#   ./setup/setup.sh dev    # prepare THIS checkout against an already-built environment: apply the
#                           #   submodule patches + editable re-install (run once after mounting)
#
# Run `env` (or `pip`) where there is internet, `build` + `check` where there is a GPU.
# Environment name comes from $HURO_ENV (default: huro). `pip`/`build`/`check` fall back to
# the current interpreter when that conda env (or conda itself) is absent.
#
# Model weights are NOT downloaded here. `setup/download.sh` fetches them, and the MANO
# models additionally require manual registration.
#
# No `set -u`: conda's cuda-toolkit (de)activation scripts reference unset variables.
set -eo pipefail

ENV_NAME="${HURO_ENV:-huro}"
STEP="${1:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -d pipeline ] && [ -d common ] || { echo "pipeline/ and common/ not found in $ROOT"; exit 1; }

activate_env() {
  # conda layout: activate $ENV_NAME if it exists. Otherwise (e.g. inside the container,
  # where the /opt/venv interpreter IS the environment) there is nothing to activate.
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      conda activate "$ENV_NAME"
    fi
  fi
}

apply_patches() {
  echo "== patches =="
  local p
  declare -A DIRS=(
    [droidcalib]=submodules/droidcalib
    [droidcalib-lietorch]=submodules/droidcalib/thirdparty/lietorch
    [anycalib]=submodules/anycalib
    [100doh]=submodules/100doh
    [hawor]=submodules/hawor
    [geocalib]=submodules/geocalib
    [detectron2]=submodules/detectron2
    [sam2]=submodules/sam2
    [propainter]=submodules/propainter
    [allex_model]=submodules/allex_model
  )
  for p in "${!DIRS[@]}"; do
    local dir="${DIRS[$p]}" patch="$ROOT/submodules_patches/$p.patch"
    [ -e "$dir/.git" ] || { echo "missing submodule $dir. Run: git submodule update --init --recursive"; exit 1; }
    if ( cd "$dir" && git apply --check "$patch" 2>/dev/null ); then
      ( cd "$dir" && git apply "$patch" ) && echo "  applied $p.patch"
    elif ( cd "$dir" && git apply --check --reverse "$patch" 2>/dev/null ); then
      echo "  $p.patch already applied"
    else
      echo "  ERROR: $p.patch neither applies nor is applied. Inspect $dir"; exit 1
    fi
  done
}

pip_stack() {
  # droidcalib is built via `python setup.py install` (two setup() calls) -> needs setuptools<82
  pip install "setuptools<82" wheel

  # torch stack: isaacsim-core 5.1.0 requires exactly torch 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

  # Isaac Sim 5.1.0 (stage 10) — several GB of kit + extension-cache wheels
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

  # JAX CUDA build (stage 9). Its nvidia-* floors are all >=, satisfied by torch's cu128 pin set
  pip install "jax[cuda12]==0.4.30"

  # stage-7 narration stack (Qwen3.5 support is only in a transformers dev commit)
  pip install "transformers @ git+https://github.com/huggingface/transformers.git@a91232af09f59e2e1c96561901c92e01e238c355"
  pip install qwen-vl-utils==0.0.14 accelerate==1.13.0 lemminflect==0.2.3
  # prebuilt wheel for torch 2.7/cu12/cp311, with --no-cache-dir for network-filesystem pip caches
  pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

  # stages 1-6, 8 pip deps
  pip install kornia scikit-learn
  pip install evo --upgrade --no-binary evo
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  pip install easydict "ultralytics==8.4.32" "lap==0.5.13"
  pip install "smplx==0.1.28" "einops==0.8.2" "yacs==0.1.8" "timm==1.0.26" \
    "pytorch-lightning==2.6.1" "lightning-utilities==0.15.3" "pyrootutils==1.0.4" "loguru==0.7.3" \
    "mmengine==0.10.4" "pytorch-minimize==0.1.0" "plyfile==1.1.3" "HTML4Vision==0.6.1" \
    "rich==14.3.3" "hydra-core==1.3.2" "scikit-image==0.25.2" fvcore iopath
  pip install --no-build-isolation "chumpy@git+https://github.com/mattloper/chumpy"
  pip install "utils3d@git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
  pip install "filelock>=3.13"
  pip install cloudpickle tensorboard black pycocotools

  # anycalib (pure python)
  ( cd submodules/anycalib && pip install . --no-build-isolation )

  # stage-9 IK stack: pyroki/jaxls dependencies first, then the two --no-deps.
  # jaxls@50a58be declares jax>=0.6.0 but runs on 0.4.30. --no-deps keeps jax at 0.4.30.
  pip install "jax-dataclasses>=1.6.2" "jaxlie>=1.0.0" jaxtyping termcolor "typing-extensions>=4.5" \
    tyro robot_descriptions yourdfpy trimesh pyliblzfse
  pip install --no-deps "jaxls @ git+https://github.com/brentyi/jaxls.git@50a58be88c5ef74532f09e3f55268b4f02c490e3"
  pip install --no-deps submodules/pyroki

  # viser: last series accepting websockets 12.0 (isaacsim-kernel pins ==12.0)
  pip install "viser==0.2.11"

  # shared I/O
  pip install av pyarrow pandas pyyaml tqdm

  # pytorch3d source, cloned now so the `build` step needs no network
  [ -d build/pytorch3d_src ] || git clone --branch V0.7.8 --depth 1 https://github.com/facebookresearch/pytorch3d build/pytorch3d_src

  # FINAL PINS, order matters: isaacsim's exact pins, then the omnidir-capable opencv build.
  # Several installs above drag plain opencv in, so the uninstall/reinstall must come last.
  pip install "websockets==12.0" "scipy==1.15.3"
  pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
  pip install --no-cache-dir "numpy==1.26.0" "opencv-contrib-python-headless==4.7.0.72"
}

step_env() {
  command -v conda >/dev/null || { echo "conda not found (containers use './setup/setup.sh pip' instead)"; exit 1; }
  apply_patches
  echo "== conda env: $ENV_NAME =="
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # Every conda call here overrides the channel list. Anaconda's own `defaults` channels
  # refuse to install until their Terms of Service are accepted interactively, which halts
  # this script on a fresh conda. Nothing needed here comes from them anyway.
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
    || conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.11 -y
  conda activate "$ENV_NAME"
  [ -x "$CONDA_PREFIX/bin/ffmpeg" ] || conda install --override-channels -c conda-forge ffmpeg -y
  # nvcc 12.8 for the CUDA-extension builds (self-contained, node-independent, matching cu128 torch)
  [ -x "$CONDA_PREFIX/bin/nvcc" ] || conda install --override-channels \
    -c "nvidia/label/cuda-12.8.1" -c conda-forge cuda-toolkit -y
  pip_stack
  echo "== env step done =="
}

step_pip() {
  apply_patches
  activate_env
  echo "== python stack onto $(command -v python) =="
  pip_stack
  echo "== pip step done =="
}

step_build() {
  activate_env
  echo "== CUDA extensions (nvcc $(nvcc --version | grep -oE 'release [0-9.]+')) =="
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0}"
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-8}"

  # droidcalib CUDA extensions (droid_backends + lietorch_backends)
  ( cd submodules/droidcalib && python setup.py install )

  # Detectron2 + SAM2 CUDA ops (FORCE_CUDA=1, and SAM2 builds its _C by default)
  ( cd submodules/detectron2 && MAX_JOBS=6 pip install . --no-build-isolation --no-deps )
  ( cd submodules/sam2       && MAX_JOBS=6 pip install . --no-build-isolation --no-deps )

  # pytorch3d V0.7.8 from the clone made in the env/pip step (template-heavy, tens of minutes)
  pip install ./build/pytorch3d_src --no-build-isolation --no-deps

  # 100DoH CUDA ops. Its patched setup.py honors FORCE_CUDA, so this also builds without a GPU
  ( cd submodules/100doh/lib && pip install . --no-build-isolation --no-deps )

  echo "== build step done =="
}

# Prepare a mounted/fresh checkout to run against an already-built environment (the container):
# apply the submodule patches here, then re-install the four path-bound backends from THIS
# checkout so edits to submodules/{pyroki,anycalib,detectron2,sam2} apply without a reinstall.
step_dev() {
  apply_patches
  activate_env
  export FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-6}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0}"
  echo "== editable re-install from $ROOT =="
  pip install -e submodules/pyroki --no-deps
  ( cd submodules/anycalib   && pip install -e . --no-build-isolation --no-deps )
  ( cd submodules/detectron2 && pip install -e . --no-build-isolation --no-deps )
  ( cd submodules/sam2       && pip install -e . --no-build-isolation --no-deps )
  echo "== dev step done =="
}

step_check() {
  activate_env
  echo "== import checks =="
  python -c "import torch; from common.paths import setup_submodule_imports; setup_submodule_imports(); \
import droid_backends, kornia, cv2, model._C, pytorch3d, pytorch3d._C; from droid import Droid; \
from anycalib import AnyCalib; from model.faster_rcnn.resnet import resnet; \
from ultralytics.trackers.bot_sort import BOTSORT; import lap, smplx, chumpy; \
print('  human core OK | torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| omnidir', hasattr(cv2, 'omnidir'))"
  python -c "import sys; sys.path.insert(0,'submodules/hawor'); import torch; \
from scripts.scripts_test_video.hawor_video import load_hawor, estimate_3d_hand_keypoints; \
from hawor.utils.process import get_mano_faces_closed; from lib.vis.renderer import Renderer; print('  hawor OK')"
  python -c "import torch; from common.paths import setup_extrinsics_imports; setup_extrinsics_imports(); \
import droid_backends; from droid import Droid; from moge.model.v2 import MoGeModel; \
from lib.pipeline.est_scale import est_scale_hybrid; from geocalib import GeoCalib; print('  stage5 OK')"
  python -c "import torch, detectron2, sam2; import detectron2._C, sam2._C; \
from pipeline.segmentation.detectors import DetectorDetectron2, DetectorSam2; \
from detectron2.config import LazyConfig; from common.paths import DETECTRON_CFG; \
LazyConfig.load(str(DETECTRON_CFG)); print('  stage6 OK')"
  python -c "import torch, transformers, qwen_vl_utils, lemminflect, flash_attn; \
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor; \
from pipeline.captioning.caption import load_vlm_model, get_caption; \
from pipeline.captioning.tracker import RejectionTracker; print('  caption OK')"
  python -c "from common.robot_config import load_robot_config; \
from pipeline.retargeting.retargeter import Retargeter; \
r = Retargeter(load_robot_config('allex')); \
print('  robot OK:', len(r.robot.joints.actuated_names), 'actuated joints')"
  # Isaac Sim renders through the RTX raytracer, which needs RT cores. The datacenter parts
  # built without them give black frames or never finish rather than failing cleanly, so warn
  # here, because this is the step that runs where the GPU actually is.
  CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
  NO_RT=$(echo "$CAPS" | grep -xE '7\.0|8\.0|9\.0' | paste -sd' ' || true)
  if [ -n "$NO_RT" ]; then
    echo
    echo "  ##########################################################################"
    echo "  ##  WARNING: this GPU has no RT cores (compute capability $NO_RT)"
    echo "  ##"
    echo "  ##  Stage 10 renders through Isaac Sim's RTX raytracer and cannot run"
    echo "  ##  here: V100 (7.0), A100 (8.0) and H100 (9.0) produce black frames or"
    echo "  ##  never finish, rather than failing cleanly."
    echo "  ##"
    echo "  ##  Stages 1-9 are unaffected. Set LAST_STAGE=9 in"
    echo "  ##  run_pipeline.sh, or run stage 10 on an RTX card (L40S, RTX 4090,"
    echo "  ##  A40, RTX 6000 Ada)."
    echo "  ##########################################################################"
    echo
  fi

  # importing stage 10 runs its Vulkan/driver preflight before Isaac Sim. Isaac Sim is
  # proprietary: setting OMNI_KIT_ACCEPT_EULA=Y accepts NVIDIA's licence, so this script
  # does not set it. Without it the overlay probe is skipped, not auto-accepted.
  if [ "${OMNI_KIT_ACCEPT_EULA:-}" = "Y" ]; then
    python -c "import pipeline.stage10_robot_overlay; \
import isaacsim, torch, numpy, cv2; \
assert hasattr(cv2, 'omnidir'); print('  overlay OK', torch.__version__, numpy.__version__)"
  else
    echo "  overlay SKIPPED: stage 10 runs on Isaac Sim under the NVIDIA Omniverse License"
    echo "    Agreement (site-packages/isaacsim/LICENSE.txt)."
    echo "    Read it, then re-run with OMNI_KIT_ACCEPT_EULA=Y to verify the overlay stack."
    SKIPPED_OVERLAY=1
  fi
  if [ -n "$NO_RT" ]; then
    echo "== checks passed, but this GPU has no RT cores: stage 10 cannot run here (above) =="
  elif [ -n "$SKIPPED_OVERLAY" ]; then
    echo "== checks passed, except the overlay: stage 10 is NOT verified =="
  else
    echo "== all checks passed =="
  fi
  echo "next: download the model weights, then run the pipeline stages"
}

case "$STEP" in
  env)   step_env ;;
  pip)   step_pip ;;
  build) step_build ;;
  check) step_check ;;
  dev)   step_dev ;;
  all)   step_env; step_build; step_check ;;
  *) echo "usage: $0 [env|pip|build|check|dev|all]"; exit 1 ;;
esac

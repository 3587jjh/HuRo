#!/bin/bash
# Run the whole pipeline over one clip directory: stage 1 -> 10, in order.
#
#   ./run_pipeline.sh
#
# No command-line arguments. Edit the settings block below. Every stage is resumable and
# skips the clips it already finished, so a re-run picks up where it stopped.
#
# Each GPU in GPUS gets a worker over its own slice of the clips. Stage 10 needs every clip
# at once, so the workers join after stage 9 and one process then converts them all.
#
# Outputs land beside INPUT_DIR (all git-ignored):
#   <clips>_intr  _contact  _contact_refined  _hand  _extr  _chunked  _lerobot
#
# Conda users: `conda activate huro` first. Inside the docker container there is nothing
# to activate. To keep a transcript: ./run_pipeline.sh 2>&1 | tee run.log
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[ -d pipeline ] && [ -d common ] || { echo "pipeline/ and common/ not found beside this script"; exit 1; }


# ─── settings ────────────────────────────────────────────────────────────────

INPUT_DIR="examples/clips"   # directory of 30 fps .mp4 clips, untrimmed is fine
GPUS=(0)                     # one worker per GPU listed. The clips are split --part i/N across
                             # them, e.g. GPUS=(0 1) -> 2 workers, half the clips each
ROBOT="allex"                # robot config for stages 8-10 (configs/<ROBOT>.yaml)
FIRST_STAGE=1                # run stages FIRST_STAGE..LAST_STAGE
LAST_STAGE=10
PROGRESS=0                   # 1 keeps the tqdm bars, and 0 passes --no_tqdm (log-friendly)

# Optional per-stage flag overrides, keyed by stage number.
declare -A STAGE_ARGS=(
  # [6]="--stride 5 --n_caption_samples 4"
)


# ─── runner ──────────────────────────────────────────────────────────────────

SCRIPTS=(
  [1]=stage1_annot_intrinsics.py       [2]=stage2_annot_contact.py
  [3]=stage3_annot_contact_refine.py   [4]=stage4_annot_hand.py
  [5]=stage5_annot_extrinsics.py       [6]=stage6_annot_narr.py
  [7]=stage7_annot_inpaint.py          [8]=stage8_robot_retarget.py
  [9]=stage9_robot_overlay.py          [10]=stage10_lerobot_convert.py
)
SHARDED_LAST_STAGE=9   # the last stage that may run on more than one worker

# Outputs go beside INPUT_DIR (<clips>_chunked), so drop any trailing slash.
while [[ "$INPUT_DIR" == ?*/ ]]; do INPUT_DIR="${INPUT_DIR%/}"; done
# Only stages 1-6 read the source videos. From stage 7 on, the work comes out of <clips>_chunked,
# and INPUT_DIR itself need not exist.
if [ "$FIRST_STAGE" -le 6 ]; then
  [ -d "$INPUT_DIR" ] || { echo "INPUT_DIR is not a directory: $INPUT_DIR"; exit 1; }
  compgen -G "$INPUT_DIR/*.mp4" >/dev/null || { echo "no .mp4 clips in $INPUT_DIR"; exit 1; }
elif [ ! -d "${INPUT_DIR}_chunked" ]; then
  echo "stages $FIRST_STAGE+ need ${INPUT_DIR}_chunked, which is not there"; exit 1
fi
[ "${#GPUS[@]}" -gt 0 ] || { echo "GPUS is empty"; exit 1; }
[ "$FIRST_STAGE" -ge 1 ] && [ "$LAST_STAGE" -le 10 ] && [ "$FIRST_STAGE" -le "$LAST_STAGE" ] \
  || { echo "bad stage range: $FIRST_STAGE..$LAST_STAGE"; exit 1; }

# Stage 9 renders through Isaac Sim, which is proprietary: setting OMNI_KIT_ACCEPT_EULA=Y
# accepts the NVIDIA Omniverse License Agreement, so this script does not set it.
# Checked here so a run that includes stage 9 fails now rather than after stages 1-8.
if [ "$FIRST_STAGE" -le 9 ] && [ "$LAST_STAGE" -ge 9 ] \
   && [[ ! "${OMNI_KIT_ACCEPT_EULA,,}" =~ ^(y|yes|1)$ ]]; then
  echo "stage 9 renders through Isaac Sim, licensed under the NVIDIA Omniverse License Agreement"
  echo "(site-packages/isaacsim/LICENSE.txt). Read it, then accept with"
  echo
  echo "    export OMNI_KIT_ACCEPT_EULA=Y"
  echo
  echo "and re-run. To stop before the overlay instead, set LAST_STAGE=8."
  exit 1
fi

# Stage 9's RTX renderer crashes on a driver branch newer than R580 (setup/README.md).
# Checked here so a run that includes stage 9 fails now rather than after stages 1-8.
if [ "$FIRST_STAGE" -le 9 ] && [ "$LAST_STAGE" -ge 9 ] && [ -z "$HURO_SKIP_DRIVER_CHECK" ]; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  ISAAC=$(python -c "import importlib.metadata as m; print(m.version('isaacsim'))" 2>/dev/null)
  DRIVER_MAJOR="${DRIVER%%.*}"
  if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && [ "$DRIVER_MAJOR" -ge 590 ] \
     && [ "${ISAAC%%.*}" = 5 ]; then
    echo "stage 9 renders through Isaac Sim $ISAAC, which does not work with NVIDIA"
    echo "driver $DRIVER. Its RTX renderer crashes during startup. Use a driver no newer"
    echo "than R580, or set LAST_STAGE=8 and render the overlay on another host."
    echo
    echo "Set HURO_SKIP_DRIVER_CHECK=1 to run anyway."
    exit 1
  fi
fi

if [ "$PROGRESS" = 1 ]; then TQDM=(); else TQDM=(--no_tqdm); fi

vulkan_icd_env() {
  # Isaac Sim needs VK_ICD_FILENAMES to name a manifest that exists, or it renders black
  # frames. Where the NVIDIA driver puts one varies by host, so fall back through the known
  # locations. Pin exactly one: two manifests naming the same library list every GPU twice.
  local named="${VK_DRIVER_FILES:-${VK_ICD_FILENAMES:-}}" candidate
  if [ -n "$named" ] && [ -f "$named" ]; then return 0; fi

  for candidate in /usr/share/vulkan/icd.d/nvidia_icd.json \
                   /etc/vulkan/icd.d/nvidia_icd.json \
                   /usr/local/share/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$candidate" ]; then
      echo "   vulkan: ${named:-VK_ICD_FILENAMES} missing, using $candidate"
      export VK_ICD_FILENAMES="$candidate"
      unset VK_DRIVER_FILES
      return 0
    fi
  done

  echo "   vulkan: no NVIDIA ICD manifest found, so stage 9 will report what is missing"
}

isaac_env() {
  # Isaac Sim: keep the shader caches on node-local disk, per GPU. OMNI_KIT_ACCEPT_EULA is
  # deliberately NOT set here, because setting it accepts NVIDIA's licence. The check above
  # refuses to start stage 9 until it is set.
  export HOME="/tmp/huro_ov_cache_$(id -un)_gpu${CUDA_VISIBLE_DEVICES}"
  mkdir -p "$HOME"
  export __GL_SHADER_DISK_CACHE=1 __GL_SHADER_DISK_CACHE_PATH="$HOME/nvgl" CUDA_CACHE_PATH="$HOME/cuda"
  vulkan_icd_env
}

run_stage() {  # run_stage <stage-number> <part>
  local n=$1 part=$2
  # STAGE_ARGS unquoted on purpose: an override string splits into separate flags
  local args=(--input_dir "$INPUT_DIR" --part "$part" "${TQDM[@]}" ${STAGE_ARGS[$n]})
  case $n in 8|9|10) args+=(--robot_name "$ROBOT") ;; esac

  echo ""
  echo "== stage $n: ${SCRIPTS[$n]}  (part $part, $(date '+%F %T')) =="
  # subshell: stage 9's Isaac Sim environment must not leak into later stages
  (
    if [ "$n" = 9 ]; then isaac_env; fi
    python "pipeline/${SCRIPTS[$n]}" "${args[@]}"
  )
}

run_worker() {  # run_worker <part> <first-stage> <last-stage>
  local part=$1 first=$2 last=$3 n
  for n in $(seq "$first" "$last"); do run_stage "$n" "$part"; done
}

N=${#GPUS[@]}
LAST_SHARDED=$LAST_STAGE
if [ "$LAST_SHARDED" -gt "$SHARDED_LAST_STAGE" ]; then LAST_SHARDED=$SHARDED_LAST_STAGE; fi

echo "== $INPUT_DIR | gpus ${GPUS[*]} | robot $ROBOT | stages $FIRST_STAGE..$LAST_STAGE =="

# Stage 6's VLM (~18 GB) downloads on first use. Fetch it once rather than once per worker.
if [ "$N" -gt 1 ] && [ "$FIRST_STAGE" -le 6 ] && [ "$LAST_STAGE" -ge 6 ]; then
  echo ""
  echo "== stage 6 VLM: warming the HF cache before fanning out =="
  python -c "from pipeline.captioning.caption import MODEL_ID, VLM_CACHE_DIR; \
from huggingface_hub import snapshot_download; \
print(' ', snapshot_download(repo_id=MODEL_ID, cache_dir=VLM_CACHE_DIR))"
fi

# Stages 1-9: one worker per GPU, each over its own slice of the clips.
if [ "$FIRST_STAGE" -le "$SHARDED_LAST_STAGE" ]; then
  if [ "$N" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
    run_worker "1/1" "$FIRST_STAGE" "$LAST_SHARDED"
  else
    PIDS=()
    for ((i = 0; i < N; i++)); do
      (
        export CUDA_VISIBLE_DEVICES="${GPUS[i]}"
        run_worker "$((i + 1))/$N" "$FIRST_STAGE" "$LAST_SHARDED"
      ) 2>&1 | sed -u "s/^/[gpu ${GPUS[i]}] /" &
      PIDS+=($!)
    done
    FAILED=0
    for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done   # the barrier
    if [ "$FAILED" -ne 0 ]; then
      echo ""
      if [ "$LAST_STAGE" -ge 10 ]; then echo "== a worker failed, not starting stage 10 =="; else echo "== a worker failed =="; fi
      exit 1
    fi
  fi
fi

# Stage 10 sees every clip at once, so it runs once, after every worker has finished.
if [ "$LAST_STAGE" -ge 10 ]; then
  export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
  run_stage 10 "1/1"
fi

echo ""
echo "== done: stages $FIRST_STAGE..$LAST_STAGE over $INPUT_DIR (gpus ${GPUS[*]}) =="

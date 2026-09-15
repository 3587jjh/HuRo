#!/bin/bash
# Download every model weight the pipeline needs into the submodule layout the stages read
# from. Idempotent (existing non-empty files are skipped), verifies each fresh download
# against a recorded md5, and keeps going on failure. A summary of anything that could not
# be fetched (dead link, moved file) is printed at the end with the path to place the file
# at manually. Needs internet, no python environment required.
#
# One item cannot be automated: the MANO hand models require registration at
# https://mano.is.tue.mpg.de (license). Download mano_v*_*.zip manually and place:
#   submodules/hawor/_DATA/data/mano/MANO_RIGHT.pkl
#   submodules/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl
set -o pipefail
cd "$(dirname "$0")/.."
[ -d submodules/hawor ] || { echo "submodules/hawor not found. Run: git submodule update --init --recursive"; exit 1; }

FAILED=()

fetch() {  # fetch <dest> <md5|-> <url> [fallback command...]
  local dest=$1 md5=$2 url=$3; shift 3
  if [ -s "$dest" ]; then echo "  have  $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  echo "  fetch $dest"
  rm -f "$dest"  # clear 0-byte leftovers from interrupted runs
  if ! wget -q --show-progress -O "$dest.part" "$url"; then
    rm -f "$dest.part"
    if [ $# -gt 0 ] && "$@"; then :; else
      rm -f "$dest"
      FAILED+=("$dest  <-  $url")
      echo "  FAILED $dest"
      return
    fi
  else
    mv "$dest.part" "$dest"
  fi
  if [ "$md5" != "-" ]; then
    local got; got=$(md5sum "$dest" | awk '{print $1}')
    [ "$got" = "$md5" ] || echo "  WARNING: $dest md5 $got != expected $md5. The upstream file may have changed, so verify it before use"
  fi
}

# 100DoH fallback: the plain release-asset URL tried first works while this repository is
# public. This fallback reaches the same asset through an authenticated gh, which is what a
# private fork needs.
doh_gh() {
  command -v gh >/dev/null && gh auth status >/dev/null 2>&1 || { echo "  (no authenticated gh. Run: gh auth login)"; return 1; }
  gh release download assets -p faster_rcnn_1_8_132028.pth -D submodules/100doh
}

echo "== stage 1: AnyCalib (droidcalib.pth is committed in its submodule and needs no fetch) =="
fetch submodules/anycalib/anycalib_gen.pt 4a030293d48b298460748491b991c4b2 \
  https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_gen.pt

# The upstream Google Drive link for this checkpoint is dead, so what follows is an archival
# mirror on this repo's release assets.
echo "== stage 2: 100DoH detector (handobj_100K+ego, archival mirror) =="
fetch submodules/100doh/faster_rcnn_1_8_132028.pth f0b494c2d38f8773cef15f2573dced45 \
  https://github.com/3587jjh/HuRo/releases/download/assets/faster_rcnn_1_8_132028.pth \
  doh_gh

echo "== stages 4-5: HAWOR + DROID-SLAM weights + MANO mean params =="
fetch submodules/hawor/weights/hawor/checkpoints/hawor.ckpt da4227007b0a8c4f5098cef603659bd1 \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt
fetch submodules/hawor/weights/hawor/model_config.yaml a81be05b37366d4fc23b83bc7543f568 \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml
fetch submodules/hawor/weights/external/droid.pth 4d00031278f6cac82b0d6a00beab1f30 \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/external/droid.pth
# The hawor README omits this file, but its model config requires it. The HaMeR demo space
# ships the identical file.
fetch submodules/hawor/_DATA/data/mano_mean_params.npz 1adc23a4d1ea9b8956d9d3b74da22d99 \
  https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/_DATA/data/mano_mean_params.npz

echo "== stage 5: GeoCalib + MoGe-2 =="
fetch submodules/geocalib/checkpoints/pinhole.tar ca068f68f7f62c7fc1b64c33a8883633 \
  https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar
fetch submodules/moge-2/checkpoints/moge-2-vitl-normal/model.pt 4217c2b75880ae93aac615845a9101af \
  https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt

echo "== stages 6-7: Detectron2 ViTDet-H + SAM 2 =="
fetch submodules/detectron2/checkpoints/model_final_f05665.pkl f05665c9b95a34ceeba32316800c5e11 \
  https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
fetch submodules/sam2/checkpoints/sam2_hiera_large.pt 08083462423be3260cd6a5eef94dc01c \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

echo "== stage 7: ProPainter =="
fetch submodules/propainter/weights/raft-things.pth 55b58de5d9022eb37893916d246e14a3 \
  https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
fetch submodules/propainter/weights/recurrent_flow_completion.pth 2879dbdd08fa50c656ff3ff1659dd660 \
  https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
fetch submodules/propainter/weights/ProPainter.pth 83e3941395917f6c1943dcf2f7655454 \
  https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth

echo "== done =="
[ -s submodules/hawor/_DATA/data/mano/MANO_RIGHT.pkl ] && [ -s submodules/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl ] \
  || echo "REMINDER: the MANO models are still missing (registration required, see the header of this script)"
echo "The stage-6 Qwen3.5-9B VLM (~18 GB) downloads itself into \$HF_HOME on first run. To pre-fetch: hf download Qwen/Qwen3.5-9B"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "${#FAILED[@]} download(s) FAILED. The upstream link may have moved or died:"
  for f in "${FAILED[@]}"; do echo "  $f"; done
  echo "Obtain each file another way (search by filename, and see this script for the expected md5)"
  echo "and place it at the path shown on the left."
  exit 1
fi

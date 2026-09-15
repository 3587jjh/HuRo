"""Repo-root-anchored paths + the shared `--part a/b` partition helper.
Submodule locations derive from the repo root, so stage scripts run from any cwd."""
import os
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULES = REPO_ROOT / "submodules"

DROIDCALIB_DIR = SUBMODULES / "droidcalib"
ANYCALIB_DIR = SUBMODULES / "anycalib"
DROIDCALIB_WEIGHTS = DROIDCALIB_DIR / "droidcalib.pth"
ANYCALIB_WEIGHTS = ANYCALIB_DIR / "anycalib_gen.pt"

# Hand detection (stage 2): 100DoH Faster R-CNN.
HOD_DIR = SUBMODULES / "100doh"
CONTACT_CFG = HOD_DIR / "cfgs" / "res101.yml"
CONTACT_WEIGHTS = HOD_DIR / "faster_rcnn_1_8_132028.pth"

# Hand-mesh reconstruction (stage 4): HAWOR.
HAWOR_DIR = SUBMODULES / "hawor"
HAWOR_CKPT = HAWOR_DIR / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"

# Camera extrinsics (stage 5): DROID-SLAM + MoGe-2 + GeoCalib.
HAWOR_DROIDSLAM_DIR = HAWOR_DIR / "thirdparty" / "DROID-SLAM"
DROID_WEIGHTS = HAWOR_DIR / "weights" / "external" / "droid.pth"

MOGE2_DIR = SUBMODULES / "moge-2"
MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"            # HF hub id (fallback download)
MOGE2_WEIGHTS_DIR = MOGE2_DIR / "checkpoints" / "moge-2-vitl-normal"
MOGE2_WEIGHTS = MOGE2_WEIGHTS_DIR / "model.pt"

GEOCALIB_DIR = SUBMODULES / "geocalib"
GEOCALIB_WEIGHTS = GEOCALIB_DIR / "checkpoints" / "pinhole.tar"

# Arm segmentation (stage 7): detectron2 person detection + SAM2 video segmentation.
DETECTRON2_DIR = SUBMODULES / "detectron2"
DETECTRON_CFG = DETECTRON2_DIR / "projects" / "ViTDet" / "configs" / "COCO" / "cascade_mask_rcnn_vitdet_h_75ep.py"
DETECTRON_WEIGHTS = DETECTRON2_DIR / "checkpoints" / "model_final_f05665.pkl"

SAM2_DIR = SUBMODULES / "sam2"
SAM2_CONFIG_DIR = SAM2_DIR / "sam2" / "configs" / "sam2"
SAM2_MODEL_CFG = "sam2_hiera_l.yaml"
SAM2_WEIGHTS = SAM2_DIR / "checkpoints" / "sam2_hiera_large.pt"

# Arm inpainting (stage 7): ProPainter.
PROPAINTER_DIR = SUBMODULES / "propainter"
PROPAINTER_WEIGHTS_DIR = PROPAINTER_DIR / "weights"

# Robot retargeting (stage 8): PyRoKi over the robot's URDF. Each robot's config gives the
# URDF path as `robot_urdf_path` (repo-root relative). See robot_urdf().
PYROKI_DIR = SUBMODULES / "pyroki"

# Overlay rendering (stage 9): the URDF is imported once into an Isaac-ready USD, cached
# here per robot and regenerated whenever the URDF is newer.
OVERLAY_USD_DIR = REPO_ROOT / "build" / "overlay"

# Per-robot YAML configs (joint groups -> state layout, keypoint mapping, home pose).
ROBOT_CONFIG_DIR = REPO_ROOT / "configs"


def robot_urdf(rel_path) -> Path:
    """Absolute path for a config's repo-root-relative `robot_urdf_path`."""
    return REPO_ROOT / rel_path


def overlay_usd(robot_name: str) -> Path:
    """Where the Isaac-ready USD imported from `robot_name`'s URDF is cached."""
    return OVERLAY_USD_DIR / f"{robot_name}.usd"


def setup_hawor_imports():
    """Put the hawor submodule on sys.path (top-level `lib`/`hawor`/`scripts`/`infiller` imports)."""
    sp = str(HAWOR_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def setup_submodule_imports():
    """Put droidcalib + anycalib on sys.path (anchored to the repo root)."""
    for p in (DROIDCALIB_DIR, DROIDCALIB_DIR / "droid_slam", ANYCALIB_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.append(sp)


def setup_extrinsics_imports():
    """Stage-5 sys.path: hawor, its bundled DROID-SLAM frontend, moge-2, geocalib."""
    setup_hawor_imports()
    for p in (HAWOR_DROIDSLAM_DIR, HAWOR_DROIDSLAM_DIR / "droid_slam",
              MOGE2_DIR, GEOCALIB_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def setup_propainter_imports():
    """Put the propainter submodule on sys.path (top-level `model`/`core` imports)."""
    sp = str(PROPAINTER_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def mp4_clip_id(path):
    """Clip id of a source video path: its file name without `.mp4`."""
    return Path(path).name[:-4]


def partition(items, part, key=None):
    """`--part 'a/b'` split. With `key`, an item belongs to part a when crc32(key(item)) % b is
    a-1, so a clip lands in the same part at every stage whatever else is on disk. Without
    `key`, every b-th item starting at index (a-1)."""
    a, b = map(int, part.split('/'))
    if key is None:
        return items[(a - 1)::b]
    return [x for x in items if zlib.crc32(os.fsencode(key(x))) % b == a - 1]

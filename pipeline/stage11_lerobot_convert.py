# Stage 11 — LeRobot conversion: one episode per segment, grouped by output resolution under
# {output}/{HxW}/. Reads stage 10's overlay/{annot,video} tree. Writes episode parquets,
# resized videos and meta/ files. No meta/stats.json: normalisation statistics belong to
# the training corpus, not to one conversion run.
#
#   python pipeline/stage11_lerobot_convert.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition
from common.robot_config import load_robot_config


# ---------------------------------------------------------------------------
# Target joint orders
# ---------------------------------------------------------------------------

ALLEX_TARGET_48_JOINT_NAMES = [
    # right_arm [0:7]
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    # left_arm [7:14]
    "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
    "L_Elbow_Joint", "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    # right_hand [14:29]
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_ABAD_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_ABAD_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_ABAD_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
    # left_hand [29:44]
    "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
    "L_Index_ABAD_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint",
    "L_Middle_ABAD_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint",
    "L_Ring_ABAD_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint",
    "L_Little_ABAD_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint",
    # neck [44:46]
    "Neck_Pitch_Joint", "Neck_Yaw_Joint",
    # waist [46:48]
    "Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint",
]

XYZ_ROT6D_NAMES = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"]


# ---------------------------------------------------------------------------
# Joint remapping
# ---------------------------------------------------------------------------

def build_joint_remap(source_joint_names: list[str], target_joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (src_indices, tgt_indices) remapping state_qpos into target order.
    All source joints must exist in target. Unmatched target slots stay zero."""
    target_lookup = {name: i for i, name in enumerate(target_joint_names)}
    src_indices = []
    tgt_indices = []
    for src_i, name in enumerate(source_joint_names):
        if name not in target_lookup:
            raise ValueError(f"Source joint '{name}' not found in target joint list")
        src_indices.append(src_i)
        tgt_indices.append(target_lookup[name])
    return np.array(src_indices), np.array(tgt_indices)


def remap_joints(source: np.ndarray, src_indices: np.ndarray, tgt_indices: np.ndarray, target_dim: int) -> np.ndarray:
    """Apply the remapping. Unmatched target slots stay zero."""
    T = source.shape[0]
    out = np.zeros((T, target_dim), dtype=np.float64)
    out[:, tgt_indices] = source[:, src_indices]
    return out


# ---------------------------------------------------------------------------
# Image / video resize helpers
# ---------------------------------------------------------------------------

def compute_output_size(h: int, w: int, shorter_side: int) -> tuple:
    """Output (height, width) scaling the shorter side, rounded to even (H.264 requirement)."""
    if h <= w:
        new_h = shorter_side
        new_w = round(w * shorter_side / h)
    else:
        new_w = shorter_side
        new_h = round(h * shorter_side / w)
    new_h = (new_h + 1) // 2 * 2
    new_w = (new_w + 1) // 2 * 2
    return new_h, new_w


def process_img(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize to (out_h, out_w) preserving aspect ratio, no padding. Input/output: RGB uint8."""
    h, w = img.shape[:2]
    if h != out_h or w != out_w:
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Episode-level diagnostic filtering
# ---------------------------------------------------------------------------

# Diagnostic columns read during the pre-scan only, not carried into the output dataset.
_SOURCE_DIAG_COLS = [
    "err_tip_mm_left", "err_tip_mm_right",
    "err_palm_mm_left", "err_palm_mm_right",
    "err_local_dir_left", "err_local_dir_right",
    "err_ddq", "err_cam_pos_mm", "err_cam_rot_deg",
]

# Episode filter {field: percentile}: reject episodes above p(N). 100 rejects nothing.
_DIAG_FILTER = {
    "err_palm_mm_left": 100,
    "err_palm_mm_right": 100,
    "err_tip_mm_left": 100,
    "err_tip_mm_right": 100,
    "err_local_dir_left": 100,
    "err_local_dir_right": 100,
    "err_ddq": 100,
    "err_cam_pos_mm": 99,           # p99 = visual break point
    "err_cam_rot_deg": 99,          # p99 = visual break point
}
# The cuts apply only over at least this many episodes. A p(N) cut always drops the worst
# episode of each field, which in a run of a few episodes is far more than (100 - N)% of it.
_DIAG_FILTER_MIN_EPISODES = 100

# Scalar float diagnostic columns in the source parquet (the rest are per-frame lists).
_SOURCE_SCALAR_FLOAT_COLS = {
    "err_palm_mm_left", "err_palm_mm_right",
    "err_local_dir_left", "err_local_dir_right",
    "err_cam_pos_mm", "err_cam_rot_deg",
}


def _compute_diag_means(table: pa.Table) -> dict:
    """Episode-level diagnostic means (field -> float) from source parquet columns."""
    diag = {}
    col_names = set(table.column_names)

    for col in _SOURCE_DIAG_COLS:
        if col not in col_names:
            continue
        if col in _SOURCE_SCALAR_FLOAT_COLS:
            vals = np.array([v.as_py() for v in table.column(col)], dtype=np.float32)
        else:
            vals = _col_to_np(table, col, dtype=np.float32)
        diag[col] = float(np.nanmean(vals))

    return diag


def compute_diag_thresholds(episodes: list) -> dict:
    """Per-field upper thresholds {field: p_N value} from _DIAG_FILTER over all episodes."""
    thresholds = {}
    for field, percentile in _DIAG_FILTER.items():
        values = [ep.diag_means.get(field) for ep in episodes]
        values = [v for v in values if v is not None and not np.isnan(v)]
        if not values:
            continue
        thresholds[field] = float(np.percentile(values, percentile))
    return thresholds


def filter_episodes_by_diag(episodes: list, thresholds: dict) -> tuple[list, int]:
    """Reject episodes violating any threshold. Returns (filtered_list, num_rejected)."""
    filtered = []
    rejected = 0
    for ep in episodes:
        keep = True
        for field, threshold in thresholds.items():
            val = ep.diag_means.get(field)
            if val is None or np.isnan(val):
                continue
            if val > threshold:
                keep = False
                break
        if keep:
            filtered.append(ep)
        else:
            rejected += 1
    return filtered, rejected


def compute_diag_filter_hash() -> str:
    """SHA256 of the hardcoded diagnostic filter config for the args guard."""
    content = json.dumps({"filter": _DIAG_FILTER, "min_episodes": _DIAG_FILTER_MIN_EPISODES},
                         sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Per-robot target spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeRobotTarget:
    """How a robot's retargeted data maps into the LeRobot dataset."""
    target_joint_names: list        # output joint order for observation.state / action
    state_name_suffix: str          # appended to joint names for observation.state feature names
    video_key: str                  # observation.images.{video_key}
    create_modality_json: Callable


def create_allex_modality_json():
    """Allex modality: 48-DOF target order [RA(7), LA(7), RH(15), LH(15), N(2), W(2)]."""
    return {
        "state": {
            "right_arm_joints":  {"original_key": "observation.state", "start": 0,  "end": 7},
            "left_arm_joints":   {"original_key": "observation.state", "start": 7,  "end": 14},
            "right_hand_joints": {"original_key": "observation.state", "start": 14, "end": 29},
            "left_hand_joints":  {"original_key": "observation.state", "start": 29, "end": 44},
            "neck_joints":       {"original_key": "observation.state", "start": 44, "end": 46},
            "waist_joints":      {"original_key": "observation.state", "start": 46, "end": 48},
            "eef_left":   {"original_key": "observation.state_eef_left",  "start": 0, "end": 9},
            "eef_right":  {"original_key": "observation.state_eef_right", "start": 0, "end": 9},
            "cam_frame":  {"original_key": "observation.state_cam_frame", "start": 0, "end": 9},
        },
        "action": {
            "right_arm_joints":  {"original_key": "action", "start": 0,  "end": 7},
            "left_arm_joints":   {"original_key": "action", "start": 7,  "end": 14},
            "right_hand_joints": {"original_key": "action", "start": 14, "end": 29},
            "left_hand_joints":  {"original_key": "action", "start": 29, "end": 44},
            "neck_joints":       {"original_key": "action", "start": 44, "end": 46},
            "waist_joints":      {"original_key": "action", "start": 46, "end": 48},
            "eef_left":   {"original_key": "action_eef_left",  "start": 0, "end": 9},
            "eef_right":  {"original_key": "action_eef_right", "start": 0, "end": 9},
        },
        "video": {
            "zed_left": {"original_key": "observation.images.zed_left"},
        },
        "annotation": {
            "human.coarse_action": {"original_key": "annotation.human.coarse_action"},
        },
    }


# robot_name -> target spec. A new robot needs its joint order and modality builder here.
_LEROBOT_TARGETS = {
    "allex": LeRobotTarget(
        target_joint_names=ALLEX_TARGET_48_JOINT_NAMES,
        state_name_suffix="_Qpos",
        video_key="zed_left",
        create_modality_json=create_allex_modality_json,
    ),
}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert stage 10 overlay output to LeRobot V2.0 datasets")
    p.add_argument("--robot_name", default="allex",
                   help="Robot config name (configs/<robot_name>.yaml)")
    p.add_argument("--input_dir", required=True, help="Root clip directory")
    p.add_argument("--output_dir", default=None,
                   help="LeRobot dataset output path. If omitted, auto-derived as "
                        "{input_dir}_lerobot/{robot_name}")
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--shorter_side", type=int, default=192,
                   help="Resize video shorter side to this value (no padding)")
    p.add_argument("--chunk_size", type=int, default=1000, help="Episodes per chunk directory")
    p.add_argument("--part", default="1/1", help="Partition string a/b for distributed runs")
    return p.parse_args()


def validate_part(part_str: str) -> tuple:
    """Parse and validate the partition string. Returns (part_a, part_b)."""
    try:
        part_a, part_b = map(int, part_str.split("/"))
    except ValueError:
        raise ValueError(f"Invalid --part format: {part_str!r}, expected 'a/b' (e.g. '1/4')")
    if not (1 <= part_a <= part_b):
        raise ValueError(f"Invalid --part: {part_str}, need 1 <= a <= b")
    return part_a, part_b


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------

class SegmentInfo:
    """Metadata for a single pipeline segment (clip_id + frame range)."""
    __slots__ = ("clip_id", "seg_name", "parquet_path", "video_path")

    def __init__(self, clip_id: str, seg_name: str, parquet_path: Path, video_path: Path):
        self.clip_id = clip_id
        self.seg_name = seg_name
        self.parquet_path = parquet_path
        self.video_path = video_path


def discover_segments(overlay_root: Path) -> list:
    """Enumerate stage 10 segments: annot/{clip_id}/{seg}.parquet + video/{clip_id}/{seg}.mp4."""
    annot_root = overlay_root / "annot"
    video_root = overlay_root / "video"
    assert annot_root.exists(), f"Annotation root not found: {annot_root}"
    assert video_root.exists(), f"Video root not found: {video_root}"

    parquet_paths = sorted(annot_root.glob("*/*.parquet"))

    segments = []
    skipped_no_video = 0
    for pq_path in parquet_paths:
        clip_id = pq_path.parent.name
        seg_name = pq_path.stem
        video_path = video_root / clip_id / f"{seg_name}.mp4"
        if not video_path.exists():
            skipped_no_video += 1
            continue
        segments.append(SegmentInfo(clip_id, seg_name, pq_path, video_path))

    if skipped_no_video:
        print(f"  Warning: skipped {skipped_no_video} segments with missing video "
              f"(annot: {annot_root}, video: {video_root})")

    return segments


# ---------------------------------------------------------------------------
# EpisodeSpec list (one episode per segment)
# ---------------------------------------------------------------------------

class EpisodeSpec:
    """One episode = one segment with merged narration."""
    __slots__ = (
        "seg_info", "narr_merged",
        "num_frames", "out_h", "out_w",
        "episode_index", "task_index", "global_frame_offset",
        "diag_means",
    )

    def __init__(self, seg_info: SegmentInfo, narr_merged: str,
                 num_frames: int, out_h: int, out_w: int):
        self.seg_info = seg_info
        self.narr_merged = narr_merged
        self.num_frames = num_frames
        self.out_h = out_h
        self.out_w = out_w
        # Assigned later in assign_global_indices()
        self.episode_index = -1
        self.task_index = -1
        self.global_frame_offset = -1
        # Assigned in build_episodes()
        self.diag_means = {}


def _probe_video_dims(video_path: Path) -> tuple:
    """Read the video container header to get (height, width). No frame decode."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        h, w = stream.height, stream.width
    if not h or not w:
        raise ValueError(f"Invalid video dimensions {h}x{w} from {video_path}")
    return h, w


def build_episodes(segments: list, shorter_side: int,
                   read_diag: bool = True, disable_tqdm: bool = False) -> list:
    """Pre-scan all segments: read language + num_frames (+ diag_means), probe video dims."""
    iterator = segments
    if not disable_tqdm:
        iterator = tqdm(segments, desc="Pre-scan segments")

    # Cache video dims per clip_id (all segments of one clip share a resolution)
    clip_dims: dict[str, tuple] = {}

    episodes = []
    skipped_corrupt = 0
    for seg in iterator:
        try:
            pf = pq.ParquetFile(seg.parquet_path)
            num_frames = pf.metadata.num_rows

            # schema_arrow, not schema: pf.schema.names flattens nested types
            available = set(pf.schema_arrow.names)
            cols_to_read = ["language"]
            if read_diag:
                cols_to_read += [c for c in _SOURCE_DIAG_COLS if c in available]

            table = pf.read_row_group(0, columns=cols_to_read)
            merged = table.column("language")[0].as_py()

            if merged is None:
                continue

            if seg.clip_id not in clip_dims:
                clip_dims[seg.clip_id] = _probe_video_dims(seg.video_path)
            src_h, src_w = clip_dims[seg.clip_id]
            out_h, out_w = compute_output_size(src_h, src_w, shorter_side)

            ep = EpisodeSpec(seg, merged, num_frames, out_h, out_w)
            if read_diag:
                ep.diag_means = _compute_diag_means(table)
            episodes.append(ep)
        except Exception as e:
            skipped_corrupt += 1
            print(f"  Warning: skipping corrupt segment {seg.parquet_path}: {e}")
            continue

    if skipped_corrupt:
        print(f"  Warning: skipped {skipped_corrupt} corrupt segments (parquet or video)")

    return episodes


def assign_global_indices(episodes: list) -> dict:
    """Assign sequential episode_index, task_index, global_frame_offset. Returns task mapping."""
    # Deterministic task mapping: sorted unique task texts
    all_texts = sorted(set(ep.narr_merged for ep in episodes))
    task_text_to_index = {text: idx for idx, text in enumerate(all_texts)}

    frame_offset = 0
    for idx, ep in enumerate(episodes):
        ep.episode_index = idx
        ep.task_index = task_text_to_index[ep.narr_merged]
        ep.global_frame_offset = frame_offset
        frame_offset += ep.num_frames

    return task_text_to_index


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def output_video_path(output_dir: Path, episode_index: int, chunk_size: int,
                      video_key: str) -> Path:
    chunk = episode_index // chunk_size
    return (output_dir / "videos" / f"chunk-{chunk:03d}" /
            f"observation.images.{video_key}" / f"episode_{episode_index:06d}.mp4")


def process_video(src_path: Path, dst_path: Path, expected_frames: int,
                  fps: int, out_h: int, out_w: int):
    """Streaming decode → resize → H.264 encode. Atomic write via tmp + rename."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.mp4")

    try:
        with av.open(str(src_path)) as container_in:
            container_out = av.open(str(tmp_path), mode="w")
            try:
                stream = container_out.add_stream("libx264", rate=Fraction(fps))
                stream.width = out_w
                stream.height = out_h
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23", "preset": "medium", "g": str(fps)}

                n_frames = 0
                for frame in container_in.decode(video=0):
                    img = frame.to_ndarray(format="rgb24")
                    img = process_img(img, out_h, out_w)
                    out_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    for pkt in stream.encode(out_frame):
                        container_out.mux(pkt)
                    n_frames += 1

                for pkt in stream.encode():
                    container_out.mux(pkt)
            finally:
                container_out.close()

        assert n_frames == expected_frames, (
            f"Frame count mismatch: video {src_path} has {n_frames} frames, "
            f"parquet has {expected_frames} rows"
        )
        os.replace(tmp_path, dst_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def process_videos(episodes: list, output_dir: Path, chunk_size: int,
                   fps: int, video_key: str, disable_tqdm: bool = False):
    """Encode videos in parallel (one per episode/segment)."""
    video_tasks = []
    for ep in episodes:
        dst = output_video_path(output_dir, ep.episode_index, chunk_size, video_key)
        assert ep.seg_info.video_path.exists(), f"Missing source video: {ep.seg_info.video_path}"
        video_tasks.append((ep.seg_info.video_path, dst, ep.num_frames, fps, ep.out_h, ep.out_w))

    num_workers = max(1, int(os.cpu_count() * 0.8))
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(process_video, *task) for task in video_tasks]
        iterator = as_completed(futures)
        if not disable_tqdm:
            iterator = tqdm(iterator, desc="Encoding videos", total=len(futures))
        for fut in iterator:
            fut.result()


# ---------------------------------------------------------------------------
# Parquet construction per episode
# ---------------------------------------------------------------------------

def output_parquet_path(output_dir: Path, episode_index: int, chunk_size: int) -> Path:
    chunk = episode_index // chunk_size
    return (output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet")


def _cam_pose_to_xyz_rot6d(cam_pose_flat: np.ndarray) -> np.ndarray:
    """(T, 16) flattened cam_pose_base (4x4 cam-to-base, OpenCV) -> (T, 9) xyz_rot6d."""
    T = cam_pose_flat.shape[0]
    mat = cam_pose_flat.reshape(T, 4, 4)
    pos = mat[:, :3, 3]                          # (T, 3)
    rot6d = mat[:, :2, :3].reshape(T, 6)         # first two rows → (T, 6)
    return np.concatenate([pos, rot6d], axis=1)  # (T, 9)


def _fixed_list_array(data: np.ndarray, dtype=pa.float64()) -> pa.Array:
    """Convert a 2D numpy array to a PyArrow FixedSizeListArray."""
    assert data.ndim == 2
    flat = pa.array(data.ravel(), type=dtype)
    return pa.FixedSizeListArray.from_arrays(flat, list_size=data.shape[1])


def _build_parquet_schema(target_joint_dim: int) -> pa.Schema:
    """Build the explicit Arrow output schema with fixed-size lists for all vector columns."""
    return pa.schema([
        # Joint state/action in target order
        pa.field("observation.state", pa.list_(pa.float64(), target_joint_dim)),
        pa.field("action", pa.list_(pa.float64(), target_joint_dim)),
        # Wrist EEF (pos3 + rot6d6 per side, base frame)
        pa.field("observation.state_eef_left", pa.list_(pa.float64(), 9)),
        pa.field("observation.state_eef_right", pa.list_(pa.float64(), 9)),
        pa.field("action_eef_left", pa.list_(pa.float64(), 9)),
        pa.field("action_eef_right", pa.list_(pa.float64(), 9)),
        # Camera frame
        pa.field("observation.state_cam_frame", pa.list_(pa.float64(), 9)),
        # Standard LeRobot fields
        pa.field("timestamp", pa.float64()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("annotation.human.coarse_action", pa.int64()),
        pa.field("next.done", pa.bool_()),
        pa.field("next.reward", pa.float64()),
    ])


def write_parquet(columns: dict, schema: pa.Schema, dst_path: Path):
    """Atomic parquet write via tmp + rename, with explicit schema."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.parquet")
    table = pa.table(columns, schema=schema)
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, dst_path)


def _col_to_np(table, col_name, dtype=np.float64):
    """Convert a PyArrow column of equal-length lists to a 2D numpy array."""
    col = table.column(col_name)
    return np.array(col.to_pylist(), dtype=dtype)


def _next_frame_shift(arr: np.ndarray) -> np.ndarray:
    """Action = next frame's state. The last frame copies itself."""
    out = np.empty_like(arr)
    if arr.shape[0] > 1:
        out[:-1] = arr[1:]
    out[-1] = arr[-1]
    return out


def build_episode_parquet(ep: EpisodeSpec, target: LeRobotTarget, joint_remap: tuple,
                          source_joint_dim: int, fps: int, chunk_size: int,
                          output_dir: Path, schema: pa.Schema):
    """Read the source parquet, validate, transform columns, write the output parquet."""
    table = pq.read_table(ep.seg_info.parquet_path)
    T = table.num_rows
    assert T == ep.num_frames, f"Frame count changed: {ep.seg_info.parquet_path}"

    # --- EEF: truncate to wrist-only (first 9 dims = pos3 + rot6d6) per side ---
    eef_left_full = _col_to_np(table, "state_eef_left")
    eef_right_full = _col_to_np(table, "state_eef_right")
    state_eef_left = eef_left_full[:, :9].astype(np.float64)
    state_eef_right = eef_right_full[:, :9].astype(np.float64)
    action_eef_left = _next_frame_shift(state_eef_left)
    action_eef_right = _next_frame_shift(state_eef_right)

    # --- Joint remapping into target order ---
    state_joint_src = _col_to_np(table, "state_qpos")
    assert state_joint_src.shape[1] == source_joint_dim
    src_idx, tgt_idx = joint_remap
    state_joint = remap_joints(state_joint_src, src_idx, tgt_idx, len(target.target_joint_names))
    action_joint = _next_frame_shift(state_joint)

    # --- cam_pose_base → cam_frame (9-dim xyz_rot6d) ---
    cam_pose_col = table.column("cam_pose_base")
    cam_pose_flat = np.array(
        [np.array(v.as_py(), dtype=np.float64).flatten() for v in cam_pose_col],
        dtype=np.float64,
    )
    assert cam_pose_flat.shape == (T, 16)
    cam_frame = _cam_pose_to_xyz_rot6d(cam_pose_flat)  # (T, 9)

    # --- Build output columns ---
    out = {
        "observation.state": _fixed_list_array(state_joint),
        "action": _fixed_list_array(action_joint),
        "observation.state_eef_left": _fixed_list_array(state_eef_left),
        "observation.state_eef_right": _fixed_list_array(state_eef_right),
        "action_eef_left": _fixed_list_array(action_eef_left),
        "action_eef_right": _fixed_list_array(action_eef_right),
        "observation.state_cam_frame": _fixed_list_array(cam_frame),
        "timestamp": pa.array(np.arange(T, dtype=np.float64) / fps),
        "frame_index": pa.array(list(range(T)), type=pa.int64()),
        "episode_index": pa.array([ep.episode_index] * T, type=pa.int64()),
        "index": pa.array(list(range(ep.global_frame_offset, ep.global_frame_offset + T)), type=pa.int64()),
        "task_index": pa.array([ep.task_index] * T, type=pa.int64()),
        "annotation.human.coarse_action": pa.array([ep.task_index] * T, type=pa.int64()),
        "next.done": pa.array([False] * (T - 1) + [True], type=pa.bool_()),
        "next.reward": pa.array([0.0] * T, type=pa.float64()),
    }

    dst = output_parquet_path(output_dir, ep.episode_index, chunk_size)
    write_parquet(out, schema, dst)


# ---------------------------------------------------------------------------
# Metadata files
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, content: str):
    """Atomic text file write via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(content)
    os.replace(tmp, path)


def write_tasks_jsonl(task_text_to_index: dict, output_dir: Path):
    lines = []
    for text, idx in sorted(task_text_to_index.items(), key=lambda x: x[1]):
        lines.append(json.dumps({"task_index": idx, "task": text}))
    _atomic_write_text(output_dir / "meta" / "tasks.jsonl", "\n".join(lines) + "\n")


def write_episodes_jsonl(episodes: list, output_dir: Path):
    lines = []
    for ep in episodes:
        entry = {
            "episode_index": ep.episode_index,
            "tasks": [ep.narr_merged],
            "length": ep.num_frames,
            "clip_id": ep.seg_info.clip_id,
            "seg_name": ep.seg_info.seg_name,
        }
        lines.append(json.dumps(entry))
    _atomic_write_text(output_dir / "meta" / "episodes.jsonl", "\n".join(lines) + "\n")


def write_info_json(episodes: list, task_text_to_index: dict, robot_name: str,
                    target: LeRobotTarget, fps: int, chunk_size: int,
                    video_shape: list, output_dir: Path):
    J = len(target.target_joint_names)
    total_episodes = len(episodes)
    total_frames = sum(ep.num_frames for ep in episodes)

    eef_feature = {"dtype": "float64", "shape": [9], "names": XYZ_ROT6D_NAMES}

    action_names = target.target_joint_names
    state_names = [n + target.state_name_suffix for n in action_names]
    features = {
        "observation.state": {"dtype": "float64", "shape": [J], "names": state_names},
        "action": {"dtype": "float64", "shape": [J], "names": action_names},
        f"observation.images.{target.video_key}": {
            "dtype": "video",
            "shape": video_shape,
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state_eef_left": eef_feature,
        "observation.state_eef_right": eef_feature,
        "action_eef_left": eef_feature,
        "action_eef_right": eef_feature,
        "observation.state_cam_frame": {"dtype": "float64", "shape": [9], "names": XYZ_ROT6D_NAMES},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "annotation.human.coarse_action": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "next.reward": {"dtype": "float64", "shape": [1]},
    }

    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_name,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_text_to_index),
        "total_videos": total_episodes,
        "total_chunks": (total_episodes + chunk_size - 1) // chunk_size,
        "chunks_size": chunk_size,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    _atomic_write_text(output_dir / "meta" / "info.json", json.dumps(info, indent=2) + "\n")


def write_modality_json(target: LeRobotTarget, output_dir: Path):
    modality = target.create_modality_json()
    _atomic_write_text(output_dir / "meta" / "modality.json", json.dumps(modality, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Args guard: prevent silent data corruption
# ---------------------------------------------------------------------------

def compute_episodes_hash(episodes: list) -> str:
    """SHA256 over deterministic episode content tuples."""
    parts = []
    for ep in episodes:
        parts.append(f"{ep.seg_info.clip_id}\t{ep.seg_info.seg_name}\t"
                     f"{ep.narr_merged}\t{ep.num_frames}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def compute_guard_dict(args, total_episodes: int, episodes_hash: str) -> dict:
    return {
        "robot_name": args.robot_name,
        "input_dir": args.input_dir,
        "fps": args.fps,
        "shorter_side": args.shorter_side,
        "chunk_size": args.chunk_size,
        "diag_filter_hash": compute_diag_filter_hash(),
        "total_episodes": total_episodes,
        "episodes_hash": episodes_hash,
    }


def check_or_write_args_guard(guard: dict, output_dir: Path):
    """Every partition checks/writes. Race-safe via atomic write."""
    guard_path = output_dir / "args.json"
    if guard_path.exists():
        saved = json.loads(guard_path.read_text())
        if saved != guard:
            diff = {k: (saved.get(k), guard.get(k))
                    for k in set(saved) | set(guard) if saved.get(k) != guard.get(k)}
            raise RuntimeError(
                f"Args mismatch. Delete {output_dir} and re-run.\nDiff: {diff}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp = guard_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        with open(tmp, "w") as f:
            json.dump(guard, f, indent=2)
        os.replace(tmp, guard_path)


# ---------------------------------------------------------------------------
# Resumability: output files as done signal
# ---------------------------------------------------------------------------

def segment_is_complete(seg_episodes: list, output_dir: Path, chunk_size: int,
                        video_key: str) -> bool:
    """Check whether ALL expected output files for a segment's episodes exist."""
    for ep in seg_episodes:
        if not output_parquet_path(output_dir, ep.episode_index, chunk_size).exists():
            return False
        if not output_video_path(output_dir, ep.episode_index, chunk_size, video_key).exists():
            return False
    return True


def clean_segment_outputs(seg_episodes: list, output_dir: Path, chunk_size: int,
                          video_key: str):
    """Remove all existing outputs for a segment so it can be fully regenerated."""
    for ep in seg_episodes:
        for path in [output_parquet_path(output_dir, ep.episode_index, chunk_size),
                     output_video_path(output_dir, ep.episode_index, chunk_size, video_key)]:
            if path.exists():
                path.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_resolution_group(res_tag: str, group_episodes: list, args,
                              target: LeRobotTarget, joint_remap: tuple,
                              source_joint_dim: int, base_output_dir: Path,
                              part_a: int):
    """Process one resolution group as its own LeRobot dataset under {base_output_dir}/{res_tag}/."""
    output_dir = base_output_dir / res_tag
    out_h, out_w = group_episodes[0].out_h, group_episodes[0].out_w
    video_shape = [out_h, out_w, 3]
    schema = _build_parquet_schema(len(target.target_joint_names))

    task_text_to_index = assign_global_indices(group_episodes)
    print(f"  [{res_tag}] {len(group_episodes)} episodes, "
          f"{len(task_text_to_index)} tasks, video {out_h}x{out_w}")

    episodes_hash = compute_episodes_hash(group_episodes)
    guard = compute_guard_dict(args, len(group_episodes), episodes_hash)
    guard["resolution"] = res_tag
    check_or_write_args_guard(guard, output_dir)

    all_seg_keys = [(ep.seg_info.clip_id, ep.seg_info.seg_name) for ep in group_episodes]
    part_seg_keys_list = partition(all_seg_keys, args.part)
    part_seg_keys = set(part_seg_keys_list)
    part_episodes = [ep for ep in group_episodes
                     if (ep.seg_info.clip_id, ep.seg_info.seg_name) in part_seg_keys]

    # Group by segment for resumability
    seg_to_episodes: dict[tuple, list] = {}
    for ep in part_episodes:
        key = (ep.seg_info.clip_id, ep.seg_info.seg_name)
        seg_to_episodes.setdefault(key, []).append(ep)

    # Skip complete segments, clean incomplete ones
    segments_to_process = []
    skipped = 0
    for key in part_seg_keys_list:
        seg_eps = seg_to_episodes.get(key, [])
        if not seg_eps:
            continue
        if segment_is_complete(seg_eps, output_dir, args.chunk_size, target.video_key):
            skipped += 1
            continue
        clean_segment_outputs(seg_eps, output_dir, args.chunk_size, target.video_key)
        segments_to_process.append(key)
    if skipped:
        print(f"  [{res_tag}] Skipped {skipped} already-complete segments")

    episodes_to_process = []
    for key in segments_to_process:
        episodes_to_process.extend(seg_to_episodes[key])

    if episodes_to_process:
        process_videos(episodes_to_process, output_dir, args.chunk_size,
                       args.fps, target.video_key, disable_tqdm=args.no_tqdm)

    if episodes_to_process:
        iterator = episodes_to_process
        if not args.no_tqdm:
            iterator = tqdm(episodes_to_process, desc=f"[{res_tag}] Building parquets")
        for ep in iterator:
            build_episode_parquet(ep, target, joint_remap, source_joint_dim,
                                  args.fps, args.chunk_size, output_dir, schema)

    # Only partition 1 writes meta/
    if part_a == 1:
        write_tasks_jsonl(task_text_to_index, output_dir)
        write_episodes_jsonl(group_episodes, output_dir)
        write_info_json(group_episodes, task_text_to_index, args.robot_name,
                        target, args.fps, args.chunk_size, video_shape, output_dir)
        write_modality_json(target, output_dir)
        print(f"  [{res_tag}] Wrote meta/ files")


def main():
    args = parse_args()
    if args.robot_name not in _LEROBOT_TARGETS:
        raise NotImplementedError(
            f"robot '{args.robot_name}' is not supported. Available: {sorted(_LEROBOT_TARGETS)}")
    target = _LEROBOT_TARGETS[args.robot_name]

    robot_cfg = load_robot_config(args.robot_name)
    joint_remap = build_joint_remap(robot_cfg.joint_names, target.target_joint_names)

    input_dir = osp.normpath(args.input_dir)
    overlay_root = Path(input_dir + "_chunked") / args.robot_name / "overlay"

    if args.output_dir is None:
        args.output_dir = str(Path(input_dir + "_lerobot") / args.robot_name)
    base_output_dir = Path(args.output_dir)

    part_a, _ = validate_part(args.part)
    print(f"Overlay root: {overlay_root}")
    print(f"Base output directory: {base_output_dir}")

    segments = discover_segments(overlay_root)
    assert len(segments) > 0, f"No segments found under {overlay_root}"
    print(f"Discovered {len(segments)} segments")

    # Diagnostic column reads are skipped when every filter is at 100 (no-op)
    diag_active = any(v < 100 for v in _DIAG_FILTER.values())
    episodes = build_episodes(segments, args.shorter_side,
                              read_diag=diag_active, disable_tqdm=args.no_tqdm)
    assert len(episodes) > 0, "No episodes to convert"
    print(f"Total: {len(episodes)} episodes (before diagnostic filter)")

    thresholds = compute_diag_thresholds(episodes) if diag_active else {}
    if thresholds and len(episodes) < _DIAG_FILTER_MIN_EPISODES:
        print(f"Diagnostic filter: skipped, {len(episodes)} episodes "
              f"(it applies from {_DIAG_FILTER_MIN_EPISODES})")
        thresholds = {}
    if thresholds:
        episodes, n_rejected = filter_episodes_by_diag(episodes, thresholds)
        for field, value in sorted(thresholds.items()):
            print(f"  {field}: upper {value:.4f}")
        print(f"Diagnostic filter: {n_rejected} rejected, {len(episodes)} remaining")
        assert len(episodes) > 0, "No episodes remaining after diagnostic filtering"

    # Group episodes by output resolution → separate LeRobot datasets
    res_groups: dict[str, list] = {}
    for ep in episodes:
        tag = f"{ep.out_h}x{ep.out_w}"
        res_groups.setdefault(tag, []).append(ep)

    res_summary = ", ".join(f"{tag}: {len(eps)}" for tag, eps in sorted(res_groups.items()))
    print(f"Resolution groups: {res_summary}")

    for res_tag, group_episodes in sorted(res_groups.items()):
        _process_resolution_group(res_tag, group_episodes, args, target, joint_remap,
                                  robot_cfg.state_dim, base_output_dir, part_a)

    print("Done.")


if __name__ == "__main__":
    main()

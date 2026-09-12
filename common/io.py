"""Shared Parquet I/O: ONE cumulative SCHEMA for every pipeline stage.
Stages 2 to 6 write chunked shards (`{clip_id}_{NN}.parquet` + an atomic `{clip_id}.done`),
filling only the fields they compute and leaving the rest None."""
import os
import struct
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_F32 = pa.float32()

# dictionary-encoded columns
_DICT_ENCODE = {
    "clip_id": True, "height": True, "width": True, "intr_model": True,
    "hands.item.side": True,
    "fx": True, "fy": True, "cx": True, "cy": True, "xi": True,
    "n_person_det": True,
}


def _fixed_list(t, n):
    """Fixed-size list type, portable across pyarrow versions."""
    fsl = getattr(pa, "fixed_size_list", None)
    if fsl is not None:
        return fsl(t, n)
    try:
        return pa.list_(t, list_size=n)
    except TypeError:
        return pa.list_(t)


def _valid_narr(text):
    """A narration field that carries a usable instruction, else None."""
    if text is None or text == "n/a" or text.strip() == "":
        return None
    return text


def merge_narration(narr):
    """One instruction out of the per-hand fields: bimanual, else the longer of
    left/right (right on a tie). None when nothing usable is present."""
    if not narr:
        return None
    left = _valid_narr(narr.get("left"))
    right = _valid_narr(narr.get("right"))
    bimanual = _valid_narr(narr.get("bimanual"))
    if bimanual is not None:
        return bimanual
    if left is not None and right is not None:
        return right if len(right) >= len(left) else left
    return left if left is not None else right


def encode_mask_bool(mask_bool):
    """(H,W) bool -> [magic 'BMK1' | H | W] + bit-packed payload."""
    assert mask_bool.dtype == np.bool_
    H, W = map(int, mask_bool.shape)
    packed = np.packbits(mask_bool.reshape(-1), bitorder="big").tobytes()
    return struct.pack(">4sII", b"BMK1", H, W) + packed


def decode_mask_bytes(b):
    """Inverse of encode_mask_bool -> (H,W) bool ndarray."""
    magic, H, W = struct.unpack(">4sII", b[:12])
    assert magic == b"BMK1"
    flat = np.unpackbits(np.frombuffer(b[12:], dtype=np.uint8), bitorder="big")[:H * W]
    return flat.reshape(H, W).astype(bool)


_KPTS3D_TYPE = _fixed_list(_fixed_list(_F32, 3), 21)      # (21, 3)
_WRIST_ROT_TYPE = _fixed_list(_F32, 3)                    # axis-angle (3,)
_FINGER_ROT_TYPE = _fixed_list(_fixed_list(_F32, 3), 15)  # (15, 3) axis-angles
_EXTR_TYPE = pa.list_(pa.list_(_F32))                     # 4x4 T_cam2world (or None)

HAND_ITEM = pa.struct([
    ("box",   _fixed_list(_F32, 4)),   # [x1,y1,x2,y2]
    ("conf",  _F32),
    ("side",  pa.int8()),              # 0=L, 1=R
    ("side_conf", _F32),
    ("kpts3d", _KPTS3D_TYPE),
    ("wrist_rot", _WRIST_ROT_TYPE),
    ("finger_rot", _FINGER_ROT_TYPE),
    ("hand_mask", pa.binary()),        # bit-packed (H,W) bool
])

NARR_TYPE = pa.struct([
    ("think", pa.string()), ("left", pa.string()),
    ("right", pa.string()), ("bimanual", pa.string()),
])

SCHEMA = pa.schema([
    ("clip_id", pa.string()),
    ("frame_id", pa.int32()),
    ("height",  pa.int32()),
    ("width",   pa.int32()),
    ("intr_model", pa.string()),
    ("hands",   pa.list_(HAND_ITEM)),
    ("fx", pa.float64()), ("fy", pa.float64()),
    ("cx", pa.float64()), ("cy", pa.float64()), ("xi", pa.float64()),
    ("cam_pose", _EXTR_TYPE),        # 4x4 cam-to-world, OpenCV (stage 5)
    ("cam_pose_base", _EXTR_TYPE),   # 4x4 cam-to-robot-base, OpenCV (stage 9)
    ("arm_mask", pa.binary()),
    ("n_person_det", pa.int32()),
    ("narr", NARR_TYPE),
    ("language", pa.string()),   # narr merged down to one instruction, see merge_narration
    # Retargeting output. List widths are recorded in schema metadata.
    # state_mask_* is false where that side's hand is missing. The IK
    # solves those frames with no hand target, so their state follows
    # from the frames around them. A side missing from the whole
    # segment is held at the home pose.
    ("state_qpos", pa.list_(_F32)),        # active joint angles, config group order
    ("state_eef_left", pa.list_(_F32)),    # wrist pos3 + rot6d6 + hand joints, base frame
    ("state_eef_right", pa.list_(_F32)),
    ("state_mask_left", pa.bool_()),
    ("state_mask_right", pa.bool_()),
    ("retarget_cost", _F32),               # segment-level solver cost, per frame
    ("err_tip_mm_left", pa.list_(_F32)),   # per-fingertip L2 to the target keypoint
    ("err_tip_mm_right", pa.list_(_F32)),
    ("err_palm_mm_left", _F32),
    ("err_palm_mm_right", _F32),
    ("err_local_dir_left", _F32),          # mean cosine distance over connectivity pairs
    ("err_local_dir_right", _F32),
    ("err_ddq", pa.list_(_F32)),           # per-joint 2nd-order finite difference
    ("err_cam_pos_mm", _F32),
    ("err_cam_rot_deg", _F32),
    # False where Isaac Sim failed to render the frame
    ("overlay_valid", pa.bool_()),
])

# Top-level float-array columns: ndarray on write, float32 ndarray on read.
_ARRAY_FIELDS = (
    "state_qpos", "state_eef_left", "state_eef_right",
    "err_tip_mm_left", "err_tip_mm_right", "err_ddq",
)

# pyarrow: null fixed-size-list inside a struct corrupts -> NaN-pad
_HAND_FSL_NAN = {
    "kpts3d": [[float("nan")] * 3] * 21,
    "wrist_rot": [float("nan")] * 3,
    "finger_rot": [[float("nan")] * 3] * 15,
}


def hand_for_side(row, side, require_kpts3d=False):
    """The hand for `side` (0=L, 1=R), or None. require_kpts3d skips hands without kpts3d."""
    for h in (row.get("hands") or []):
        if h is None or h.get("side") != side:
            continue
        if require_kpts3d and h.get("kpts3d") is None:
            continue
        return h
    return None


def normalize_rows(rows):
    """Write-side encode for SCHEMA: bit-pack masks, ndarray -> list, NaN-pad absent
    per-hand fixed-size-list fields."""
    out = []
    for r in rows:
        r2 = dict(r)
        hands = r2.get("hands")
        if hands is not None:
            nh = []
            for h in hands:
                if h is None:
                    nh.append(None); continue
                h2 = dict(h)
                hm = h2.get("hand_mask")
                h2["hand_mask"] = encode_mask_bool(hm) if hm is not None else None
                for k, v in list(h2.items()):
                    if isinstance(v, np.ndarray):
                        h2[k] = v.astype("float32").tolist()
                for k, filler in _HAND_FSL_NAN.items():
                    if h2.get(k) is None:
                        h2[k] = filler
                nh.append(h2)
            r2["hands"] = nh
        for key in ("cam_pose", "cam_pose_base"):
            cp = r2.get(key)
            if isinstance(cp, np.ndarray):
                r2[key] = cp.astype("float32").tolist()
        am = r2.get("arm_mask")
        if am is not None:
            assert isinstance(am, np.ndarray) and am.dtype == np.bool_
            r2["arm_mask"] = encode_mask_bool(am)
        for k in _ARRAY_FIELDS:
            v = r2.get(k)
            if isinstance(v, np.ndarray):
                r2[k] = v.astype("float32").tolist()
        out.append(r2)
    return out


def decode_row(row):
    """Read-side inverse of normalize_rows: mask bytes -> bool arrays, lists -> float32
    arrays, all-NaN per-hand fixed-size-lists -> None."""
    hands = row.get("hands")
    if hands is not None:
        dec = []
        for it in hands:
            if it is None:
                dec.append(None); continue
            it = dict(it)
            hm = it.get("hand_mask")
            if isinstance(hm, (bytes, bytearray, memoryview)):
                it["hand_mask"] = decode_mask_bytes(hm)
            for k, v in list(it.items()):
                if k == "hand_mask" or v is None:
                    continue
                if isinstance(v, (list, np.ndarray)):
                    it[k] = np.asarray(v, dtype=np.float32)
            for k in ("kpts3d", "wrist_rot", "finger_rot"):
                v = it.get(k)
                if isinstance(v, np.ndarray) and np.isnan(v).all():
                    it[k] = None
            dec.append(it)
        row["hands"] = dec
    for key in ("cam_pose", "cam_pose_base"):
        cp = row.get(key)
        if cp is not None:
            row[key] = np.asarray(cp, dtype=np.float32)
    am = row.get("arm_mask")
    if isinstance(am, (bytes, bytearray, memoryview)):
        row["arm_mask"] = decode_mask_bytes(am)
    for k in _ARRAY_FIELDS:
        v = row.get(k)
        if isinstance(v, list):
            row[k] = np.asarray(v, dtype=np.float32)
    return row


def rows_to_parquet(rows, out_path, metadata=None):
    """Write rows as one Parquet shard. `metadata` attaches schema-level key/value strings."""
    schema = SCHEMA if metadata is None else SCHEMA.with_metadata(metadata)
    table = pa.Table.from_pylist(normalize_rows(rows), schema=schema)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path,
        compression="zstd",
        compression_level=1,  # avoids a CPU peak
        write_batch_size=512,
        use_dictionary=_DICT_ENCODE,
    )


def write_clip_chunks(clip_id, rows, out_dir, chunk_idx):
    if not rows: return
    rows_to_parquet(rows, os.path.join(out_dir, f"{clip_id}_{chunk_idx:02d}.parquet"))


def mark_done(output_dir, clip_id):
    """Atomic per-clip completion marker (write tmp, then rename)."""
    done = os.path.join(output_dir, f"{clip_id}.done")
    tmp = done + ".tmp"
    with open(tmp, "wb"): pass
    os.replace(tmp, done)


class ParquetReader:
    """Random-access reader for a clip's chunked shards (`{root}/{clip_id}_NN.parquet` +
    `{clip_id}.done`). Raises FileNotFoundError if the clip has no shards or no `.done`."""

    def __init__(self, root, clip_id, cols=None, decode=decode_row):
        self.root = Path(root)
        self.clip_id = clip_id
        self.cols = None if cols is None else list(cols)
        self._decode = decode
        self._read_cols = None if self.cols is None else sorted(set(self.cols + ["frame_id"]))

        files = sorted(self.root.glob(f"{clip_id}_*.parquet"))  # 2-digit names sort correctly
        done_path = self.root / f"{clip_id}.done"
        if not files or not done_path.is_file():
            raise FileNotFoundError(
                f"No completed data for {clip_id} under {self.root} "
                f"(parquet: {bool(files)}, done: {done_path.exists()})"
            )
        self.files = [str(p) for p in files]

        self._fid_to_loc = {}
        for file_idx, path in enumerate(self.files):
            fids = pq.read_table(path, columns=["frame_id"])["frame_id"].to_numpy(zero_copy_only=False)
            self._fid_to_loc.update({int(fid): (file_idx, i) for i, fid in enumerate(fids)})

        self._cur_idx = None
        self._tbl = None

    def _ensure_table_loaded(self, file_idx):
        if self._cur_idx == file_idx and self._tbl is not None:
            return
        path = self.files[file_idx]
        self._tbl = pq.read_table(path, columns=self._read_cols) if self._read_cols else pq.read_table(path)
        self._cur_idx = file_idx

    def get(self, frame_id):
        """Decoded row dict for `frame_id`, or None if absent."""
        loc = self._fid_to_loc.get(int(frame_id))
        if loc is None:
            return None
        file_idx, row_idx = loc
        self._ensure_table_loaded(file_idx)
        raw = self._tbl.slice(row_idx, 1).to_pylist()[0]
        row = raw if self.cols is None else {k: raw.get(k) for k in self.cols}
        return self._decode(row) if self._decode is not None else row

    def get_frame_ids(self):
        return sorted(self._fid_to_loc.keys())

    def get_max_frame_id(self):
        """Largest frame_id across all shards (-1 if empty)."""
        return max(self._fid_to_loc) if self._fid_to_loc else -1

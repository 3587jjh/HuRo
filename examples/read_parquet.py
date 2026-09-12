"""Reads one segment out of a `<clips>_chunked` tree, and the reference for reading these
Parquet tables.

  <clips>_chunked/
    original/annot/<clip_id>/<start>_<end>.parquet          human annotations (stages 2-7)
    <robot>/annot/<clip_id>/<start>_<end>.parquet           retargeted robot (stage 9)
    <robot>/overlay/annot/<clip_id>/<start>_<end>.parquet   the same plus overlay_valid (stage 10)

All three use the one schema in `common/io.py`. A stage fills the columns it computes and
leaves every other column null. Rows line up across the three by `frame_id`, which restarts at
0 in each segment, so a row's index in the clip is `<start> + frame_id`.

Running this prints one segment's schema metadata, the columns each table filled, and one
frame. Each table is read with `pq.read_table(...).to_pylist()` and decoded with `decode_row`.

  python examples/read_parquet.py examples/clips_chunked
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import decode_row


def describe(value):
    """One line for whatever a decoded column holds."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return f"array{tuple(value.shape)} {value.dtype}"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"{len(value)} hand(s)"
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        filled = [k for k, v in value.items() if v not in (None, "")]
        return "{" + ", ".join(filled) + "}"
    if isinstance(value, str):
        return repr(value if len(value) <= 48 else value[:45] + "...")
    return repr(value)


def report(path, title):
    table = pq.read_table(path)
    rows = [decode_row(r) for r in table.to_pylist()]
    print(f"\n{title}")
    print(f"  {path}")
    print(f"  {len(rows)} rows, frame_id {rows[0]['frame_id']}..{rows[-1]['frame_id']}")

    # Schema-level metadata, which `to_pylist` drops. Stages 9 and 10 record the robot here, so
    # their tables say what `state_qpos` and `state_eef_*` hold without the config beside them.
    meta = table.schema.metadata or {}
    if meta:
        print(f"  schema metadata ({len(meta)} keys):")
        for key, value in sorted(meta.items()):
            text = value.decode()
            if len(text) > 60:
                text = text[:57] + "..."
            print(f"    {key.decode():34} {text}")

    filled = [c for c in rows[0] if any(r.get(c) is not None for r in rows)]
    print(f"  columns with data ({len(filled)} of {len(rows[0])}):")
    mid = rows[len(rows) // 2]
    for col in filled:
        shown = describe(mid.get(col))
        print(f"    {col:22} {shown if shown is not None else '(null on this frame)'}")
    return rows


def report_hands(row):
    print("\nhands on one frame (the per-hand struct)")
    for hand in row.get("hands") or []:
        side = {0: "left", 1: "right"}.get(hand.get("side"), "?")
        print(f"  {side}:")
        for key in ("box", "conf", "side_conf", "kpts3d", "wrist_rot", "finger_rot", "hand_mask"):
            print(f"    {key:12} {describe(hand.get(key))}")


def main(root):
    root = Path(root)
    human = sorted(root.glob("original/annot/*/*.parquet"))
    if not human:
        sys.exit(f"no original/annot/<clip>/<seg>.parquet under {root}")
    human_path = human[0]
    clip_id, seg = human_path.parent.name, human_path.stem
    print(f"{root}: clip {clip_id}, segment {seg}")
    if len(human) > 1:
        print(f"  ({len(human)} segments here, showing the first)")

    human_rows = report(human_path, "human table (stages 2-7)")
    report_hands(human_rows[len(human_rows) // 2])

    for robot_dir in sorted(d for d in root.iterdir() if d.is_dir() and d.name != "original"):
        for sub, title in (("annot", "stage 9"), ("overlay/annot", "stage 10")):
            path = robot_dir / sub / clip_id / f"{seg}.parquet"
            if path.is_file():
                report(path, f"{robot_dir.name} table, {title}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))

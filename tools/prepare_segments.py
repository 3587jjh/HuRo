#!/usr/bin/env python3
"""Cut your own copy of a source corpus to the segment boundaries of a published HuRo dataset.

The HuRo datasets ship annotations, and for some subsets the arm-removed video, but never the
unmodified source frames. This rebuilds those frames from a corpus you downloaded yourself, in
the layout the pipeline reads, so stages 7 to 10 run on the result:

    tools/prepare_segments.py --tables  <dataset>/clips_chunked \
                              --source  <your copy of the corpus> \
                              --out     clips

writes `clips_chunked/original/video/<clip_id>/<start>_<end>.mp4`, one file per segment of the
dataset, each the frames its table describes.

Two things make this more than `ffmpeg -ss`:

  * `start` and `end` index the clip **at the frame size and rate the tables record**, not the
    original file. That is the README's 256-pixel-shorter-side, 30 fps encode for EPIC-KITCHENS
    and Ego4D, and the subset's own size for Egocentric-10K and EgoVerse — so the target is read
    from the table rather than assumed. A clip that already matches is used as it is; anything
    else is re-encoded once and cached beside the output.
  * Stage 6 **rectifies** every frame before writing a segment, so the geometry columns in the
    dataset describe the rectified frame. Where `xi` is non-zero the same maps are rebuilt here
    from the table's own `fx, fy, cx, cy, xi` and applied. Rectification preserves the frame
    size, so a cut made without it is wrong in a way nothing downstream notices: the hand
    keypoints and camera poses simply do not line up with the pixels. Most EPIC-KITCHENS and Ego4D
    segments are affected; Egocentric-10K and EgoVerse carry `xi = 0` throughout.

Every segment written is checked against its table: frame count, frame size, and the pinhole the
rebuilt maps produce against the `pinhole_*` columns the dataset publishes. A segment that fails
is not written, and the exit status is non-zero if any did.

    --check-only   run every check, write nothing
    --clips        comma list, or @file with one clip id per line (default: all in --tables)
    --jobs         clips in parallel (default 4; each uses one ffmpeg and one decoder)

ffmpeg is needed only to re-encode a clip that is not already at the tables' geometry. A corpus
that already is, and `--check-only`, run without it.
"""
import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.video import save_video_task

VIDEO_SUFFIXES = (".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".webm")
FPS = 30


def find_source(source_root: Path, clip_id: str):
    """The source file for `clip_id`, searched recursively. A clip id is the source file's name
    without its suffix (`common/paths.py:mp4_clip_id`), which holds for every HuRo corpus."""
    for suffix in VIDEO_SUFFIXES:
        direct = source_root / (clip_id + suffix)
        if direct.is_file():
            return direct
    hits = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix in VIDEO_SUFFIXES
                  and p.stem == clip_id)
    if len(hits) > 1:
        raise FileNotFoundError(f"{clip_id}: {len(hits)} source files match, cannot choose: "
                                + ", ".join(str(h) for h in hits[:4]))
    return hits[0] if hits else None


def probe(path: Path):
    """(width, height, fps) of a video's first stream."""
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        return st.codec_context.width, st.codec_context.height, float(st.average_rate or 0)


def source_at_table_geometry(src: Path, cache: Path, W: int, H: int, clip_id: str):
    """The clip at the frame size and rate the dataset's indices refer to.

    The target comes from the table, not from a fixed recipe: the published subsets were not all
    encoded the same way. EPIC-KITCHENS and Ego4D carry the README's 256-pixel-shorter-side
    re-encode, while Egocentric-10K and EgoVerse were prepared at their own size upstream. A clip
    that already matches is used as it is; anything else is re-encoded to exactly W x H at 30 fps
    and cached, so re-running costs nothing."""
    sw, sh, sfps = probe(src)
    if (sw, sh) == (W, H) and abs(sfps - FPS) < 0.01:
        return src
    if abs((sw / sh) - (W / H)) > 0.02:
        raise RuntimeError(f"{clip_id}: the source is {sw}x{sh} and the tables describe {W}x{H}, "
                           "a different aspect ratio. Re-encoding would distort the frame, so "
                           "this is almost certainly the wrong source file.")
    if cache.is_file() and cache.stat().st_size > 0 and probe(cache)[:2] == (W, H):
        return cache
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{clip_id}: the source is {sw}x{sh} at {sfps:.3f} fps and the tables describe "
            f"{W}x{H} at {FPS} fps, so it has to be re-encoded first, and ffmpeg is not on "
            "PATH. setup/setup.sh installs it. A corpus already at the tables' geometry needs "
            "no ffmpeg at all.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src),
           "-vf", f"fps={FPS},scale={W}:{H}", "-c:v", "libx264", "-crf", "18", "-an", str(tmp)]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src}:\n{r.stderr.decode()[-2000:]}")
    os.replace(tmp, cache)
    return cache


def segment_table(annot_dir: Path, stem: str):
    """(row count, row 0) of a segment's table — everything this tool needs from it."""
    t = pq.read_table(annot_dir / f"{stem}.parquet")
    cols = ["height", "width", "fx", "fy", "cx", "cy", "xi",
            "pinhole_fx", "pinhole_fy", "pinhole_cx", "pinhole_cy"]
    head = {c: t.column(c)[0].as_py() for c in cols if c in t.schema.names}
    return t.num_rows, head


def clip_maps(head, clip_id):
    """(needs_undistort, map1, map2) for a clip, with the rebuilt pinhole checked against the
    `pinhole_*` columns. A mismatch means the maps are not stage 6's, so the cut would be wrong."""
    shape = (int(head["height"]), int(head["width"]))
    intr = [head["fx"], head["fy"], head["cx"], head["cy"], head["xi"]]
    nd = need_undistort(intr, shape)
    if nd is None:
        raise RuntimeError(f"{clip_id}: the table's intrinsics are not a usable camera model")
    if not nd:
        return False, None, None
    map1, map2, _, intr_new = build_undistort_maps(shape, intr, auto=True)
    want = [head.get("pinhole_fx"), head.get("pinhole_fy"),
            head.get("pinhole_cx"), head.get("pinhole_cy")]
    if all(w is not None for w in want):
        got = [float(x) for x in intr_new[:4]]
        if not np.allclose(got, [float(w) for w in want], rtol=1e-4, atol=1e-3):
            raise RuntimeError(
                f"{clip_id}: the rebuilt pinhole {['%.4f' % g for g in got]} does not match the "
                f"published {['%.4f' % float(w) for w in want]}. The rectification this tool "
                "applies is not the one the annotations were made against.")
    return True, map1, map2


def do_clip(tables_root, source_root, out_root, clip_id, check_only):
    """Re-encode one clip and cut every segment the dataset lists for it. Returns a report."""
    annot_dir = Path(tables_root) / "original" / "annot" / clip_id
    stems = sorted(p.stem for p in annot_dir.glob("*.parquet"))
    rep = {"clip": clip_id, "segments": len(stems), "written": 0, "skipped": 0,
           "missing_source": 0, "rectified": False, "reencoded": 0, "errors": []}
    if not stems:
        rep["errors"].append(f"{clip_id}: no segment tables under {annot_dir}")
        return rep

    out_video = Path(out_root + "_chunked") / "original" / "video" / clip_id
    want = {}                                   # stem -> (start, end, rows, head)
    for stem in stems:
        start, end = (int(x) for x in stem.split("_"))
        rows, head = segment_table(annot_dir, stem)
        if end - start + 1 != rows:
            rep["errors"].append(f"{clip_id}/{stem}: the table has {rows} rows but the name spans "
                                 f"{end - start + 1} frames (start and end are inclusive)")
            continue
        if not check_only and (out_video / f"{stem}.mp4").is_file():
            rep["skipped"] += 1
            continue
        want[stem] = (start, end, rows, head)
    if not want:
        return rep

    # The table checks and the rectification cross-check come first: they need no source video,
    # and they are what --check-only exists for.
    head0 = next(iter(want.values()))[3]
    try:
        undist, map1, map2 = clip_maps(head0, clip_id)
    except Exception as e:
        rep["errors"].append(str(e))
        return rep
    rep["rectified"] = bool(undist)

    src = find_source(Path(source_root), clip_id)
    if src is None:
        rep["missing_source"] = 1
        rep["errors"].append(f"{clip_id}: no source video found under {source_root}")
        return rep

    if check_only:
        rep["written"] = len(want)
        return rep

    H, W = int(head0["height"]), int(head0["width"])
    cache = Path(out_root + "_source30") / f"{clip_id}.mp4"
    try:
        enc = source_at_table_geometry(src, cache, W, H, clip_id)
    except Exception as e:
        rep["errors"].append(str(e))
        return rep
    rep["reencoded"] = int(enc != src)
    out_video.mkdir(parents=True, exist_ok=True)
    # One decode pass. Segments may overlap, so every segment covering the current frame keeps a
    # copy; a segment is encoded and dropped as soon as its last frame goes by, which bounds
    # memory by the longest overlap rather than by the clip.
    active, pending = [], sorted(want.items(), key=lambda kv: kv[1][0])
    try:
        with av.open(str(enc)) as container:
            for idx, frame in enumerate(container.decode(video=0)):
                while pending and pending[0][1][0] == idx:
                    stem, (s, e, rows, _) = pending.pop(0)
                    active.append({"stem": stem, "end": e, "rows": rows, "frames": []})
                if not active and not pending:
                    break
                if not active:
                    continue
                im = frame.to_ndarray(format="bgr24")
                if (im.shape[0], im.shape[1]) != (H, W):
                    raise RuntimeError(
                        f"{clip_id}: decoded {im.shape[1]}x{im.shape[0]} but the tables describe "
                        f"{W}x{H}. The source file is not the one these tables index.")
                if undist:
                    im = undistort_apply(im, map1, map2)
                for buf in active:
                    buf["frames"].append(im)
                for buf in [b for b in active if b["end"] == idx]:
                    active.remove(buf)
                    _write(out_video, buf, W, H, clip_id, rep)
    except Exception as e:
        rep["errors"].append(f"{clip_id}: {e}")
        return rep
    for buf in active:
        rep["errors"].append(f"{clip_id}/{buf['stem']}: the re-encode ends at frame {idx}, before "
                             f"the segment's last frame {buf['end']}")
    return rep


def _write(out_video, buf, W, H, clip_id, rep):
    if len(buf["frames"]) != buf["rows"]:
        rep["errors"].append(f"{clip_id}/{buf['stem']}: cut {len(buf['frames'])} frames, the "
                             f"table has {buf['rows']} rows")
        return
    save_video_task(str(out_video / f"{buf['stem']}.mp4"), buf["frames"], FPS, (W, H))
    rep["written"] += 1
    buf["frames"].clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True,
                    help="the dataset's clips_chunked directory, as downloaded")
    ap.add_argument("--source", required=True,
                    help="your own copy of the source corpus, searched recursively")
    ap.add_argument("--out", required=True,
                    help="output path WITHOUT the _chunked suffix, as the stages take it")
    ap.add_argument("--clips", default=None, help="comma list, or @file with one clip id per line")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--check-only", action="store_true", help="run every check, write nothing")
    args = ap.parse_args()

    annot_root = Path(args.tables) / "original" / "annot"
    if not annot_root.is_dir():
        sys.exit(f"{annot_root} does not exist — is --tables the dataset's clips_chunked?")
    if args.clips is None:
        clips = sorted(p.name for p in annot_root.iterdir() if p.is_dir())
    elif args.clips.startswith("@"):
        clips = [ln.strip() for ln in open(args.clips[1:]) if ln.strip()]
    else:
        clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    if not clips:
        sys.exit("no clips selected")
    print(f"{len(clips)} clips, {args.jobs} at a time"
          + (" — checking only, nothing is written" if args.check_only else ""))
    totals = {"segments": 0, "written": 0, "skipped": 0, "missing_source": 0, "reencoded": 0}
    errors, rectified = [], 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(do_clip, args.tables, args.source, args.out, c, args.check_only): c
                for c in clips}
        for n, fut in enumerate(as_completed(futs), 1):
            rep = fut.result()
            for k in totals:
                totals[k] += rep[k]
            rectified += int(rep["rectified"])
            errors += rep["errors"]
            for e in rep["errors"]:
                print("  !", e)
            print(f"[{n}/{len(clips)}] {rep['clip']}: {rep['written']} of {rep['segments']}"
                  + (f", {rep['skipped']} already there" if rep["skipped"] else ""))

    verb = "would write" if args.check_only else "wrote"
    print(f"\n{verb} {totals['written']} segments of {totals['segments']}, "
          f"{totals['skipped']} already present, {len(errors)} failed")
    print(f"{rectified} of {len(clips)} clips need rectification; their geometry columns "
          "describe the rectified frame")
    if totals["missing_source"]:
        print(f"{totals['missing_source']} clips have no source video under {args.source} — "
              "every other check above still ran on their tables")
    if not args.check_only and totals["written"]:
        print(f"segments are under {args.out}_chunked/original/video/ — point the stages at "
              f"--input_dir {args.out}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

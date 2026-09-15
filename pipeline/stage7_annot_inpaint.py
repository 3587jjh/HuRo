# Stage 7: arm segmentation (Detectron2 + SAM2) and arm inpainting (ProPainter), per segment.
# Reads stage 6's segment videos and <seg>_narr.parquet tables. Writes <seg>.parquet, the same
# table with arm_mask filled, and <seg>_inpainted.mp4 beside the source video. A table that
# already carries arm_mask is inpainted with it and not segmented again.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage7_annot_inpaint.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')

import argparse, gc, glob, os, os.path as osp, shutil, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import numpy as np
import cv2
import av
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition, setup_propainter_imports, PROPAINTER_WEIGHTS_DIR
from common.camera import pinhole_from_row
from common.geometry import project_3d_kpts_to_2d
from common.io import (decode_row, decode_mask_bytes, encode_mask_bool, hand_for_side, write_table,
                       atomic_path, remove_stale_temps)
from common.video import count_frames, save_video_task
from pipeline.segmentation.detectors import DetectorDetectron2, DetectorSam2

# IO executor for async video saving
io_executor = ThreadPoolExecutor(max_workers=2)

NARR_SUFFIX = '_narr.parquet'


def _segment_stems(annot_dir):
    """Segment stems in a clip's annot directory, from <seg>_narr.parquet or <seg>.parquet."""
    stems = set()
    for name in os.listdir(annot_dir):
        if name.endswith(NARR_SUFFIX):
            stems.add(name[:-len(NARR_SUFFIX)])
        elif name.endswith('.parquet'):
            stems.add(name[:-len('.parquet')])
    return sorted(stems)


def _read_table(parquet_path, columns):
    try:
        return pq.read_table(parquet_path, columns=columns)
    except pa.ArrowInvalid as e:
        raise RuntimeError(f"{parquet_path} cannot be read: {e}") from e


def _check_video_matches(parquet_path, video_path):
    """One frame per table row, at the table's height and width."""
    table = _read_table(parquet_path, ["height", "width"])
    n_rows = table.num_rows
    size = table.slice(0, 1).to_pylist()[0]
    with av.open(video_path, "r") as reader:
        stream = reader.streams.video[0]
        video_hw = (stream.height, stream.width)
    n_frames = count_frames(video_path)
    assert n_rows == n_frames, f"{video_path} has {n_frames} frames but {parquet_path} has {n_rows} rows"
    assert video_hw == (size["height"], size["width"]), \
        f"{video_path} is {video_hw[1]}x{video_hw[0]} but {parquet_path} says {size['width']}x{size['height']}"


def _has_arm_mask(parquet_path):
    column = _read_table(parquet_path, ["arm_mask"])["arm_mask"]
    return column.null_count < len(column)


def _check_arm_mask_source(final_path, narr_path):
    """<seg>.parquet carries arm_mask, or <seg>_narr.parquet is there to compute it from."""
    assert osp.isfile(narr_path) or _has_arm_mask(final_path), \
        f"{final_path} has no arm_mask on any row. The segment needs {narr_path} to compute it " \
        f"from, or has to be removed."


def _has_inpainted(inpainted_path, video_path):
    """True when <seg>_inpainted.mp4 exists with one frame per frame of <seg>.mp4. An inpainted
    video that cannot be read or has another frame count raises and is kept."""
    if not osp.exists(inpainted_path):
        return False
    try:
        n_inpainted = count_frames(inpainted_path)
    except Exception as e:
        raise RuntimeError(f"{inpainted_path} cannot be read ({e}). "
                           f"Delete it to inpaint the segment again.") from e
    n_frames = count_frames(video_path)
    if n_inpainted != n_frames:
        raise RuntimeError(f"{inpainted_path} has {n_inpainted} frames but {video_path} has {n_frames}. "
                           f"Delete it to inpaint the segment again.")
    return True


def _mark(path):
    tmp = path + ".tmp"
    with open(tmp, "wb"):
        pass
    os.replace(tmp, path)


def _overlap_score(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return 0.0 if area1 <= 0.0 or area2 <= 0.0 else inter / min(area1, area2)


def compute_arm_masks(rows, frames, get_detector, get_sam2, min_seq_len, max_seq_len, gap_fill, where):
    """Arm masks for one segment. rows[i] and frames[i] (undistorted BGR) are frame i.
    Returns one (H,W) bool mask per frame, None where no SAM2 window covered the frame."""
    T = len(frames)
    meta = next((r for r in rows if r.get('fx') is not None), None)
    assert meta is not None, f"{where}: no row carries intrinsics"
    image_h, image_w = int(meta['height']), int(meta['width'])
    frame_shape = (image_h, image_w)
    assert frames[0].shape[:2] == frame_shape, \
        f"{where}: video frames are {frames[0].shape[1]}x{frames[0].shape[0]}, the table says {image_w}x{image_h}"
    img_fx, img_fy, img_cx, img_cy = pinhole_from_row(meta, where)

    arm_masks: List[Optional[np.ndarray]] = [None] * T

    def _extract_side_prompt(fid: int, side: int):
        """Valid hand prompt (bbox, 2D kpts, min-dist-to-edge) for `side`, or None."""
        h = hand_for_side(rows[fid], side, require_kpts3d=True)
        if h is None:
            return None
        bbox = h.get("box")
        assert bbox is not None and np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1], \
            f"{where} frame {fid}: hands[].box is {bbox}, not a filled [x1, y1, x2, y2] box"
        k2d = project_3d_kpts_to_2d(h["kpts3d"], img_fx, frame_shape, cx=img_cx, cy=img_cy, fy=img_fy)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        min_dist_to_edge = min(cx, cy, image_w - cx, image_h - cy)
        return bbox, k2d, min_dist_to_edge

    def _has_valid_hand_any_side(fid: int) -> bool:
        return not (_extract_side_prompt(fid, 0) is None and _extract_side_prompt(fid, 1) is None)

    def _register_mask(fid: int, mask_bool: np.ndarray):
        """OR-merge a mask into the frame's arm mask (sides and windows may overlap)."""
        assert mask_bool.shape == frame_shape
        m = mask_bool.astype(bool)
        arm_masks[fid] = m if arm_masks[fid] is None else np.logical_or(arm_masks[fid], m)

    # ---- overlapping-window state (shared across left/right) ----
    seq_len = max_seq_len
    overlap = seq_len // 16
    stride = seq_len - overlap
    st: Dict[str, object] = {
        "active": False, "frames": [], "left_prompts": [], "right_prompts": [],
        "next_start_idx": 0, "last_start_fid": None, "gap": 0,
    }

    def _clear_buffer():
        st["frames"].clear(); st["left_prompts"].clear(); st["right_prompts"].clear()
        st["next_start_idx"] = 0; st["last_start_fid"] = None; st["gap"] = 0

    def _start_new_sequence_if_needed():
        if not st["active"]:
            st["active"] = True; _clear_buffer()

    def _run_armseg_window(frame_ids, left_prompts, right_prompts):
        """SAM2 arm segmentation on one window: per side, seed from the max dist-to-edge
        frame (hand bbox expanded via detectron2 + wrist point), propagate bidirectionally."""
        if not frame_ids: return
        H, W = frame_shape
        imgs = [frames[fid] for fid in frame_ids]
        seeds: Dict[int, Dict[str, object]] = {}

        def _prepare_seed(side: int, prompts):
            candidates = [(idx, p) for idx, p in enumerate(prompts) if p is not None]
            if not candidates: return
            best_idx, best_prompt = max(candidates, key=lambda item: item[1][2])
            seed_hand_bbox, seed_kpts2d, _ = best_prompt
            seed_img_bgr = imgs[best_idx]
            seed_points = seed_kpts2d[0:1, :]  # (1,2) wrist
            if not np.isfinite(seed_points).all(): return

            det_bboxes, _ = get_detector().get_bboxes(seed_img_bgr)
            expanded_bbox = seed_hand_bbox.copy()
            if det_bboxes is not None and len(det_bboxes) > 0:
                scores = [_overlap_score(seed_hand_bbox, det_box) for det_box in det_bboxes]
                best_det_idx = int(np.argmax(scores))
                if float(scores[best_det_idx]) > 0.5:
                    expanded_bbox = det_bboxes[best_det_idx]
            seeds[side] = {"frame_idx": int(best_idx), "bbox": expanded_bbox, "points": seed_points}

        _prepare_seed(0, left_prompts)
        _prepare_seed(1, right_prompts)
        if not seeds: return

        masks_fwd_by_side, masks_rev_by_side = get_sam2().segment_video_from_arrays_bidir_multi(
            frames=imgs, seeds=seeds, output_bboxes=None)

        for side, masks_fwd in masks_fwd_by_side.items():
            masks_rev = masks_rev_by_side.get(side, {})
            masks_by_local: Dict[int, np.ndarray] = {}

            def _merge(segments):
                for local_idx, m_arr in segments.items():
                    assert m_arr.shape == (H, W)
                    m2d = m_arr.astype(bool)
                    masks_by_local[local_idx] = np.logical_or(masks_by_local[local_idx], m2d) \
                        if local_idx in masks_by_local else m2d
            _merge(masks_fwd)
            _merge(masks_rev)

            for local_idx, fid in enumerate(frame_ids):
                mask = masks_by_local.get(local_idx)
                if mask is None: continue
                _register_mask(fid, mask)

    def _run_armseg_if_needed():
        if not st["active"]: return
        while len(st["frames"]) - st["next_start_idx"] >= seq_len:
            s_idx = st["next_start_idx"]; e_idx = s_idx + seq_len
            _run_armseg_window(st["frames"][s_idx:e_idx],
                               st["left_prompts"][s_idx:e_idx], st["right_prompts"][s_idx:e_idx])
            st["last_start_fid"] = st["frames"][s_idx]
            st["next_start_idx"] += stride
        while len(st["frames"]) > seq_len:
            st["frames"].pop(0)
            st["left_prompts"].pop(0); st["right_prompts"].pop(0)
            if st["next_start_idx"] > 0: st["next_start_idx"] -= 1

    def _end_sequence():
        if not st["active"]: return
        n = len(st["frames"])
        if n < min_seq_len:
            st["active"] = False; _clear_buffer(); return
        if n >= seq_len:
            last_start_idx = n - seq_len
            final_first_fid = int(st["frames"][last_start_idx])
            if st["last_start_fid"] != final_first_fid:
                _run_armseg_window(
                    st["frames"][last_start_idx:last_start_idx + seq_len],
                    st["left_prompts"][last_start_idx:last_start_idx + seq_len],
                    st["right_prompts"][last_start_idx:last_start_idx + seq_len])
        else:
            _run_armseg_window(st["frames"], st["left_prompts"], st["right_prompts"])
        st["active"] = False; _clear_buffer()

    # ---- valid-hand-driven windowing over the segment's frames ----
    for fid in range(T):
        if not _has_valid_hand_any_side(fid):
            if st["active"]:
                st["gap"] += 1
                if st["gap"] > gap_fill:
                    _end_sequence()
                else:
                    st["frames"].append(fid)
                    st["left_prompts"].append(None); st["right_prompts"].append(None)
                    _run_armseg_if_needed()
            continue

        _start_new_sequence_if_needed()
        st["gap"] = 0
        st["frames"].append(fid)
        st["left_prompts"].append(_extract_side_prompt(fid, 0))
        st["right_prompts"].append(_extract_side_prompt(fid, 1))
        _run_armseg_if_needed()
    _end_sequence()
    return arm_masks


def inpaint_windows(propainter, frames, masks, window):
    """ProPainter over `frames` in windows of `window` frames overlapping by window // 8, the
    last one aligned to the end. A segment of at most `window` frames is inpainted in one call.
    A frame covered by several windows gets their mean."""
    T = len(frames)
    if T <= window:
        return propainter.infer(frames, masks, subvideo_length=80)
    stride = window - window // 8
    starts = list(range(0, T - window + 1, stride))
    if starts[-1] != T - window:
        starts.append(T - window)
    sums: List[Optional[np.ndarray]] = [None] * T
    counts = np.zeros(T, dtype=np.int64)
    for s in starts:
        out = propainter.infer(frames[s:s + window], masks[s:s + window], subvideo_length=80)
        for i, img in enumerate(out):
            img16 = img.astype(np.uint16)
            sums[s + i] = img16 if sums[s + i] is None else sums[s + i] + img16
            counts[s + i] += 1
        del out
    return [((sums[t].astype(np.int64) + counts[t] // 2) // counts[t]).astype(np.uint8)
            for t in range(T)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--min_seq_len', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--gap_fill', type=int, default=3, help='Max consecutive missing frames to bridge')
    parser.add_argument('--inpaint_window', type=int, default=720,
                        help='ProPainter window in frames. A longer segment is inpainted in '
                             'overlapping windows')
    args = parser.parse_args()
    assert args.inpaint_window >= 8, '--inpaint_window must be at least 8 frames'
    input_dir = osp.normpath(args.input_dir)

    if osp.isdir(input_dir) or osp.isdir(input_dir + '_chunked'):
        chunked_dir = osp.join(input_dir + '_chunked', 'original')
        # List the clips in stage 6's chunked output.
        clip_ids = partition(sorted(
            osp.basename(p) for p in glob.glob(osp.join(chunked_dir, 'annot', '*'))
            if osp.isdir(p)), args.part, key=lambda c: c)
    else:
        assert osp.isfile(input_dir) and input_dir.lower().endswith('.mp4'), \
            f"--input_dir {input_dir} is neither a clips directory (or a path with {input_dir}_chunked " \
            f"beside it) nor an .mp4 clip"
        chunked_dir = osp.join(osp.dirname(input_dir), 'chunked', 'original')
        clip_ids = [osp.basename(input_dir)[:-4]]
    if len(clip_ids) == 0:
        return

    video_root = osp.join(chunked_dir, 'video')
    annot_root = osp.join(chunked_dir, 'annot')

    def narr_done(clip_id):       # stage 6 finished the clip
        return osp.join(video_root, f"{clip_id}.done")

    def masks_done(clip_id):      # every segment has its <seg>.parquet
        return osp.join(annot_root, f"{clip_id}.done")

    def inpaint_done(clip_id):    # every segment has its <seg>_inpainted.mp4
        return osp.join(video_root, f"{clip_id}_inpaint.done")

    print(f'Got {len(clip_ids)} clips in total.')
    clip_ids = [c for c in clip_ids if not osp.exists(inpaint_done(c))]
    print(f'Processing {len(clip_ids)} clips (rest already done).')
    if len(clip_ids) == 0:
        return
    for clip_id in clip_ids:
        clip_annot_dir = osp.join(annot_root, clip_id)
        if osp.isdir(clip_annot_dir) and not osp.exists(narr_done(clip_id)):
            raise RuntimeError(f"{clip_annot_dir} exists but {narr_done(clip_id)} does not: stage 6 "
                               f"has not finished this clip, or its marker is missing")
    os.makedirs(video_root, exist_ok=True)
    os.makedirs(annot_root, exist_ok=True)

    models = {}

    def get_detector():
        if 'detector' not in models:
            models['detector'] = DetectorDetectron2()
        return models['detector']

    def get_sam2():
        if 'sam2' not in models:
            models['sam2'] = DetectorSam2()
        return models['sam2']

    ############################################################################################
    # 1. Arm masks: <seg>.parquet for every segment
    with torch.inference_mode():
        for clip_id in tqdm(clip_ids, desc="arm masks", unit="clip", position=1, leave=True,
                            dynamic_ncols=True, disable=args.no_tqdm):
            clip_annot_dir = osp.join(annot_root, clip_id)
            clip_video_dir = osp.join(video_root, clip_id)
            remove_stale_temps(clip_annot_dir)   # temps of processes that have exited
            remove_stale_temps(clip_video_dir)
            if osp.exists(masks_done(clip_id)):
                continue
            if not osp.isdir(clip_annot_dir):
                assert osp.exists(narr_done(clip_id)), \
                    f"{narr_done(clip_id)} is missing: stage 6 has not finished {clip_id}"
                _mark(masks_done(clip_id)); continue

            # Check every segment before any mask is copied or computed.
            stems_to_compute = []
            for stem in _segment_stems(clip_annot_dir):
                video_path = osp.join(clip_video_dir, stem + '.mp4')
                final_path = osp.join(clip_annot_dir, stem + '.parquet')
                narr_path = osp.join(clip_annot_dir, stem + NARR_SUFFIX)
                assert osp.isfile(video_path), f"{video_path} is missing"
                if osp.isfile(final_path) and _has_arm_mask(final_path):
                    _check_video_matches(final_path, video_path)
                    continue
                _check_arm_mask_source(final_path, narr_path)
                inpainted_path = osp.join(clip_video_dir, stem + '_inpainted.mp4')
                if osp.exists(inpainted_path):
                    raise RuntimeError(f"{inpainted_path} exists but there is no {final_path} with "
                                       f"arm_mask. Delete {inpainted_path} to recompute the masks "
                                       f"from {narr_path}.")
                _check_video_matches(narr_path, video_path)
                stems_to_compute.append(stem)

            for stem in tqdm(stems_to_compute, desc=f"{clip_id} masks", unit="seg",
                             position=0, leave=False, disable=args.no_tqdm):
                video_path = osp.join(clip_video_dir, stem + '.mp4')
                final_path = osp.join(clip_annot_dir, stem + '.parquet')
                narr_path = osp.join(clip_annot_dir, stem + NARR_SUFFIX)
                tmp_path = atomic_path(final_path)
                if _has_arm_mask(narr_path):
                    shutil.copyfile(narr_path, tmp_path)
                else:
                    table = pq.read_table(narr_path)
                    frame_ids = table["frame_id"].to_pylist()
                    frames = []
                    with av.open(video_path, "r") as reader:
                        for frame in reader.decode(video=0):
                            frames.append(frame.to_ndarray(format="bgr24"))
                    assert sorted(frame_ids) == list(range(len(frames))), \
                        f"{narr_path}: frame_id is not 0..{len(frames) - 1}"
                    rows = [None] * len(frames)
                    for row in table.to_pylist():
                        rows[row["frame_id"]] = decode_row(row)
                    arm_masks = compute_arm_masks(rows, frames, get_detector, get_sam2,
                                                  args.min_seq_len, args.max_seq_len, args.gap_fill,
                                                  narr_path)
                    column = pa.array([encode_mask_bool(arm_masks[fid]) if arm_masks[fid] is not None
                                       else None for fid in frame_ids], type=pa.binary())
                    table = table.set_column(table.schema.get_field_index("arm_mask"), "arm_mask", column)
                    write_table(table, tmp_path)
                os.replace(tmp_path, final_path)
            _mark(masks_done(clip_id))

    # Free the last computed segment's frames, masks and table before inpainting.
    frames = rows = table = column = arm_masks = None
    models.clear()
    gc.collect()
    torch.cuda.empty_cache()

    ############################################################################################
    # 2. Inpainting: <seg>_inpainted.mp4 from <seg>.parquet's arm_mask
    propainter = None
    with torch.inference_mode():
        for clip_id in tqdm(clip_ids, desc="clips", unit="clip", position=1, leave=True,
                             dynamic_ncols=True, disable=args.no_tqdm):
            clip_video_dir = osp.join(video_root, clip_id)
            clip_annot_dir = osp.join(annot_root, clip_id)
            remove_stale_temps(clip_annot_dir)   # temps of processes that have exited
            remove_stale_temps(clip_video_dir)
            if not osp.isdir(clip_annot_dir):
                _mark(inpaint_done(clip_id)); continue

            # Check every segment against its table before any inpainted video is looked at.
            stems = _segment_stems(clip_annot_dir)
            for stem in stems:
                video_path = osp.join(clip_video_dir, stem + '.mp4')
                final_path = osp.join(clip_annot_dir, stem + '.parquet')
                assert osp.isfile(video_path), f"{video_path} is missing"
                assert osp.isfile(final_path), \
                    f"{final_path} is missing but {masks_done(clip_id)} exists. Delete that marker " \
                    f"to compute the masks."
                _check_arm_mask_source(final_path, osp.join(clip_annot_dir, stem + NARR_SUFFIX))
                _check_video_matches(final_path, video_path)
            # Skip existing inpainted segments. One with another frame count raises.
            existing_stems = {stem for stem in stems if _has_inpainted(
                osp.join(clip_video_dir, stem + '_inpainted.mp4'), osp.join(clip_video_dir, stem + '.mp4'))}

            io_futures = []
            for seg_name in tqdm(stems, desc=f"{clip_id} segments", unit="seg",
                                 position=0, leave=False, disable=args.no_tqdm):
                if seg_name in existing_stems:
                    continue
                seg_video_path = osp.join(clip_video_dir, seg_name + ".mp4")
                output_path = osp.join(clip_video_dir, seg_name + "_inpainted.mp4")

                # frame_id -> arm_mask (None for frames no window covered, which become a zero mask below)
                table_path = osp.join(clip_annot_dir, seg_name + ".parquet")
                rows = pq.read_table(table_path, columns=["frame_id", "arm_mask"]).to_pylist()
                fid_to_arm_mask = {r["frame_id"]: (decode_mask_bytes(r["arm_mask"])
                                                   if r["arm_mask"] is not None else None)
                                   for r in rows}
                assert sorted(fid_to_arm_mask) == list(range(len(rows))), \
                    f"{table_path}: frame_id is not 0..{len(rows) - 1}"

                frames = []
                orig_h, orig_w = None, None
                with av.open(seg_video_path, "r") as reader:
                    for frame in reader.decode(video=0):
                        im_bgr = frame.to_ndarray(format="bgr24")
                        if orig_h is None:
                            orig_h, orig_w = im_bgr.shape[:2]
                        frames.append(im_bgr)
                assert len(frames) == len(rows), \
                    f"{seg_video_path} has {len(frames)} frames but its table has {len(rows)} rows"
                masks = [fid_to_arm_mask[fid] for fid in range(len(frames))]

                # Resize to a /8-divisible processing resolution (ProPainter requirement)
                proc_w = int(round(orig_w / 8) * 8)
                proc_h = int(round(orig_h / 8) * 8)
                proc_frames, proc_masks = [], []
                for img, mask in zip(frames, masks):
                    proc_frames.append(cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_CUBIC))
                    if mask is not None:
                        m = cv2.resize(mask.astype(np.uint8), (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        m = np.zeros((proc_h, proc_w), dtype=np.uint8)
                    proc_masks.append(m > 0)

                if propainter is None:
                    setup_propainter_imports()
                    from inference_propainter_custom import ProPainterInference
                    with torch.inference_mode(False):
                        propainter = ProPainterInference(model_dir=str(PROPAINTER_WEIGHTS_DIR))
                inpainted = inpaint_windows(propainter, proc_frames, proc_masks, args.inpaint_window)

                output_frames = []
                for img in inpainted:
                    if img.shape[1] != orig_w or img.shape[0] != orig_h:
                        img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                    output_frames.append(img)

                io_futures.append(io_executor.submit(save_video_task, output_path, output_frames,
                                                     30.0, (orig_w, orig_h)))
            for fut in io_futures:
                fut.result()
            _mark(inpaint_done(clip_id))


if __name__ == "__main__":
    main()

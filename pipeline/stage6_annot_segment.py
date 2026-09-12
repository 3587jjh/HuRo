# Stage 6 — arm segmentation (Detectron2 + SAM2).
# Reads stage 5's shards and writes the same rows with the per-hand hand_mask set to null, plus
# arm_mask (bit-packed) and n_person_det.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage6_annot_segment.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')

import os, os.path as osp, glob, argparse, sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np
import cv2
import torch
import av
from tqdm import tqdm

# repo root on sys.path so `common` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.geometry import project_3d_kpts_to_2d
from common.io import ParquetReader, write_clip_chunks, mark_done, hand_for_side
from pipeline.segmentation.detectors import DetectorDetectron2, DetectorSam2


def _drop_hand_mask(row):
    """Null per-hand hand_mask going forward (only stage 5 consumes it)."""
    for h in (row.get("hands") or []):
        if h is not None:
            h["hand_mask"] = None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--min_seq_len', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--chunk_size', type=int, default=5000, help='frames per parquet chunk')
    parser.add_argument('--gap_fill', type=int, default=3, help='Max consecutive missing frames to bridge')
    parser.add_argument('--pdet_stride', type=int, default=8, help='Person detection sampling stride (every N-th frame)')
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    if osp.isdir(input_dir):
        paths_mp4 = sorted(glob.glob(osp.join(input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part)
        if len(paths_mp4) == 0: return
        extr_dir = input_dir + '_extr'
        output_dir = input_dir + '_arm'
    else:
        assert osp.isfile(input_dir)
        assert input_dir.lower().endswith('.mp4')
        paths_mp4 = [input_dir]
        base_dir = osp.dirname(input_dir)
        extr_dir = osp.join(base_dir, 'extr')
        output_dir = osp.join(base_dir, 'arm')

    def is_done(path_mp4):
        clip_id = osp.basename(path_mp4)[:-4]
        return osp.exists(osp.join(output_dir, f"{clip_id}.done"))

    print(f'Got {len(paths_mp4)} mp4 files in total.')
    paths_mp4_new = [p for p in paths_mp4 if not is_done(p)]
    print(f'Skipping {len(paths_mp4) - len(paths_mp4_new)} mp4 files (already processed)')
    paths_mp4 = paths_mp4_new
    if len(paths_mp4) == 0: return
    os.makedirs(output_dir, exist_ok=True)

    detectron_detector = DetectorDetectron2()
    detector_sam = DetectorSam2()

    with torch.inference_mode():
        for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1, leave=True, dynamic_ncols=True):
            clip_id = osp.basename(path_mp4)[:-4]

            # read stage 5's shards (intrinsics + hands carried as columns)
            try: parquet_reader = ParquetReader(extr_dir, clip_id)
            except FileNotFoundError: mark_done(output_dir, clip_id); continue
            contact_ids = set(parquet_reader.get_frame_ids())
            if not contact_ids: mark_done(output_dir, clip_id); continue

            meta = parquet_reader.get(min(contact_ids))
            intr = np.array([meta['fx'], meta['fy'], meta['cx'], meta['cy'], meta['xi']], dtype=np.float64)
            image_h, image_w = int(meta['height']), int(meta['width'])
            frame_shape = (image_h, image_w)
            # raw intrinsics and intr_model carried forward (used to fill dummy gap rows below)
            clip_intr = {'fx': meta['fx'], 'fy': meta['fy'], 'cx': meta['cx'], 'cy': meta['cy'], 'xi': meta['xi'],
                         'intr_model': meta['intr_model']}

            need_undist = need_undistort(intr, frame_shape)
            if need_undist is None: mark_done(output_dir, clip_id); continue
            if need_undist:
                try: map1, map2, _, intr = build_undistort_maps(frame_shape, intr, auto=True)
                except Exception: mark_done(output_dir, clip_id); continue
            img_focal = 0.5 * (intr[0] + intr[1])
            img_cx, img_cy = float(intr[2]), float(intr[3])

            # ---- output accumulators ----
            chunk_idx = 0
            rows: List[Dict] = []
            tail_rows: "OrderedDict[int, Dict]" = OrderedDict()   # fid -> row with optional left/right_mask
            flushed_fids: set = set()
            pdet_counts: Dict[int, int] = {}

            def _extract_side_prompt(row: Dict, side: int):
                """Valid hand prompt (bbox, 2D kpts, min-dist-to-edge) for `side`, or None."""
                h = hand_for_side(row, side, require_kpts3d=True)
                if h is None:
                    return None
                bbox = h["box"]
                k2d = project_3d_kpts_to_2d(h["kpts3d"], img_focal, frame_shape, cx=img_cx, cy=img_cy)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                min_dist_to_edge = min(cx, cy, image_w - cx, image_h - cy)
                return bbox, k2d, min_dist_to_edge

            def _has_valid_hand_any_side(row: Dict) -> bool:
                return not (_extract_side_prompt(row, 0) is None and _extract_side_prompt(row, 1) is None)

            def _flush_rows_if_needed(force: bool = False):
                nonlocal rows, chunk_idx
                while rows and (force or len(rows) >= args.chunk_size):
                    out_rows = rows[:args.chunk_size]
                    write_clip_chunks(clip_id, out_rows, output_dir, chunk_idx)
                    chunk_idx += 1
                    rows = rows[args.chunk_size:]

            def _finalize_tail_row(fid: int, row: Dict):
                """Turn a tail_rows entry (optional left/right_mask) into one arm_mask row."""
                H, W = frame_shape
                left_mask = row.pop("left_mask", None)
                right_mask = row.pop("right_mask", None)
                if left_mask is None: left_mask = np.zeros((H, W), dtype=bool)
                if right_mask is None: right_mask = np.zeros((H, W), dtype=bool)
                row["arm_mask"] = np.logical_or(left_mask, right_mask)
                row["n_person_det"] = pdet_counts.get(fid)
                rows.append(row)
                flushed_fids.add(fid)
                _flush_rows_if_needed(force=False)

            def _register_side_mask(side: int, fid: int, mask_bool: np.ndarray):
                """OR-merge a per-side mask into the frame's tail row (windows may overlap)."""
                assert mask_bool.shape == frame_shape
                side_name = "left" if side == 0 else "right"
                if fid not in tail_rows:
                    row_in = parquet_reader.get(fid)
                    if row_in is not None:
                        base = _drop_hand_mask(dict(row_in))
                    else:
                        # gap frame not in parquet: a dummy row carrying the raw intrinsics and intr_model
                        base = {"clip_id": clip_id, "frame_id": fid, "height": image_h,
                                "width": image_w,
                                "hands": [], "cam_pose": None, **clip_intr}
                    base["left_mask"] = None
                    base["right_mask"] = None
                    tail_rows[fid] = base
                row = tail_rows[fid]
                existing = row[f"{side_name}_mask"]
                row[f"{side_name}_mask"] = mask_bool.astype(bool) if existing is None \
                    else np.logical_or(existing, mask_bool.astype(bool))
                # bound how many frames stay in tail_rows (FIFO finalize)
                while len(tail_rows) > args.max_seq_len:
                    oldest_fid, oldest_row = tail_rows.popitem(last=False)
                    _finalize_tail_row(oldest_fid, oldest_row)

            # ---- global overlapping-window state (shared across left/right) ----
            seq_len = args.max_seq_len
            overlap = seq_len // 16
            stride = seq_len - overlap
            global_state: Dict[str, object] = {
                "active": False, "seq_id": 0, "frames": [], "imgs": [],
                "left_prompts": [], "right_prompts": [],
                "next_start_idx": 0, "last_start_fid": None, "gap": 0,
            }

            def _clear_global_buffer():
                st = global_state
                st["frames"].clear(); st["imgs"].clear()
                st["left_prompts"].clear(); st["right_prompts"].clear()
                st["next_start_idx"] = 0; st["last_start_fid"] = None; st["gap"] = 0

            def _start_new_sequence_if_needed():
                st = global_state
                if not st["active"]:
                    st["active"] = True; st["seq_id"] += 1; _clear_global_buffer()

            def _run_armseg_if_needed():
                st = global_state
                if not st["active"]: return
                while len(st["frames"]) - st["next_start_idx"] >= seq_len:
                    s_idx = st["next_start_idx"]; e_idx = s_idx + seq_len
                    _run_armseg_window(st["frames"][s_idx:e_idx], st["imgs"][s_idx:e_idx],
                                       st["left_prompts"][s_idx:e_idx], st["right_prompts"][s_idx:e_idx])
                    st["last_start_fid"] = st["frames"][s_idx]
                    st["next_start_idx"] += stride
                while len(st["frames"]) > seq_len:
                    st["frames"].pop(0); st["imgs"].pop(0)
                    st["left_prompts"].pop(0); st["right_prompts"].pop(0)
                    if st["next_start_idx"] > 0: st["next_start_idx"] -= 1

            def _end_sequence():
                st = global_state
                if not st["active"]: return
                n = len(st["frames"])
                if n < args.min_seq_len:
                    st["active"] = False; _clear_global_buffer(); return
                if n >= seq_len:
                    last_start_idx = n - seq_len
                    final_first_fid = int(st["frames"][last_start_idx])
                    if st["last_start_fid"] != final_first_fid:
                        _run_armseg_window(
                            st["frames"][last_start_idx:last_start_idx + seq_len],
                            st["imgs"][last_start_idx:last_start_idx + seq_len],
                            st["left_prompts"][last_start_idx:last_start_idx + seq_len],
                            st["right_prompts"][last_start_idx:last_start_idx + seq_len])
                else:
                    _run_armseg_window(st["frames"], st["imgs"], st["left_prompts"], st["right_prompts"])
                st["active"] = False; _clear_global_buffer()

            def _run_armseg_window(frame_ids, imgs, left_prompts, right_prompts):
                """SAM2 arm segmentation on one window: per side, seed from the max dist-to-edge
                frame (hand bbox expanded via detectron2 + wrist point), propagate bidirectionally."""
                if not frame_ids: return
                H, W = frame_shape
                seeds: Dict[int, Dict[str, object]] = {}

                def _prepare_seed(side: int, prompts):
                    candidates = [(idx, p) for idx, p in enumerate(prompts) if p is not None]
                    if not candidates: return
                    best_idx, best_prompt = max(candidates, key=lambda item: item[1][2])
                    seed_hand_bbox, seed_kpts2d, _ = best_prompt
                    seed_img_bgr = imgs[best_idx]
                    seed_points = seed_kpts2d[0:1, :]  # (1,2) wrist
                    if not np.isfinite(seed_points).all(): return

                    det_bboxes, _ = detectron_detector.get_bboxes(seed_img_bgr)
                    expanded_bbox = seed_hand_bbox.copy()
                    if det_bboxes is not None and len(det_bboxes) > 0:
                        def _overlap_score(box1, box2):
                            x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
                            x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
                            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
                            area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
                            return 0.0 if area1 <= 0.0 or area2 <= 0.0 else inter / min(area1, area2)
                        scores = [_overlap_score(seed_hand_bbox, det_box) for det_box in det_bboxes]
                        best_det_idx = int(np.argmax(scores))
                        if float(scores[best_det_idx]) > 0.5:
                            expanded_bbox = det_bboxes[best_det_idx]
                    seeds[side] = {"frame_idx": int(best_idx), "bbox": expanded_bbox, "points": seed_points}

                _prepare_seed(0, left_prompts)
                _prepare_seed(1, right_prompts)
                if not seeds: return

                masks_fwd_by_side, masks_rev_by_side = detector_sam.segment_video_from_arrays_bidir_multi(
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
                        _register_side_mask(side, fid, mask)

            # ---- sequential decode + valid-hand-driven windowing ----
            with av.open(path_mp4, "r") as reader:
                s = reader.streams.video[0]
                total = s.frames or (int(float(s.duration * s.time_base) * 30 + 0.5)
                                     if s.duration and s.time_base else None)
                for frame_id, frame in enumerate(tqdm(reader.decode(video=0), total=total, desc=f"{clip_id}",
                        unit="frame", position=0, leave=False, dynamic_ncols=True, disable=args.no_tqdm)):
                    if frame_id == 0:
                        im_bgr0 = frame.to_ndarray(format="bgr24")
                        assert frame_shape == im_bgr0.shape[:2]
                    if frame_id % args.pdet_stride == 0:
                        pdet_img = im_bgr0 if frame_id == 0 else frame.to_ndarray(format="bgr24")
                        if need_undist: pdet_img = undistort_apply(pdet_img, map1, map2)
                        pdet_bboxes, _ = detectron_detector.get_bboxes(pdet_img)
                        pdet_counts[frame_id] = len(pdet_bboxes) if pdet_bboxes is not None else 0

                    # classify frame: valid (has a usable hand) vs gap
                    is_gap = False
                    row = None
                    if frame_id not in contact_ids:
                        is_gap = True
                    else:
                        row = parquet_reader.get(frame_id)
                        if not _has_valid_hand_any_side(row): is_gap = True

                    if is_gap:
                        st = global_state
                        if st["active"]:
                            st["gap"] += 1
                            if st["gap"] > args.gap_fill:
                                _end_sequence()
                            else:
                                im_bgr = frame.to_ndarray(format="bgr24")
                                if need_undist: im_bgr = undistort_apply(im_bgr, map1, map2)
                                st["frames"].append(frame_id); st["imgs"].append(im_bgr)
                                st["left_prompts"].append(None); st["right_prompts"].append(None)
                                _run_armseg_if_needed()
                        continue

                    im_bgr = frame.to_ndarray(format="bgr24")
                    if need_undist: im_bgr = undistort_apply(im_bgr, map1, map2)
                    _start_new_sequence_if_needed()
                    st = global_state
                    st["gap"] = 0
                    st["frames"].append(frame_id); st["imgs"].append(im_bgr)
                    st["left_prompts"].append(_extract_side_prompt(row, 0))
                    st["right_prompts"].append(_extract_side_prompt(row, 1))
                    _run_armseg_if_needed()
                _end_sequence()

            # finalize remaining tail rows, then write null-mask rows for the rest
            for fid, row in list(tail_rows.items()):
                _finalize_tail_row(fid, row)
            tail_rows.clear()

            for fid in sorted(contact_ids):
                if fid in flushed_fids: continue
                row = _drop_hand_mask(dict(parquet_reader.get(fid)))
                row["arm_mask"] = None
                row["n_person_det"] = pdet_counts.get(fid)
                rows.append(row)
                _flush_rows_if_needed(force=False)
            _flush_rows_if_needed(force=True)
            mark_done(output_dir, clip_id)


if __name__ == "__main__":
    main()

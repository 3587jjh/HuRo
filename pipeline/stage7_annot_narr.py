# Stage 7 — narration (Qwen3.5 VLM) + chunking into per-segment videos and parquets.
# Reads stage 6's shards and writes per-segment undistorted videos + parquets (frame_id reset,
# narr and the merged `language` instruction added).
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage7_annot_narr.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional
from pathlib import Path
import torch
import glob, av, os, os.path as osp, sys, argparse, numpy as np, cv2
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.video import save_video_task, validate_segment_outputs
from common.io import merge_narration, ParquetReader, rows_to_parquet, hand_for_side
from pipeline.captioning.caption import load_vlm_model, get_caption
from pipeline.captioning.tracker import RejectionTracker

io_executor = ThreadPoolExecutor(max_workers=2)


#############################################################################################
# Hand Marker Drawing
#############################################################################################

def _invert_T(T):
    """Invert a (4,4) SE3 matrix."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=8, gap_len=6):
    """Draw a dashed line from pt1 to pt2."""
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = (dx**2 + dy**2) ** 0.5
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    drawn = 0.0
    while drawn < dist:
        seg_end = min(drawn + dash_len, dist)
        p1 = (int(round(pt1[0] + ux * drawn)), int(round(pt1[1] + uy * drawn)))
        p2 = (int(round(pt1[0] + ux * seg_end)), int(round(pt1[1] + uy * seg_end)))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        drawn = seg_end + gap_len


def draw_hand_trajectory(
    frame: np.ndarray,
    left_traj_2d: Optional[List[np.ndarray]],
    left_cur: int,
    left_gap_mask: Optional[List[bool]],
    left_break_mask: Optional[List[bool]],
    right_traj_2d: Optional[List[np.ndarray]],
    right_cur: int,
    right_gap_mask: Optional[List[bool]],
    right_break_mask: Optional[List[bool]],
    dot_ratio: float = 0.022,
    line_ratio: float = 0.007,
    left_raw_idx: Optional[List[int]] = None,
    right_raw_idx: Optional[List[int]] = None,
) -> np.ndarray:
    """Draw hand trajectory overlays (left=green, right=blue, square=start, ring=current).
    gap_mask points draw dashed, break_mask points draw no line, and cur < 0 draws no current marker."""
    img = frame.copy()
    H, W = img.shape[:2]

    dot_radius = max(4, int(round(dot_ratio * min(H, W))) // 2)
    line_thickness = max(2, int(round(line_ratio * min(H, W))))
    # Trajectory is at raw 30fps density, so ~500ms ≈ 15 raw frames
    waypoint_interval = 15

    for traj, cur_pos, gap_mask, break_mask, color, raw_idx in [
        (left_traj_2d, left_cur, left_gap_mask, left_break_mask, (0, 200, 0), left_raw_idx),
        (right_traj_2d, right_cur, right_gap_mask, right_break_mask, (255, 0, 0), right_raw_idx),
    ]:
        if not traj or len(traj) < 1:
            continue
        draw_end = cur_pos if (cur_pos >= 0 and cur_pos < len(traj)) else len(traj) - 1

        # 1. Trajectory lines (solid, dashed, or broken)
        if cur_pos >= 0 and cur_pos < len(traj):
            _clip_r = max(dot_radius + 2, int(dot_radius * 1.6))
        else:
            _clip_r = 0
        for i in range(1, draw_end + 1):
            if break_mask and break_mask[i]:
                continue  # segment break: don't draw any line
            pt1 = (int(round(traj[i - 1][0])), int(round(traj[i - 1][1])))
            pt2 = (int(round(traj[i][0])), int(round(traj[i][1])))
            # Clip line endpoint at bullseye edge so it doesn't poke into the ring
            if i == cur_pos and _clip_r > 0:
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist > _clip_r + 1:
                    ratio = (_clip_r + 1) / dist
                    pt2 = (int(round(pt2[0] - dx * ratio)), int(round(pt2[1] - dy * ratio)))
            is_gap = gap_mask and (gap_mask[i - 1] or gap_mask[i])
            if is_gap:
                _draw_dashed_line(img, pt1, pt2, color, line_thickness)
            else:
                cv2.line(img, pt1, pt2, color, line_thickness, cv2.LINE_AA)

        # 2. Start marker: filled square at first point + after each break
        segment_starts = [0]
        if break_mask:
            segment_starts += [i for i in range(1, draw_end + 1) if break_mask[i]]
        for si in segment_starts:
            sq_half = max(3, int(dot_radius * 0.55))
            pt_s = (int(round(traj[si][0])), int(round(traj[si][1])))
            cv2.rectangle(img,
                          (pt_s[0] - sq_half - 1, pt_s[1] - sq_half - 1),
                          (pt_s[0] + sq_half + 1, pt_s[1] + sq_half + 1),
                          (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img,
                          (pt_s[0] - sq_half, pt_s[1] - sq_half),
                          (pt_s[0] + sq_half, pt_s[1] + sq_half),
                          color, -1, cv2.LINE_AA)

        # 3. Waypoint markers: graduated circles every ~500ms (15 raw frames), per-segment
        if draw_end > 0:
            seg_starts = [0]
            if break_mask:
                seg_starts += [i for i in range(1, draw_end + 1) if break_mask[i]]
            seg_ends = seg_starts[1:] + [draw_end + 1]

            wp_indices = []
            wp_seg_id = []
            for si_idx, (seg_s, seg_e) in enumerate(zip(seg_starts, seg_ends)):
                seg_e = min(seg_e, draw_end + 1)
                if raw_idx is not None:
                    base_raw = raw_idx[seg_s]
                    next_wp_threshold = waypoint_interval
                    for j in range(seg_s + 1, seg_e):
                        if raw_idx[j] - base_raw >= next_wp_threshold:
                            wp_indices.append(j)
                            wp_seg_id.append(si_idx)
                            next_wp_threshold += waypoint_interval
                else:
                    for j in range(seg_s + waypoint_interval, seg_e, waypoint_interval):
                        wp_indices.append(j)
                        wp_seg_id.append(si_idx)

            seg_wp_counts = {}
            for si in wp_seg_id:
                seg_wp_counts[si] = seg_wp_counts.get(si, 0) + 1
            seg_wp_counter = {}
            for j, si in zip(wp_indices, wp_seg_id):
                seg_wp_counter[si] = seg_wp_counter.get(si, 0) + 1
                t = seg_wp_counter[si] / (seg_wp_counts[si] + 1)
                r = max(2, int(dot_radius * (0.3 + 0.35 * t)))
                pt_wp = (int(round(traj[j][0])), int(round(traj[j][1])))
                cv2.circle(img, pt_wp, r + 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(img, pt_wp, r, color, -1, cv2.LINE_AA)

        # 4. Current marker (hollow ring + center dot): only if hand is observed (cur_pos >= 0)
        if cur_pos >= 0 and cur_pos < len(traj):
            pt_cur = (int(round(traj[cur_pos][0])), int(round(traj[cur_pos][1])))
            ring_r = max(dot_radius + 1, int(dot_radius * 1.3))
            ring_thick = max(2, line_thickness)
            center_r = max(2, int(dot_radius * 0.35))
            cv2.circle(img, pt_cur, ring_r + 1, (0, 0, 0), ring_thick + 2, cv2.LINE_AA)
            cv2.circle(img, pt_cur, ring_r, color, ring_thick, cv2.LINE_AA)
            cv2.circle(img, pt_cur, center_r + 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, pt_cur, center_r, color, -1, cv2.LINE_AA)

    return img


def _draw_frame_label(img: np.ndarray, label: str) -> np.ndarray:
    """Draw frame label (e.g. 'Frame 1') on top-left corner with semi-transparent background."""
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(H, W) / 800.0)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = int(scale * 6)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + 2 * pad + baseline), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, label, (pad, th + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img



#############################################################################################
# Multi-person detection filter
#############################################################################################

def _is_multi_person_segment(parquet_reader, start_fid, end_fid, n_thresh, frame_thresh):
    """Check if segment has multi-person contamination.
    Returns True if >= frame_thresh frames have n_person_det >= n_thresh."""
    flagged = 0
    for fid in range(start_fid, end_fid + 1):
        row = parquet_reader.get(fid)
        if row is None:
            continue
        n_det = row.get('n_person_det')
        if n_det is not None and n_det >= n_thresh:
            flagged += 1
            if flagged >= frame_thresh:
                return True
    return False


#############################################################################################
# Main
#############################################################################################

def main():
    ############################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--min_seq_len', type=int, default=50, help='Min output segment length in raw frames (30fps)')
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--no_tqdm', action='store_true')
    # additional parameters for VLM
    parser.add_argument('--stride', type=int, default=3, help='Take every stride-th frame for VLM (effective fps = 30/stride)')
    parser.add_argument('--n_caption_samples', type=int, default=2, help='Number of VLM caption samples per batch')
    parser.add_argument('--n_caption_batches', type=int, default=2, help='Number of batched VLM predictions (total samples = n_caption_samples * n_caption_batches)')
    parser.add_argument('--uniform_max_seq_len', type=int, default=210, help='Window size for uniform sampling')
    parser.add_argument('--uniform_overlap_ratio', type=float, default=0.50, help='Overlap ratio between windows')
    parser.add_argument('--traj_dot_ratio', type=float, default=0.018, help='Dot diameter as ratio of min(H,W)')
    parser.add_argument('--traj_line_ratio', type=float, default=0.006, help='Line thickness as ratio of min(H,W)')
    parser.add_argument('--uniform_overlay_len', type=int, default=18, help='Max past raw frames (30fps) used for trajectory overlay per frame')
    parser.add_argument('--gap_fill', type=int, default=3, help='Max consecutive missing raw frames (30fps) to bridge when building contiguous contact sequences')
    parser.add_argument('--traj_gap_fill', type=int, default=3, help='Max consecutive missing raw frames (30fps) to interpolate in trajectory overlay (world-coord interpolation for visualization)')
    parser.add_argument('--pdet_n_thresh', type=int, default=3, help='Person detection count threshold per frame (0 to disable)')
    parser.add_argument('--pdet_frame_thresh', type=int, default=2, help='Min flagged frames to reject a segment')
    args = parser.parse_args()

    assert args.min_seq_len <= args.uniform_max_seq_len
    assert args.uniform_overlay_len <= args.uniform_max_seq_len

    tracker = RejectionTracker(enabled=True)

    input_dir = osp.normpath(args.input_dir)
    if osp.isdir(input_dir):
        paths_mp4 = sorted(glob.glob(osp.join(input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part)
        if len(paths_mp4) == 0: return
        arm_dir = input_dir + '_arm'
        output_video_dir = osp.join(input_dir + '_chunked', 'original', 'video')
        output_annot_dir = osp.join(input_dir + '_chunked', 'original', 'annot')
    else:
        assert osp.isfile(input_dir)
        assert input_dir.lower().endswith('.mp4')
        paths_mp4 = [input_dir]
        base_dir = osp.dirname(input_dir)
        arm_dir = osp.join(base_dir, 'arm')
        output_video_dir = osp.join(base_dir, 'chunked', 'original', 'video')
        output_annot_dir = osp.join(base_dir, 'chunked', 'original', 'annot')

    ############################################################################################
    def is_done(path_mp4):
        clip_id = osp.basename(path_mp4)[:-4]
        return osp.exists(osp.join(output_video_dir, f"{clip_id}.done"))
        
    def mark_done(clip_id):
        done = osp.join(output_video_dir, f"{clip_id}.done")
        tmp = done + ".tmp"
        with open(tmp, "wb"): pass
        os.replace(tmp, done)

    print(f'Got {len(paths_mp4)} mp4 files in total.')
    paths_mp4_new = []
    for path_mp4 in paths_mp4:
        if not is_done(path_mp4): 
            paths_mp4_new.append(path_mp4)
    print(f'Skipping {len(paths_mp4) - len(paths_mp4_new)} mp4 files (already processed)')
    paths_mp4 = paths_mp4_new
    if len(paths_mp4) == 0: return
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_annot_dir, exist_ok=True)

    print("Loading VLM model...")
    load_vlm_model()
    print("VLM model loaded.")

    ############################################################################################
    with torch.inference_mode():
        for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1, leave=True, dynamic_ncols=True):
            clip_id = osp.basename(path_mp4)[:-4]

            # early-read stage 6 (arm) shards
            try: parquet_reader = ParquetReader(arm_dir, clip_id)
            except FileNotFoundError: mark_done(clip_id); continue
            contact_ids = parquet_reader.get_frame_ids()
            if not contact_ids: mark_done(clip_id); continue
            max_contact_id = parquet_reader.get_max_frame_id()
                
            ############################################################################################
            # 1. refine contact sequences
            contact_ids = [frame_id for frame_id in contact_ids
                           if (row := parquet_reader.get(frame_id)) is not None
                           and row.get('cam_pose') is not None
                           and any(h.get('kpts3d') is not None for h in row.get('hands', []))]
            if not contact_ids: mark_done(clip_id); continue

            # Build contiguous contact sequences (no length filter here, min_seq_len applies later)
            sequences_1 = [] # [[start0, end0], ...]
            start_id = contact_ids[0]
            prev_id = contact_ids[0]

            for i in range(1, len(contact_ids)):
                curr_id = contact_ids[i]
                if curr_id <= prev_id + 1 + args.gap_fill:
                    prev_id = curr_id
                else:
                    sequences_1.append([start_id, prev_id])
                    start_id = curr_id
                    prev_id = curr_id
            sequences_1.append([start_id, prev_id])
            if not sequences_1: mark_done(clip_id); continue

            sequences = [[s, e] for s, e in sequences_1 if e - s + 1 >= args.min_seq_len]
            if not sequences: mark_done(clip_id); continue

            # Multi-person detection hard gate
            if args.pdet_n_thresh > 0:
                sequences = [
                    [s, e] for s, e in sequences
                    if not _is_multi_person_segment(parquet_reader, s, e,
                                                     args.pdet_n_thresh, args.pdet_frame_thresh)
                ]
                if not sequences: mark_done(clip_id); continue

            ############################################################################################
            # 2. Intrinsics (from stage-6 carried columns) + undistortion
            _meta = parquet_reader.get(parquet_reader.get_frame_ids()[0])
            intr = np.array([_meta['fx'], _meta['fy'], _meta['cx'], _meta['cy'], _meta['xi']], dtype=np.float64)
            image_h, image_w = int(_meta['height']), int(_meta['width'])
            frame_shape = (image_h, image_w)

            need_undist = need_undistort(intr, frame_shape)
            assert need_undist is not None
            if need_undist:
                try: map1, map2, _, intr = build_undistort_maps(frame_shape, intr, auto=True)
                except Exception: print('unexpected logic'); sys.exit(1)
            img_fx, img_fy, img_cx, img_cy = intr[0], intr[1], intr[2], intr[3]

            def _extract_hand_wrist_3d(row: Dict, side: int) -> Optional[np.ndarray]:
                """Wrist position (kpts3d[0]) for `side` (0=left, 1=right), or None."""
                h = hand_for_side(row, side, require_kpts3d=True)
                return h["kpts3d"][0].astype(np.float64) if h is not None else None

            def _compute_traj_2d(current_idx, hand_centers_3d, cam_poses, overlay_len=None):
                """Project past hand trajectory into current_idx's camera (gaps interpolated in world coords).
                Returns (traj_2d, traj_raw_idx, current_pos, gap_mask, break_mask). current_pos=-1 if hand invalid,
                gap_mask=interpolated points, break_mask=segment break before the point (no connecting line)."""
                if cam_poses[current_idx] is None:
                    return None, None, -1, None, None
                H, W = frame_shape
                margin = max(H, W) * 0.5
                T_ref_inv = _invert_T(cam_poses[current_idx])
                j_start = max(0, current_idx - overlay_len + 1) if overlay_len else 0

                raw = []  # list of (j, p_world_or_None)
                for j in range(j_start, current_idx + 1):
                    p_cam = hand_centers_3d[j]
                    cam = cam_poses[j]
                    if p_cam is not None and cam is not None:
                        p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
                        p_world = (cam @ p_h)[:3]
                        raw.append((j, p_world))
                    else:
                        raw.append((j, None))
                valid_mask = [p is not None for (_, p) in raw]

                first_valid = next((k for k in range(len(raw)) if valid_mask[k]), None)
                if first_valid is None:
                    return None, None, -1, None, None
                last_valid = next((k for k in range(len(raw) - 1, -1, -1) if valid_mask[k]), None)
                raw = raw[first_valid:last_valid + 1]
                valid_mask = valid_mask[first_valid:last_valid + 1]

                # Interpolate short gaps. Gaps > traj_gap_fill become segment breaks
                interp_world = [None] * len(raw)
                is_gap = [False] * len(raw)
                is_long_gap = [False] * len(raw)
                for k, (j, p_w) in enumerate(raw):
                    if valid_mask[k]:
                        interp_world[k] = p_w
                    else:
                        before = next((kb for kb in range(k - 1, -1, -1) if valid_mask[kb]), None)
                        after = next((ka for ka in range(k + 1, len(raw)) if valid_mask[ka]), None)
                        if before is not None and after is not None:
                            gap_len = after - before - 1
                            if gap_len <= args.traj_gap_fill:
                                t = (k - before) / (after - before)
                                interp_world[k] = raw[before][1] * (1 - t) + raw[after][1] * t
                                is_gap[k] = True
                            else:
                                is_long_gap[k] = True
                        else:
                            is_long_gap[k] = True

                traj = []
                traj_raw_idx_out = []
                gap_mask_out = []
                break_mask_out = []
                current_pos = -1
                prev_was_long_gap = False
                for k, (j, _) in enumerate(raw):
                    p_w = interp_world[k]
                    if p_w is None:
                        prev_was_long_gap = prev_was_long_gap or is_long_gap[k]
                        continue
                    p_in_ref = (T_ref_inv @ np.array([p_w[0], p_w[1], p_w[2], 1.0], dtype=np.float64))[:3]
                    if p_in_ref[2] <= 1e-6:
                        prev_was_long_gap = True  # projection failure acts like a break
                        continue
                    u = img_fx * p_in_ref[0] / p_in_ref[2] + img_cx
                    v = img_fy * p_in_ref[1] / p_in_ref[2] + img_cy
                    if u < -margin or u > W + margin or v < -margin or v > H + margin:
                        prev_was_long_gap = True
                        continue
                    if j == current_idx and hand_centers_3d[current_idx] is not None:
                        current_pos = len(traj)
                    traj.append(np.array([u, v]))
                    traj_raw_idx_out.append(j)
                    gap_mask_out.append(is_gap[k])
                    break_mask_out.append(prev_was_long_gap and len(traj) > 1)
                    prev_was_long_gap = False

                return (traj, traj_raw_idx_out, current_pos, gap_mask_out, break_mask_out) if traj else (None, None, -1, None, None)

            ############################################################################################
            # 3. Subdivide sequences into overlapping VLM windows
            windows_to_process = []  # [(start_fid, end_fid), ...]

            for seq_start, seq_end in sequences:
                window_len = args.uniform_max_seq_len
                window_stride = max(1, int(window_len * (1 - args.uniform_overlap_ratio)))
                seq_len = seq_end - seq_start + 1
                if seq_len <= window_len:
                    windows_to_process.append((seq_start, seq_end))
                else:
                    # full sliding windows + a final window aligned to the sequence end
                    w_start = seq_start
                    while w_start + window_len - 1 <= seq_end:
                        windows_to_process.append((w_start, w_start + window_len - 1))
                        w_start += window_stride
                    final_start = seq_end - window_len + 1
                    if not windows_to_process or windows_to_process[-1][0] != final_start:
                        windows_to_process.append((final_start, seq_end))
            if not windows_to_process: mark_done(clip_id); continue

            ############################################################################################
            # 4. Process each window
            seg_video_dir = osp.join(output_video_dir, clip_id)
            seg_annot_dir = osp.join(output_annot_dir, clip_id)
            os.makedirs(seg_video_dir, exist_ok=True)
            os.makedirs(seg_annot_dir, exist_ok=True)

            # Collect valid existing segment stems to skip re-processing
            existing_stems = set()
            video_stems = {f[:-4] for f in os.listdir(seg_video_dir) if f.endswith('.mp4') and not f.endswith('_inpainted.mp4')} if osp.isdir(seg_video_dir) else set()
            annot_stems = {f[:-8] for f in os.listdir(seg_annot_dir) if f.endswith('.parquet')} if osp.isdir(seg_annot_dir) else set()
            for stem in video_stems | annot_stems:
                ap = osp.join(seg_annot_dir, stem + '.parquet')
                vp = osp.join(seg_video_dir, stem + '.mp4')
                if validate_segment_outputs(annot_path=ap, video_path=vp):
                    existing_stems.add(stem)

            # Sort windows by start frame for streaming
            windows_to_process.sort(key=lambda x: x[0])

            io_futures = []

            def _process_window(frame_ids, imgs, left_centers_3d, right_centers_3d, cam_poses):
                """Process one window: VLM captioning, then save the chunked video + parquet."""
                if not frame_ids:
                    return

                effective_fps = 30.0 / args.stride

                # Trim leading/trailing dead zones on RAW frames (not subsampled)
                valid_raw = [i for i in range(len(frame_ids))
                             if cam_poses[i] is not None
                             and (left_centers_3d[i] is not None or right_centers_3d[i] is not None)]
                if not valid_raw:
                    return
                trim_start = valid_raw[0]
                trim_end = valid_raw[-1]

                indices = list(range(trim_start, trim_end + 1, args.stride))
                if not indices:
                    return

                # min_seq_len check before spending VLM compute
                if trim_end - trim_start + 1 < args.min_seq_len:
                    return

                sampled_frames_with_markers = []
                for frame_num, idx in enumerate(indices, 1):
                    if cam_poses[idx] is None:
                        # No cam_pose: send raw frame (no overlay)
                        frame_out = imgs[idx].copy()
                    else:
                        left_traj, left_raw_idx, left_cur, left_gap, left_brk = _compute_traj_2d(
                            idx, left_centers_3d, cam_poses, args.uniform_overlay_len)
                        right_traj, right_raw_idx, right_cur, right_gap, right_brk = _compute_traj_2d(
                            idx, right_centers_3d, cam_poses, args.uniform_overlay_len)
                        frame_out = draw_hand_trajectory(
                            imgs[idx],
                            left_traj, left_cur, left_gap, left_brk,
                            right_traj, right_cur, right_gap, right_brk,
                            dot_ratio=args.traj_dot_ratio,
                            line_ratio=args.traj_line_ratio,
                            left_raw_idx=left_raw_idx,
                            right_raw_idx=right_raw_idx,
                        )
                    _draw_frame_label(frame_out, f"Frame {frame_num}")
                    sampled_frames_with_markers.append(frame_out)

                # Convert BGR to RGB for VLM
                markers_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled_frames_with_markers]
                win_str = f"{frame_ids[trim_start]:06d}-{frame_ids[trim_end]:06d}"
                narr = get_caption(
                    markers_rgb,
                    n_samples=args.n_caption_samples,
                    n_batches=args.n_caption_batches,
                    tracker=tracker, clip_id=clip_id, window=win_str,
                    fps=effective_fps,
                )

                if narr is None:
                    return
                # One instruction for the segment. None means nothing usable was said
                language = merge_narration(narr)
                if language is None:
                    return

                frame_ids = frame_ids[trim_start:trim_end + 1]
                imgs = imgs[trim_start:trim_end + 1]

                segment_rows = []
                start_fid = frame_ids[0]
                for fid in frame_ids:
                    row = parquet_reader.get(fid)
                    if row is not None:
                        row = dict(row)
                    else:
                        # Gap frame not in parquet: construct minimal row
                        row = {
                            "clip_id": clip_id,
                            "height": frame_shape[0],
                            "width": frame_shape[1],
                            "intr_model": _meta['intr_model'],
                            "hands": [],
                            "arm_mask": None,
                            "cam_pose": None,
                        }
                    # Reset frame_id to be 0-indexed relative to this segment
                    row["frame_id"] = fid - start_fid
                    row["narr"] = narr
                    row["language"] = language
                    segment_rows.append(row)

                actual_start = frame_ids[0]
                actual_end = frame_ids[-1]
                file_name = f"{actual_start:06d}_{actual_end:06d}"

                if file_name in existing_stems:
                    return  # Already validated, skip

                h, w = imgs[0].shape[:2]
                video_path = osp.join(seg_video_dir, file_name + ".mp4")
                io_futures.append(io_executor.submit(save_video_task, video_path, list(imgs), 30.0, (w, h)))

                parquet_path = osp.join(seg_annot_dir, file_name + ".parquet")
                io_futures.append(io_executor.submit(rows_to_parquet, segment_rows, parquet_path))

            ############################################################################################
            # 5. Stream through video. Each frame is dispatched to ALL active windows containing it
            active_buffers = []
            next_window_idx = 0
            
            with av.open(path_mp4, "r") as reader:
                s = reader.streams.video[0]
                total = s.frames or (int(float(s.duration * s.time_base) * 30 + 0.5) if s.duration and s.time_base else None) # assume 30 fps
                
                for frame_id, frame in enumerate(tqdm(reader.decode(video=0), total=total, desc=f"{clip_id}", 
                    unit="frame", position=0, leave=False, dynamic_ncols=True, disable=args.no_tqdm)):
                    
                    if frame_id == 0:
                        assert frame_shape == (frame.height, frame.width)
                    
                    if next_window_idx >= len(windows_to_process) and not active_buffers:
                        break

                    while next_window_idx < len(windows_to_process):
                        w_start, w_end = windows_to_process[next_window_idx]
                        if w_start <= frame_id:
                            active_buffers.append({
                                'win_idx': next_window_idx,
                                'w_start': w_start,
                                'w_end': w_end,
                                'fids': [],
                                'imgs': [],
                                'left3d': [],
                                'right3d': [],
                                'cam_poses': [],
                            })
                            next_window_idx += 1
                        else:
                            break
                    
                    completed = [buf for buf in active_buffers if frame_id > buf['w_end']]
                    for buf in completed:
                        active_buffers.remove(buf)
                        if buf['fids']:
                            _process_window(buf['fids'], buf['imgs'], buf['left3d'], buf['right3d'], buf['cam_poses'])

                    if not active_buffers:
                        continue

                    if not any(buf['w_start'] <= frame_id <= buf['w_end'] for buf in active_buffers):
                        continue

                    im_bgr = frame.to_ndarray(format="bgr24")
                    if need_undist:
                        im_bgr = undistort_apply(im_bgr, map1, map2)

                    row = parquet_reader.get(frame_id)
                    left_c3d = _extract_hand_wrist_3d(row, 0) if row else None
                    right_c3d = _extract_hand_wrist_3d(row, 1) if row else None
                    cam_pose_raw = row['cam_pose'] if row else None
                    cam_pose_frame = cam_pose_raw.astype(np.float64) if cam_pose_raw is not None else None

                    for buf in active_buffers:
                        if buf['w_start'] <= frame_id <= buf['w_end']:
                            buf['fids'].append(frame_id)
                            buf['imgs'].append(im_bgr)
                            buf['left3d'].append(left_c3d)
                            buf['right3d'].append(right_c3d)
                            buf['cam_poses'].append(cam_pose_frame)

                # Flush remaining active buffers at end of video
                for buf in active_buffers:
                    if buf['fids']:
                        _process_window(buf['fids'], buf['imgs'], buf['left3d'], buf['right3d'], buf['cam_poses'])

            # Wait for all async saves to complete before marking done
            for fut in io_futures:
                fut.result()
            # Save the full caption trace log for cross-clip aggregation
            caption_log_path = osp.join(seg_annot_dir, "_caption_log.json")
            with open(caption_log_path, "w") as f:
                json.dump(tracker.to_dict(clip_id), f, indent=2)
            tracker.reset()
            mark_done(clip_id)

if __name__ == "__main__":
    main()

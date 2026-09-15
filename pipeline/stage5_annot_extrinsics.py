# Stage 5: camera extrinsics (DROID-SLAM + MoGe-2 + GeoCalib).
# Reads stage 4's shards and fills cam_pose = metric, gravity-aligned cam-to-world in the
# OpenCV convention (Z-forward, Y-down). Only valid frames get one.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage5_annot_extrinsics.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')

import math
import os, os.path as osp, glob, argparse, sys
from pathlib import Path
import numpy as np
import cv2
import torch
import av
from tqdm import tqdm

# repo root on sys.path so `common` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (partition, mp4_clip_id, setup_extrinsics_imports, DROID_WEIGHTS,
                          MOGE2_REPO, MOGE2_WEIGHTS, MOGE2_WEIGHTS_DIR, GEOCALIB_WEIGHTS)
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.stats import gaussian_kernel, mad_filter_median
from common.geometry import gaussian_slerp_smoothing
from common.io import ParquetReader, write_clip_chunks, mark_done, remove_clip_shards

setup_extrinsics_imports()
from droid import Droid                                   # hawor's DROID-SLAM frontend
from torchvision.transforms import Resize
from moge.model.v2 import MoGeModel                       # MoGe-2 metric depth
from lib.pipeline.est_scale import est_scale_hybrid       # hawor scale estimator
from lib.eval_utils.custom_utils import quaternion_to_matrix
from hawor.utils.process import block_print, enable_print
from geocalib import GeoCalib                             # gravity prediction

GRAVITY_LAP_TOP_RATIO = 0.3   # share of sharpest frames used for gravity prediction
GRAVITY_LAP_MIN_SIZE = 50     # but at least this many frames (or all, if fewer)

# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 600
OVERLAP_RATIO = 0.25
STRIDE = int(SEQ_LEN * (1 - OVERLAP_RATIO))  # 450


# ──────────────────────────────────────────────────────────────────────────────
# DROID-SLAM args
# ──────────────────────────────────────────────────────────────────────────────
def _make_droid_args():
    return argparse.Namespace(
        weights=str(DROID_WEIGHTS),
        buffer=512,
        image_size=[240, 320],  # overridden per-window
        beta=0.3,
        filter_thresh=2.4,
        warmup=8,
        keyframe_thresh=4.0,
        frontend_thresh=16.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=22.0,
        backend_radius=2,
        backend_nms=3,
        upsample=True,
        stereo=False,
        disable_vis=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_slam_dims(h0, w0):
    """DROID-SLAM internal resolution: scale to ~384x512 area, crop to a multiple of 8."""
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    h2, w2 = h1 - h1 % 8, w1 - w1 % 8
    return h2, w2, h1, w1


def _image_stream(frames_bgr, calib):
    """BGR arrays → (t, tensor(1,3,H,W), intrinsics(4,)) generator for DROID."""
    h0, w0 = frames_bgr[0].shape[:2]
    fx, fy, cx, cy = calib[:4]
    _, _, h1, w1 = _get_slam_dims(h0, w0)
    h2, w2 = h1 - h1 % 8, w1 - w1 % 8

    for t, frame in enumerate(frames_bgr):
        image = cv2.resize(frame, (w1, h1))[:h2, :w2]
        image = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        yield t, image[None], intrinsics


def _build_masks(frame_ids, hand_reader, frame_shape, box_dilate=1.2):
    """Per-frame hand masks (H,W) bool, OR-combined across hands: mesh hand_mask when
    present, else a dilated bbox square. Empty when no hands."""
    H, W = frame_shape
    masks = []
    for fid in frame_ids:
        mask = np.zeros((H, W), dtype=bool)
        row = hand_reader.get(fid) if hand_reader is not None else None
        if row is not None:
            hands = row.get('hands', [])
            if hands:
                for h in hands:
                    if h is None:
                        continue
                    hm = h.get('hand_mask')
                    if hm is not None:
                        mask |= hm.astype(bool)
                    else:
                        bbox = h.get('box')
                        if bbox is not None:
                            bbox = np.asarray(bbox, dtype=np.float32)
                            x1, y1, x2, y2 = bbox
                            bw, bh = x2 - x1, y2 - y1
                            bcx, bcy = x1 + bw * 0.5, y1 + bh * 0.5
                            L = max(bw, bh) * box_dilate
                            rx1 = max(0, int(bcx - L * 0.5))
                            ry1 = max(0, int(bcy - L * 0.5))
                            rx2 = min(W, int(bcx + L * 0.5))
                            ry2 = min(H, int(bcy + L * 0.5))
                            mask[ry1:ry2, rx1:rx2] = True
        masks.append(mask)
    return masks


def _preprocess_masks(masks_bool, slam_h, slam_w):
    """Resize bool masks → (T,1,H,W) img_msks and (T,1,H//8,W//8) conf_msks tensors."""
    t = torch.from_numpy(np.stack(masks_bool, axis=0)).float().unsqueeze(1)

    resize_1 = Resize((slam_h, slam_w), antialias=True)
    resize_2 = Resize((slam_h // 8, slam_w // 8), antialias=True)

    img_msks, conf_msks = [], []
    for i in range(0, len(t), 500):
        chunk = t[i:i+500]
        img_msks.append((resize_1(chunk) > 0.5).float())
        conf_msks.append((resize_2(chunk) > 0.5).float())
    return torch.cat(img_msks), torch.cat(conf_msks)


def _run_slam_window(frames_bgr, masks_bool, calib, droid_args, label='window'):
    """Run DROID-SLAM on one window. Returns (traj, tstamp, disps) or None on failure.
    CUDA out of memory is raised, so the clip is not marked done."""
    h0, w0 = frames_bgr[0].shape[:2]
    slam_h, slam_w, _, _ = _get_slam_dims(h0, w0)
    img_msks, conf_msks = _preprocess_masks(masks_bool, slam_h, slam_w)

    droid = None
    try:
        for (t, image, intrinsics) in _image_stream(frames_bgr, calib):
            if droid is None:
                droid_args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(droid_args)

            img_msk = img_msks[t]
            conf_msk = conf_msks[t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, mask=conf_msk)

        # terminate: replay image stream for PoseTrajectoryFiller
        traj = droid.terminate(_image_stream(frames_bgr, calib))

        n = droid.video.counter.value
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]

        return traj, tstamp, disps
    except torch.OutOfMemoryError:
        print(f'  {label}: CUDA out of memory in DROID-SLAM, stopping')
        raise
    except TypeError as e:
        # pybind11's message when droid_backends.ba() rejects the arguments
        if str(e).startswith('ba(): incompatible function arguments'):
            raise TypeError(
                f"DROID-SLAM ba() argument mismatch: the SLAM frontend passes more args than "
                f"the compiled droid_backends.ba() accepts. The frontend must match the backend "
                f"build (droid_backends comes from the droidcalib install). Original error: {e}"
            ) from e
        raise
    except RuntimeError as e:
        if 'no kernel image is available' in str(e):
            cc = torch.cuda.get_device_capability()
            raise RuntimeError(
                f"droid_backends has no CUDA kernel for this GPU (sm_{cc[0]}{cc[1]}). "
                f"Rebuild the droidcalib extension with this arch in TORCH_CUDA_ARCH_LIST."
            ) from e
        print(f'  {label}: DROID-SLAM failed, skipping ({e!r})')
        return None
    except Exception as e:
        print(f'  {label}: DROID-SLAM failed, skipping ({e!r})')
        return None
    finally:
        if droid is not None:
            del droid
        torch.cuda.empty_cache()


def _estimate_scale(frames_bgr, tstamp, disps, masks_bool, moge2, fov_x, slam_hw):
    """Estimate metric scale by aligning MoGe-2 depth to SLAM disparity on keyframes."""
    slam_h, slam_w = slam_hw
    min_threshold = 0.4
    max_threshold = 0.7

    scales = []
    for i, t in enumerate(tstamp):
        rgb = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(rgb / 255.0, dtype=torch.float32, device='cuda').permute(2, 0, 1)
        output = moge2.infer(image_tensor, fov_x=fov_x)
        pred_depth = output['depth'].cpu().numpy()
        pred_depth = np.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
        pred_depth = cv2.resize(pred_depth, (slam_w, slam_h))

        slam_depth = 1.0 / disps[i]
        msk = masks_bool[t].astype(np.uint8)

        # est_scale_hybrid needs autograd, and enable_grad alone cannot exit inference_mode
        with torch.inference_mode(mode=False):
            nt, ft = min_threshold, max_threshold
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk,
                                     near_thresh=nt, far_thresh=ft)
            while math.isnan(scale):
                nt -= 0.1
                ft += 0.1
                if nt < -1.0:
                    break
                scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk,
                                         near_thresh=nt, far_thresh=ft)
        if not math.isnan(scale):
            scales.append(scale)

    if len(scales) < 3:
        return float('nan')
    return float(np.median(scales))


def _traj_to_T_cam2world(traj, scale):
    """DROID trajectory (N,7) + scale → (N,4,4) T_cam2world.
    traj = [t_xyz | quat_xyzw] (c2w), and quaternion_to_matrix expects wxyz."""
    t_c2w = traj[:, :3] * scale
    quat_xyzw = torch.from_numpy(traj[:, 3:]).float()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    R_c2w = quaternion_to_matrix(quat_wxyz).numpy()  # (N,3,3)

    N = traj.shape[0]
    Ts = np.zeros((N, 4, 4), dtype=np.float32)
    Ts[:, :3, :3] = R_c2w
    Ts[:, :3, 3] = t_c2w
    Ts[:, 3, 3] = 1.0
    return Ts


# ──────────────────────────────────────────────────────────────────────────────
# Gravity prediction helpers
# ──────────────────────────────────────────────────────────────────────────────
def _compute_laplacian_variance(frames_bgr):
    """Laplacian variance (sharpness) per frame. Returns (N,) array."""
    laps = np.empty(len(frames_bgr), dtype=np.float64)
    for i, f in enumerate(frames_bgr):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        laps[i] = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laps


def _select_sharp_frames(laps, lap_top_ratio=GRAVITY_LAP_TOP_RATIO,
                         lap_min_size=GRAVITY_LAP_MIN_SIZE):
    """Indices of the sharpest frames by Laplacian variance (top ratio, min count)."""
    N = len(laps)
    n_select = int(N * lap_top_ratio)
    n_select = max(n_select, min(lap_min_size, N))
    if n_select <= 0:
        return np.array([], dtype=int)
    vals = np.nan_to_num(laps, nan=-np.inf)
    top_idx = np.argpartition(vals, -n_select)[-n_select:]
    return np.sort(top_idx)


GRAVITY_BATCH = 32  # frames per GeoCalib forward


def _calibrate_gravity_batch(geocalib_model, img_batch, focal_prior):
    """GeoCalib's extractor.calibrate, batched over frames from one camera. Only gravity is consumed."""
    img_data = geocalib_model.image_processor(img_batch)
    prior_focal = focal_prior[None].expand(img_batch.shape[0])
    prior_values = {"prior_focal": prior_focal * img_data["scales"][1]}
    geocalib_model.model.optimizer.set_camera_model("pinhole")
    geocalib_model.model.optimizer.shared_intrinsics = False
    out = geocalib_model.model(img_data | prior_values)
    return out["gravity"]


def _predict_gravity_frames(frames_bgr, indices, geocalib_model, img_focal, label='window'):
    """GeoCalib on selected frame indices (batched) -> list of (local_idx, up_cam_3d).
    "up" is in the OpenCV camera frame. Only downward-looking, |roll| <= 10deg frames are kept.
    A failed batch is skipped, but CUDA out of memory is raised."""
    results = []
    if len(indices) == 0:
        return results
    focal_prior = torch.tensor(img_focal, device='cuda')
    for s in range(0, len(indices), GRAVITY_BATCH):
        chunk = [int(i) for i in indices[s:s + GRAVITY_BATCH]]
        try:
            rgb = np.stack([cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in chunk])
            img_t = torch.from_numpy(rgb).to(device='cuda', dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            gravity = _calibrate_gravity_batch(geocalib_model, img_t, focal_prior)
            pitches = gravity.pitch.cpu().numpy()
            rolls = gravity.roll.cpu().numpy()
            vecs = gravity.vec3d.cpu().numpy()
        except torch.OutOfMemoryError:
            print(f'  {label}: CUDA out of memory in GeoCalib, stopping')
            raise
        except Exception as e:
            print(f'  {label}: GeoCalib batch failed, skipping ({e!r})')
            continue
        for j, i in enumerate(chunk):
            if pitches[j] >= 0:
                continue
            if abs(rolls[j]) > math.radians(10):
                continue
            up_cam = vecs[j].astype(np.float64)
            up_cam /= (np.linalg.norm(up_cam) + 1e-12)
            results.append((i, up_cam))
    return results


def _aggregate_up_in_world(up_per_frame, cam_poses_dict, fid_set=None):
    """Aggregate camera-frame up vectors (fid -> (3,)) into one normalized world-frame up
    via cam_poses_dict (fid -> T_cam2world), optionally restricted to fid_set. None if < 3 valid."""
    up_world_list = []
    for fid, up_cam in up_per_frame.items():
        if fid_set is not None and fid not in fid_set:
            continue
        T = cam_poses_dict.get(fid)
        if T is None or np.isnan(T).any():
            continue
        up_w = T[:3, :3].astype(np.float64) @ up_cam
        up_w /= (np.linalg.norm(up_w) + 1e-12)
        up_world_list.append(up_w)

    if len(up_world_list) < 3:
        return None

    # hemisphere unification (GeoCalib can flip the up vector on ambiguous frames)
    ref = up_world_list[0]
    up_world_list = [u if np.dot(u, ref) > 0 else -u for u in up_world_list]

    up_arr = np.array(up_world_list)  # (N, 3)
    up_world = mad_filter_median(up_arr)
    if up_world is None:
        return None
    return up_world / (np.linalg.norm(up_world) + 1e-12)


def _compute_R_gravity_align(up_world):
    """Rotation (3,3) aligning up to +Z. The yaw is arbitrary but deterministic."""
    Z_up = up_world / (np.linalg.norm(up_world) + 1e-12)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(Z_up, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    X = ref - np.dot(ref, Z_up) * Z_up
    X /= (np.linalg.norm(X) + 1e-12)
    Y = np.cross(Z_up, X)
    Y /= (np.linalg.norm(Y) + 1e-12)

    return np.array([X, Y, Z_up], dtype=np.float64)  # rows = new basis


def _apply_gravity_alignment(aligned_poses, up_per_frame, gap_fill=0, label=''):
    """Per-segment gravity alignment from up-vector predictions, applied in-place
    (each contiguous run may come from a different SLAM chain with its own world frame).
    A segment whose up vector cannot be aggregated has its poses removed."""
    if not aligned_poses:
        return

    segments = _split_into_segments(aligned_poses, gap_fill=gap_fill)

    for seg_fids in segments:
        fid_set = set(seg_fids)
        up_world = _aggregate_up_in_world(up_per_frame, aligned_poses, fid_set=fid_set)
        if up_world is None:
            print(f'  {label}: frames {seg_fids[0]}-{seg_fids[-1]} have too few gravity predictions, '
                  f'dropping their {len(seg_fids)} poses')
            for fid in seg_fids:
                del aligned_poses[fid]
            continue

        R_align = _compute_R_gravity_align(up_world)
        T_align = np.eye(4, dtype=np.float64)
        T_align[:3, :3] = R_align

        for fid in seg_fids:
            aligned_poses[fid] = (T_align @ aligned_poses[fid].astype(np.float64)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Window alignment
# ──────────────────────────────────────────────────────────────────────────────
def _invert_T(T):
    """Invert a batch of 4x4 SE3 matrices. T: (N,4,4) → (N,4,4)."""
    R = T[:, :3, :3]
    t = T[:, :3, 3:]
    R_inv = R.transpose(0, 2, 1)
    t_inv = -R_inv @ t
    out = np.zeros_like(T)
    out[:, :3, :3] = R_inv
    out[:, :3, 3:] = t_inv
    out[:, 3, 3] = 1.0
    return out


def _mat_to_quat(R):
    """Rotation matrix (3,3) → quaternion wxyz (4,)."""
    from scipy.spatial.transform import Rotation as ScipyR
    q_xyzw = ScipyR.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _quat_to_mat(q_wxyz):
    """Quaternion wxyz (4,) → rotation matrix (3,3)."""
    from scipy.spatial.transform import Rotation as ScipyR
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return ScipyR.from_quat(q_xyzw).as_matrix()


def _quaternion_mean(quats_wxyz):
    """Robust mean of quaternions (hemisphere-flip to first, then normalized average)."""
    quats = quats_wxyz.copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    mean_q = quats.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q) + 1e-12
    return mean_q


def _compute_alignment_transform(T_ref, T_src):
    """Sim(3) alignment (s, T_align) mapping corresponding T_src poses onto T_ref.
    T_ref, T_src: (K,4,4). Returns (scale, (4,4) SE3)."""
    T_src_inv = _invert_T(T_src)
    T_align_each = T_ref @ T_src_inv  # (K,4,4)

    quats = np.array([_mat_to_quat(T_align_each[i, :3, :3]) for i in range(len(T_align_each))])
    mean_q = _quaternion_mean(quats)
    R_align = _quat_to_mat(mean_q)

    # Scale: robust median of pairwise distance ratios
    t_ref = T_ref[:, :3, 3].astype(np.float64)           # (K,3)
    t_src_rot = (R_align @ T_src[:, :3, 3:]).squeeze(-1)  # (K,3)

    K = len(T_ref)
    ratios = []
    eps = 1e-6
    for i in range(K):
        for j in range(i + 1, K):
            d_ref = np.linalg.norm(t_ref[i] - t_ref[j])
            d_src = np.linalg.norm(t_src_rot[i] - t_src_rot[j])
            if d_src > eps:
                ratios.append(d_ref / d_src)

    if ratios:
        s = float(np.median(ratios))
        if not (0.5 <= s <= 2.0):
            s = 1.0  # unreasonable scale → fall back to SE(3)
    else:
        s = 1.0

    t_align = np.median(t_ref - s * t_src_rot, axis=0)

    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R_align
    T_align[:3, 3] = t_align
    return s, T_align.astype(np.float32)


def _apply_sim3(s, T_align, Ts):
    """Apply Sim(3) (s, T_align) to a batch of poses."""
    R_align = T_align[:3, :3].astype(np.float64)
    t_align = T_align[:3, 3].astype(np.float64)
    N = len(Ts)
    out = np.zeros((N, 4, 4), dtype=np.float64)
    out[:, :3, :3] = R_align @ Ts[:, :3, :3].astype(np.float64)
    out[:, :3, 3] = s * (R_align @ Ts[:, :3, 3:].astype(np.float64)).squeeze(-1) + t_align
    out[:, 3, 3] = 1.0
    return out.astype(np.float32)


def _slerp(R0, R1, t):
    """Spherical linear interpolation between two rotation matrices."""
    from scipy.spatial.transform import Rotation as ScipyR, Slerp
    rots = ScipyR.concatenate([ScipyR.from_matrix(R0), ScipyR.from_matrix(R1)])
    slerp = Slerp([0.0, 1.0], rots)
    return slerp([float(t)])[0].as_matrix()


def _align_and_merge_windows(window_results, gap_fill=0):
    """Align overlapping windows and merge into a global frame_id → T_cam2world dict.
    window_results: list of (global_frame_ids: list[int], T_cam2world: (T,4,4)), in frame order.
    Each chain of aligned windows has its own world frame. A chain's frames within gap_fill + 1
    of the previous chain are dropped, so segment splitting keeps the chains apart."""
    if not window_results:
        return {}

    # Group into alignment chains (break when windows share fewer than 2 frames,
    # the minimum for the Sim(3) alignment)
    chains = []
    current_chain = [0]
    for k in range(1, len(window_results)):
        prev_fids = set(window_results[k - 1][0])
        curr_fids = set(window_results[k][0])
        if len(prev_fids & curr_fids) >= 2:
            current_chain.append(k)
        else:
            chains.append(current_chain)
            current_chain = [k]
    chains.append(current_chain)

    global_poses = {}  # frame_id → list of (T_cam2world, weight)

    for chain in chains:
        # frames before min_fid would fall into the previous chain's segment
        min_fid = max(global_poses) + gap_fill + 2 if global_poses else 0
        aligned_Ts = {}  # window_idx → aligned (fids, T_array)

        for ci, widx in enumerate(chain):
            fids, Ts = window_results[widx]
            fids = list(fids)
            Ts = Ts.copy()

            if ci == 0:
                aligned_Ts[widx] = (fids, Ts)
            else:
                prev_widx = chain[ci - 1]
                prev_fids, prev_Ts = aligned_Ts[prev_widx]
                prev_map = {f: i for i, f in enumerate(prev_fids)}

                overlap_prev, overlap_curr = [], []
                for j, f in enumerate(fids):
                    if f in prev_map:
                        overlap_prev.append(prev_Ts[prev_map[f]])
                        overlap_curr.append(Ts[j])

                T_ref = np.stack(overlap_prev)
                T_src = np.stack(overlap_curr)
                s, T_align = _compute_alignment_transform(T_ref, T_src)
                Ts_aligned = _apply_sim3(s, T_align, Ts)
                aligned_Ts[widx] = (fids, Ts_aligned)

        # Merge with cosine weighting (peaks at window center, tapers at edges)
        for widx in chain:
            fids, Ts = aligned_Ts[widx]
            n = len(fids)
            center = (n - 1) / 2.0
            half_len = n / 2.0

            for i, fid in enumerate(fids):
                if fid < min_fid:
                    continue
                w = 0.5 * (1.0 + math.cos(math.pi * (i - center) / half_len))
                w = max(w, 1e-6)
                global_poses.setdefault(fid, []).append((Ts[i], w))

    # Blend: SLERP for rotation, LERP for translation
    result = {}
    for fid, pose_weights in global_poses.items():
        if len(pose_weights) == 1:
            result[fid] = pose_weights[0][0]
        else:
            total_w = sum(pw[1] for pw in pose_weights)
            t_avg = np.zeros(3, dtype=np.float64)
            for T, w in pose_weights:
                t_avg += (w / total_w) * T[:3, 3].astype(np.float64)

            sorted_pw = sorted(pose_weights, key=lambda x: -x[1])  # highest weight first
            R_avg = sorted_pw[0][0][:3, :3].astype(np.float64)
            w_acc = sorted_pw[0][1]
            for j in range(1, len(sorted_pw)):
                T_j, w_j = sorted_pw[j]
                R_j = T_j[:3, :3].astype(np.float64)
                blend = w_j / (w_acc + w_j)
                R_avg = _slerp(R_avg, R_j, blend)
                w_acc += w_j

            T_out = np.eye(4, dtype=np.float32)
            T_out[:3, :3] = R_avg.astype(np.float32)
            T_out[:3, 3] = t_avg.astype(np.float32)
            result[fid] = T_out

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Segment splitting (shared by smoothing and gravity alignment)
# ──────────────────────────────────────────────────────────────────────────────
def _split_into_segments(poses_dict, gap_fill=0):
    """Split frame IDs into contiguous runs (gaps up to gap_fill tolerated).
    Returns list of sorted-fid lists."""
    if not poses_dict:
        return []
    sorted_fids = sorted(poses_dict.keys())
    segments = []
    seg_start = 0
    for i in range(1, len(sorted_fids)):
        if sorted_fids[i] > sorted_fids[i - 1] + 1 + gap_fill:
            segments.append(sorted_fids[seg_start:i])
            seg_start = i
    segments.append(sorted_fids[seg_start:])
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Post-merge SE(3) Gaussian smoothing
# ──────────────────────────────────────────────────────────────────────────────
_SMOOTH_SIGMA = 2.0
_SMOOTH_KERNEL_SIZE = 13  # 6*sigma + 1


def _smooth_cam_poses(poses_dict, sigma=_SMOOTH_SIGMA, kernel_size=_SMOOTH_KERNEL_SIZE, gap_fill=0):
    """SE(3) Gaussian low-pass over the merged trajectory, in-place. Splits on frame
    gaps and skips segments shorter than the kernel."""
    if len(poses_dict) < kernel_size:
        return

    segments = _split_into_segments(poses_dict, gap_fill=gap_fill)

    kernel = gaussian_kernel(kernel_size, sigma)
    half_k = kernel_size // 2

    for seg_fids in segments:
        N = len(seg_fids)
        if N < kernel_size:
            continue

        Ts = np.stack([poses_dict[f] for f in seg_fids])  # (N,4,4)

        smoothed_rots = gaussian_slerp_smoothing(Ts[:, :3, :3], sigma, kernel_size)

        trans = Ts[:, :3, 3].astype(np.float64)  # (N,3)
        smoothed_trans = np.empty_like(trans)
        for i in range(N):
            start = max(0, i - half_k)
            end = min(N, i + half_k + 1)
            w = kernel[half_k - (i - start): half_k + (end - i)]
            w = w / w.sum()
            smoothed_trans[i] = w @ trans[start:end]

        for i, fid in enumerate(seg_fids):
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = smoothed_rots[i].astype(np.float32)
            T[:3, 3] = smoothed_trans[i].astype(np.float32)
            poses_dict[fid] = T


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--chunk_size', type=int, default=5000, help='frames per parquet chunk')
    parser.add_argument('--gap_fill', type=int, default=3, help='Max consecutive missing frames to bridge in segment splitting')
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    if osp.isdir(input_dir):
        paths_mp4 = sorted(glob.glob(osp.join(input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part, key=mp4_clip_id)
        if len(paths_mp4) == 0:
            return
        hand_dir = input_dir + '_hand'
        output_dir = input_dir + '_extr'
    else:
        assert osp.isfile(input_dir)
        assert input_dir.lower().endswith('.mp4')
        paths_mp4 = [input_dir]
        base_dir = osp.dirname(input_dir)
        hand_dir = osp.join(base_dir, 'hand')
        output_dir = osp.join(base_dir, 'extr')

    def is_done(path_mp4):
        clip_id = osp.basename(path_mp4)[:-4]
        return osp.exists(osp.join(output_dir, f"{clip_id}.done"))

    print(f'Got {len(paths_mp4)} mp4 files in total.')
    paths_mp4_new = [p for p in paths_mp4 if not is_done(p)]
    print(f'Skipping {len(paths_mp4) - len(paths_mp4_new)} mp4 files (already processed)')
    paths_mp4 = paths_mp4_new
    if len(paths_mp4) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)

    # ── load models (once) ──
    # MoGe-2: prefer the local checkpoint, and download it once if missing.
    if MOGE2_WEIGHTS.is_file():
        moge2_path = str(MOGE2_WEIGHTS)
    else:
        from huggingface_hub import snapshot_download
        print(f'Downloading MoGe-2 to {MOGE2_WEIGHTS_DIR} ...')
        snapshot_download(repo_id=MOGE2_REPO, local_dir=str(MOGE2_WEIGHTS_DIR))
        print('Done.')
        moge2_path = str(MOGE2_WEIGHTS)
    block_print()
    moge2 = MoGeModel.from_pretrained(moge2_path).to('cuda').eval()
    enable_print()

    geocalib_model = GeoCalib(weights=str(GEOCALIB_WEIGHTS)).to('cuda').eval()

    # ── process clips ──
    with torch.inference_mode():
        for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1,
                             leave=True, dynamic_ncols=True, disable=args.no_tqdm):
            clip_id = osp.basename(path_mp4)[:-4]
            # the clip has no .done, so its shards are written from scratch
            remove_clip_shards(output_dir, clip_id)

            # Phase 1: read stage 4's shards (intrinsics + hand masks carried in)
            try:
                hand_reader = ParquetReader(hand_dir, clip_id)
            except FileNotFoundError:
                mark_done(output_dir, clip_id)
                continue
            hand_fids = hand_reader.get_frame_ids()
            if not hand_fids:
                mark_done(output_dir, clip_id)
                continue

            meta = hand_reader.get(hand_fids[0])
            intr = np.array([meta['fx'], meta['fy'], meta['cx'], meta['cy'], meta['xi']], dtype=np.float64)
            image_h, image_w = int(meta['height']), int(meta['width'])
            frame_shape = (image_h, image_w)

            # Undistortion (rebuild pinhole from the raw MEI/UCM model)
            need_undist = need_undistort(intr, frame_shape)
            if need_undist is None:
                mark_done(output_dir, clip_id)
                continue
            map1 = map2 = None
            if need_undist:
                try:
                    map1, map2, _, intr = build_undistort_maps(frame_shape, intr, auto=True)
                except Exception:
                    mark_done(output_dir, clip_id)
                    continue
            img_focal = 0.5 * (intr[0] + intr[1])

            # SLAM calibration (square pinhole)
            calib = np.array([img_focal, img_focal, float(intr[2]), float(intr[3])], dtype=np.float64)
            fov_x = 2 * math.degrees(math.atan(image_w / (2 * img_focal)))

            # Phase 2: Windowed SLAM over a streaming decode with a bounded buffer
            slam_h, slam_w, _, _ = _get_slam_dims(image_h, image_w)

            frame_buffer = []   # rolling buffer of BGR frames
            buf_start = 0       # global frame_id of frame_buffer[0]
            next_win_start = 0
            last_win_start = -1
            window_results = []
            up_per_frame = {}   # frame_id → (3,) up vector in camera frame, across windows

            def _process_one_window(win_frames, win_fids):
                wi = len(window_results)
                label = f'{clip_id} window {wi} (frames {win_fids[0]}-{win_fids[-1]})'
                masks_bool = _build_masks(win_fids, hand_reader, frame_shape)
                slam_result = _run_slam_window(win_frames, masks_bool, calib, _make_droid_args(), label)
                if slam_result is None:
                    return None
                traj, tstamp, disps = slam_result
                # pose/frame count mismatch would misalign the merge -> skip window
                if len(traj) != len(win_fids):
                    print(f'  {label}: traj length {len(traj)} != frame count {len(win_fids)}, skipping')
                    return None
                scale = _estimate_scale(win_frames, tstamp, disps, masks_bool,
                                        moge2, fov_x, (slam_h, slam_w))
                if not math.isfinite(scale):
                    return None
                T_cam2world = _traj_to_T_cam2world(traj, scale)
                # a non-finite pose would break the merge and the smoothing -> skip window
                if not np.isfinite(T_cam2world).all():
                    print(f'  {label}: non-finite camera pose, skipping')
                    return None

                laps = _compute_laplacian_variance(win_frames)
                sharp_idx = _select_sharp_frames(laps)
                new_idx = np.array([i for i in sharp_idx if win_fids[i] not in up_per_frame], dtype=int)
                up_results = _predict_gravity_frames(win_frames, new_idx, geocalib_model, img_focal, label)
                for local_i, up_cam in up_results:
                    up_per_frame[win_fids[local_i]] = up_cam

                return (win_fids, T_cam2world)

            with av.open(path_mp4, "r") as reader:
                s = reader.streams.video[0]
                total = s.frames or (int(float(s.duration * s.time_base) * 30 + 0.5)
                                     if s.duration and s.time_base else None)
                for frame_id, frame in enumerate(tqdm(
                        reader.decode(video=0), total=total, desc=f"{clip_id}",
                        unit="frame", position=0, leave=False, dynamic_ncols=True,
                        disable=args.no_tqdm)):
                    im_bgr = frame.to_ndarray(format='bgr24')
                    if frame_id == 0:
                        assert frame_shape == im_bgr.shape[:2]
                    if need_undist:
                        im_bgr = undistort_apply(im_bgr, map1, map2)
                    frame_buffer.append(im_bgr)

                    # Process regular windows as they become available
                    buf_end = buf_start + len(frame_buffer)
                    while next_win_start + SEQ_LEN <= buf_end:
                        ws = next_win_start
                        local_s = ws - buf_start
                        win_frames = frame_buffer[local_s:local_s + SEQ_LEN]
                        win_fids = list(range(ws, ws + SEQ_LEN))
                        window_results.append(_process_one_window(win_frames, win_fids))
                        last_win_start = ws
                        next_win_start += STRIDE
                        # trim buffer up to this window's start (tail window may overlap back to here)
                        trim = ws - buf_start
                        if trim > 0:
                            del frame_buffer[:trim]
                            buf_start = ws

            total_frames = buf_start + len(frame_buffer)
            if total_frames == 0:
                mark_done(output_dir, clip_id)
                continue

            # Handle tail: last window covering the end
            if total_frames > SEQ_LEN:
                tail_start = total_frames - SEQ_LEN
                if tail_start != last_win_start and tail_start >= buf_start:
                    local_s = tail_start - buf_start
                    win_frames = frame_buffer[local_s:local_s + SEQ_LEN]
                    win_fids = list(range(tail_start, total_frames))
                    window_results.append(_process_one_window(win_frames, win_fids))
            elif last_win_start < 0:
                # Short video (< SEQ_LEN): process all available frames
                win_fids = list(range(total_frames))
                window_results.append(_process_one_window(frame_buffer, win_fids))

            del frame_buffer

            # Phase 3: Align windows, smooth, gravity alignment
            valid_results = [r for r in window_results if r is not None]
            aligned_poses = _align_and_merge_windows(valid_results, gap_fill=args.gap_fill)
            _smooth_cam_poses(aligned_poses, gap_fill=args.gap_fill)

            # segments without enough valid gravity predictions lose their poses
            _apply_gravity_alignment(aligned_poses, up_per_frame, gap_fill=args.gap_fill, label=clip_id)

            # Phase 4: Write stage 4's rows with cam_pose filled
            chunk_idx = 0
            rows = []
            for fid in sorted(hand_fids):
                row = hand_reader.get(fid)
                if row is None:
                    continue
                new_row = dict(row)
                T = aligned_poses.get(fid)
                if T is not None:
                    # Clear cam_pose on frames with large roll in the gravity-aligned world
                    up_cam = T[:3, :3].astype(np.float64).T @ np.array([0.0, 0.0, 1.0])
                    roll = math.degrees(math.atan2(up_cam[0], -up_cam[1]))
                    if abs(roll) > 15:
                        T = None
                new_row['cam_pose'] = T  # (4,4) T_cam2world ndarray or None
                rows.append(new_row)

                if len(rows) == args.chunk_size:
                    write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
                    chunk_idx += 1
                    rows.clear()

            if rows:
                write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
            mark_done(output_dir, clip_id)


if __name__ == "__main__":
    main()

# Stage 8: robot retargeting (PyRoKi IK). Human hand keypoints -> robot joint angles + EEF.
# Reads stage 7's per-segment parquets and writes one parquet per segment (state_qpos, EEF,
# masks, diagnostics). Adds cam_pose_base = cam-to-robot-base (OpenCV, robot at the origin).
# cam_pose keeps stage 5's cam-to-world.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage8_robot_retarget.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import os, sys

# Must be set before the first jax import.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ['JAX_PLATFORMS'] = 'cuda'

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, Tuple
import glob, os.path as osp, argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition
from common.io import decode_row, hand_for_side, rows_to_parquet, mark_done, remove_stale_temps
from common.geometry import interpolate_cam_poses
from common.robot_config import load_robot_config, schema_metadata
from common.video import validate_segment_outputs


def fill_missing(seq: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior gaps, edge-fill boundaries. All-missing becomes zeros."""
    T = seq.shape[0]
    filled = seq.copy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.zeros_like(seq)

    first_valid = valid_indices[0]
    if first_valid > 0:
        filled[:first_valid] = seq[first_valid]
    last_valid = valid_indices[-1]
    if last_valid < T - 1:
        filled[last_valid + 1:] = seq[last_valid]

    for k in range(len(valid_indices) - 1):
        i0 = valid_indices[k]
        i1 = valid_indices[k + 1]
        if i1 - i0 <= 1:
            continue
        alpha = np.arange(1, i1 - i0) / (i1 - i0)
        for _ in range(seq.ndim - 1):
            alpha = alpha[..., np.newaxis]
        filled[i0 + 1:i1] = (1 - alpha) * seq[i0] + alpha * seq[i1]

    return filled


def smooth_kpts_validity_aware(kpts: np.ndarray, valid: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smoothing ignoring invalid frames. kpts: (T, 21, 3), valid: (T,)."""
    v = valid.astype(np.float32)[:, None, None]
    k0 = np.where(v > 0, kpts, 0.0)
    num = gaussian_filter1d(k0 * v, sigma=sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(v, sigma=sigma, axis=0, mode="nearest")
    out = num / np.clip(den, 1e-6, None)
    return np.where(den < 0.1, kpts, out)


def downweight_outlier_frames(
    kpts_base: np.ndarray,
    valid_bool: np.ndarray,
    max_speed_mps: float = 4.0,
    fps: float = 30.0,
    min_weight: float = 0.01,
    power: int = 4,
) -> np.ndarray:
    """Soft per-frame weights in [0, 1] from wrist speed/acceleration. Call after fill_missing."""
    T = len(valid_bool)
    v = valid_bool.astype(bool)
    w = v.astype(np.float32)

    if T == 0:
        return w

    thr = max(float(max_speed_mps) / float(max(fps, 1e-6)), 1e-6)  # m/frame
    wrist = kpts_base[:, 0, :]

    speed = np.zeros(T, dtype=np.float32)
    if T > 1:
        sd = np.linalg.norm(wrist[1:] - wrist[:-1], axis=1).astype(np.float32)
        speed[1:] = sd * v[1:].astype(np.float32)

    accel = np.zeros(T, dtype=np.float32)
    if T > 2:
        ad = np.linalg.norm(wrist[2:] - 2.0 * wrist[1:-1] + wrist[:-2], axis=1).astype(np.float32)
        accel[1:-1] = ad * v[1:-1].astype(np.float32)

    score = np.maximum(speed / thr, accel / thr)
    score = np.where(v, score, 0.0)

    w = w * np.exp(-(score ** power)).astype(np.float32)
    w = np.where(v, np.maximum(w, float(min_weight)), 0.0).astype(np.float32)
    return np.clip(w, 0.0, 1.0)


def determine_mode(left_valid: np.ndarray, right_valid: np.ndarray) -> str:
    """Which hands the segment contains, from existence alone."""
    has_left, has_right = np.any(left_valid), np.any(right_valid)
    if has_right and not has_left:
        return "right_only"
    if has_left and not has_right:
        return "left_only"
    if has_left and has_right:
        return "bimanual"
    raise ValueError("No valid hand frames")


def compute_wrist_centroid_in_world(
    left_kpts_cf: np.ndarray,
    right_kpts_cf: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    cam_poses: np.ndarray,
    mode: str = "bimanual",
) -> np.ndarray:
    """(3,) robust wrist centroid in world frame (per-side medians, averaged 1:1 for bimanual)."""
    T = left_kpts_cf.shape[0]
    left_ws, right_ws = [], []
    for t in range(T):
        R = cam_poses[t, :3, :3]
        tvec = cam_poses[t, :3, 3]
        if left_valid[t]:
            left_ws.append(R @ left_kpts_cf[t, 0] + tvec)
        if right_valid[t]:
            right_ws.append(R @ right_kpts_cf[t, 0] + tvec)

    if mode == "right_only":
        return np.median(right_ws, axis=0)
    if mode == "left_only":
        return np.median(left_ws, axis=0)
    centroids = []
    if len(left_ws) > 0:
        centroids.append(np.median(left_ws, axis=0))
    if len(right_ws) > 0:
        centroids.append(np.median(right_ws, axis=0))
    return np.mean(centroids, axis=0)


def compute_yaw_from_cam_forward(cam_poses: np.ndarray, min_xy_norm: float = 0.05) -> float:
    """Facing yaw from the camera forward axis (OpenCV Z), XY-norm-weighted circular mean."""
    fwd_xy = cam_poses[:, :3, 2][:, :2]
    norms = np.linalg.norm(fwd_xy, axis=1)

    keep = norms > min_xy_norm
    if not np.any(keep):
        best = int(np.argmax(norms))
        return float(np.arctan2(fwd_xy[best, 1], fwd_xy[best, 0]))

    ang = np.arctan2(fwd_xy[keep, 1], fwd_xy[keep, 0])
    w = norms[keep]
    w_sum = w.sum()
    mean_sin = (w * np.sin(ang)).sum() / w_sum
    mean_cos = (w * np.cos(ang)).sum() / w_sum
    return float(np.arctan2(mean_sin, mean_cos))


def compute_segment_yaw(cam_poses: np.ndarray, valid_mask: np.ndarray) -> float:
    """Facing yaw over the valid frames of a segment."""
    if len(valid_mask) != len(cam_poses):
        raise ValueError("valid_mask and cam_poses length mismatch")
    if not np.any(valid_mask):
        raise ValueError("No valid frames for yaw estimation")
    return compute_yaw_from_cam_forward(cam_poses[valid_mask])


def check_wrong_hand_outlier(
    left_kpts_cf: np.ndarray,
    right_kpts_cf: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    cam_poses: np.ndarray,
    cam_valid: np.ndarray,
    optimized_offset_4dof: np.ndarray,
    wrist_dist_thr: float = 0.9,
    yaw_delta_thr: float = 20.0,
) -> Tuple[bool, float, float]:
    """Flag segments whose hands likely belong to someone else.
    Returns (is_outlier, median_wrist_dist, yaw_delta_deg)."""
    wrist_dists = []
    for t in range(len(left_valid)):
        if left_valid[t]:
            wrist_dists.append(float(np.linalg.norm(left_kpts_cf[t, 0])))
        if right_valid[t]:
            wrist_dists.append(float(np.linalg.norm(right_kpts_cf[t, 0])))
    median_wd = float(np.median(wrist_dists)) if wrist_dists else 0.0

    initial_yaw = np.degrees(compute_yaw_from_cam_forward(cam_poses[cam_valid]))
    opt_yaw = np.degrees(optimized_offset_4dof[3])
    yaw_delta = abs(((opt_yaw - initial_yaw) + 180) % 360 - 180)

    return (median_wd > wrist_dist_thr) and (yaw_delta > yaw_delta_thr), median_wd, yaw_delta


def check_camera_drift(
    cam_poses: np.ndarray,
    cam_valid: np.ndarray,
    cam_drift_thr: float = 1.2,
) -> Tuple[bool, float]:
    """Reject walking segments by first-to-last camera translation (the robot base is fixed)."""
    valid_pos = cam_poses[cam_valid, :3, 3]
    if len(valid_pos) < 2:
        return False, 0.0
    cam_drift = float(np.linalg.norm(valid_pos[-1] - valid_pos[0]))
    return cam_drift > cam_drift_thr, cam_drift


def compute_workspace_anchors(retargeter) -> dict:
    """Per-side wrist positions at the home pose (base frame)."""
    import jax.numpy as jnp
    import jaxlie

    robot = retargeter.robot
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=jnp.array(retargeter.get_neutral_config())))
    positions = np.array(T_root_link.translation())

    link_names = list(robot.links.names)
    mapping = retargeter.config["keypoint_mapping"]
    left_pos = positions[link_names.index(mapping["left"][0])]
    right_pos = positions[link_names.index(mapping["right"][0])]
    return {"left": left_pos, "right": right_pos, "center": (left_pos + right_pos) / 2.0}


def compute_T_base_world(
    yaw: float,
    centroid_world: np.ndarray,
    workspace_anchor: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """T_{base<-world}: yaw-only rotation. Puts centroid_world onto workspace_anchor + delta."""
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_yaw
    T[:3, 3] = (workspace_anchor + delta) - R_yaw @ centroid_world
    return T


def transform_kpts_batch(kpts: np.ndarray, T_per_frame: np.ndarray) -> np.ndarray:
    """Apply a per-frame 4x4 transform to (T, N, 3) points."""
    R = T_per_frame[:, :3, :3]
    t = T_per_frame[:, :3, 3]
    return np.einsum('tij,tnj->tni', R, kpts) + t[:, None, :]


class ActionCalculator:
    """Retargets segments and derives robot state, EEF, ghost camera and diagnostics."""

    def __init__(self, robot_cfg, state_slices: Dict, smooth_sigma: float = 1.0,
                 max_wrist_speed: float = 4.0, pad_lens=(224, 1024, 3072)):
        self._robot_cfg = robot_cfg
        self._state_slices = state_slices
        self._retargeter = None
        self._workspace_anchors = None
        self._smooth_sigma = smooth_sigma
        self._max_wrist_speed = max_wrist_speed
        self._pad_lens = tuple(sorted(pad_lens))

    @property
    def retargeter(self):
        if self._retargeter is None:
            from pipeline.retargeting.retargeter import Retargeter
            print(f"Loading retargeter for {self._robot_cfg.name} "
                  f"from {self._robot_cfg.urdf_path}...")
            self._retargeter = Retargeter(self._robot_cfg)
        return self._retargeter

    @property
    def workspace_anchors(self):
        if self._workspace_anchors is None:
            self._workspace_anchors = compute_workspace_anchors(self.retargeter)
            print(f"Workspace anchors: L={self._workspace_anchors['left']}, "
                  f"R={self._workspace_anchors['right']}, C={self._workspace_anchors['center']}")
        return self._workspace_anchors

    def process_segment(
        self,
        left_kpts_cf: np.ndarray,
        right_kpts_cf: np.ndarray,
        left_valid: np.ndarray,
        right_valid: np.ndarray,
        cam_poses: np.ndarray,
        cam_valid: np.ndarray,
    ) -> Dict:
        """Retarget one segment.
        kpts: (T, 21, 3) camera-frame. cam_poses: (T, 4, 4) OpenCV cam-to-world, identity in
        invalid slots. Hand validity masks are already ANDed with cam_valid."""
        from pipeline.retargeting.retargeter import offset_4dof_to_se3, se3_to_offset_4dof

        T = left_kpts_cf.shape[0]
        assert T <= self._pad_lens[-1], \
            f"Segment length {T} exceeds the longest padding length {self._pad_lens[-1]}"
        # IK pads to the shortest length that fits. Each length compiles once per process.
        pad_len = next(n for n in self._pad_lens if T <= n)

        cam_poses = np.asarray(cam_poses, dtype=np.float64)
        assert cam_poses.shape == (T, 4, 4)

        # Camera frame -> world frame
        left_kpts_world = transform_kpts_batch(left_kpts_cf, cam_poses)
        right_kpts_world = transform_kpts_batch(right_kpts_cf, cam_poses)

        # Pre-IK processing, all in world frame
        left_filled = fill_missing(left_kpts_world, left_valid)
        right_filled = fill_missing(right_kpts_world, right_valid)
        left_filled = smooth_kpts_validity_aware(left_filled, left_valid, sigma=self._smooth_sigma)
        right_filled = smooth_kpts_validity_aware(right_filled, right_valid, sigma=self._smooth_sigma)

        left_weights = downweight_outlier_frames(
            left_filled, left_valid, max_speed_mps=self._max_wrist_speed)
        right_weights = downweight_outlier_frames(
            right_filled, right_valid, max_speed_mps=self._max_wrist_speed)

        # Analytical stance guess to warm-start the offset solve
        mode = determine_mode(left_valid, right_valid)
        centroid_world = compute_wrist_centroid_in_world(
            left_kpts_cf, right_kpts_cf, left_valid, right_valid, cam_poses, mode=mode)
        anchor_key = {"right_only": "right", "left_only": "left", "bimanual": "center"}[mode]
        yaw = compute_segment_yaw(cam_poses, cam_valid)
        initial_offset_4dof = se3_to_offset_4dof(compute_T_base_world(
            yaw, centroid_world, self.workspace_anchors[anchor_key], np.zeros(3)))

        state_mask = np.column_stack([left_valid, right_valid])

        full_joints, optimized_offset_4dof, solver_cost = \
            self.retargeter.retarget_bimanual_world(
                left_filled.astype(np.float32),
                right_filled.astype(np.float32),
                left_valid=left_weights,
                right_valid=right_weights,
                target_cam_poses_world=cam_poses.astype(np.float32),
                cam_valid=cam_valid,
                initial_offset_4dof=initial_offset_4dof,
                max_seq_len=pad_len,
            )

        import jax.numpy as jnp
        T_base_world = np.array(offset_4dof_to_se3(jnp.array(optimized_offset_4dof)).as_matrix())

        state_qpos = self.retargeter.extract_joints(full_joints)

        # A side with no valid frame at all is pinned to the home pose.
        neutral = self.retargeter.extract_joints(self.retargeter.get_neutral_config())
        if np.all(~left_valid):
            state_qpos[:, self._state_slices["left_arm"]] = neutral[self._state_slices["left_arm"]]
            state_qpos[:, self._state_slices["left_hand"]] = neutral[self._state_slices["left_hand"]]
        if np.all(~right_valid):
            state_qpos[:, self._state_slices["right_arm"]] = neutral[self._state_slices["right_arm"]]
            state_qpos[:, self._state_slices["right_hand"]] = neutral[self._state_slices["right_hand"]]

        # Ghost camera: SLAM camera gap-interpolated, then rebased into base frame.
        ghost_cam = interpolate_cam_poses(cam_poses.astype(np.float64).copy(), cam_valid)
        ghost_cam = (T_base_world[None] @ ghost_cam).astype(np.float64)

        state_eef = self.retargeter.compute_eef_batch(state_qpos)

        # ---- Per-frame diagnostics: solved state vs gap-filled targets ----
        diag_fk = self.retargeter.compute_fk_positions_for_diagnostics(state_qpos)
        robot_global = diag_fk["global_pos"]
        robot_palm = diag_fk["palm_pos"]
        robot_local = diag_fk["local_pos"]
        fk_cam_se3 = diag_fk["cam_se3"]

        R_bw = T_base_world[:3, :3]
        t_bw = T_base_world[:3, 3]

        left_global_sel = left_filled[:, self.retargeter.left_global_kpt_indices]
        right_global_sel = right_filled[:, self.retargeter.right_global_kpt_indices]
        target_global = np.concatenate([left_global_sel, right_global_sel], axis=1)
        target_global_base = np.einsum('ij,tnj->tni', R_bw, target_global) + t_bw

        target_palm = np.stack([left_filled[:, 0], right_filled[:, 0]], axis=1)
        target_palm_base = np.einsum('ij,tnj->tni', R_bw, target_palm) + t_bw

        left_local_sel = left_filled[:, self.retargeter.left_local_kpt_indices]
        right_local_sel = right_filled[:, self.retargeter.right_local_kpt_indices]
        target_local = np.concatenate([left_local_sel, right_local_sel], axis=1)
        target_local_base = np.einsum('ij,tnj->tni', R_bw, target_local) + t_bw

        err_global = np.linalg.norm(robot_global - target_global_base, axis=-1) * 1000.0
        err_palm = np.linalg.norm(robot_palm - target_palm_base, axis=-1) * 1000.0

        n_left_local = len(self.retargeter.left_local_link_indices)
        n_right_local = len(self.retargeter.right_local_link_indices)
        local_kpt_mask = np.concatenate([
            np.broadcast_to(left_weights[:, None], (T, n_left_local)),
            np.broadcast_to(right_weights[:, None], (T, n_right_local)),
        ], axis=1)
        local_dir = self.retargeter.compute_local_dir_error(
            robot_local, target_local_base, local_kpt_mask)

        # A side with no valid frame at all has no target to measure, so its errors are NaN.
        if np.all(~left_valid):
            err_global[:, :5] = np.nan
            err_palm[:, 0] = np.nan
            local_dir[:, 0] = np.nan
        if np.all(~right_valid):
            err_global[:, 5:] = np.nan
            err_palm[:, 1] = np.nan
            local_dir[:, 1] = np.nan

        # Joint acceleration, edge-copied at boundaries
        err_ddq = np.zeros_like(state_qpos, dtype=np.float32)
        if T > 2:
            err_ddq[1:-1] = (state_qpos[2:] - 2 * state_qpos[1:-1] + state_qpos[:-2]).astype(np.float32)
            err_ddq[0] = err_ddq[1]
            err_ddq[-1] = err_ddq[-2]

        ghost_cam_64 = ghost_cam.astype(np.float64)
        err_cam_pos_mm = np.zeros(T, dtype=np.float32)
        err_cam_rot_deg = np.zeros(T, dtype=np.float32)
        for t in range(T):
            err_cam_pos_mm[t] = np.linalg.norm(
                fk_cam_se3[t, :3, 3] - ghost_cam_64[t, :3, 3]) * 1000.0
            R_rel = fk_cam_se3[t, :3, :3].T @ ghost_cam_64[t, :3, :3]
            trace_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            err_cam_rot_deg[t] = np.degrees(np.arccos(trace_val))

        return {
            "state_qpos": state_qpos,
            "state_eef": state_eef,
            "state_mask": state_mask,
            "cam_poses": ghost_cam.astype(np.float32),
            "optimized_offset_4dof": optimized_offset_4dof,
            "retarget_cost": float(solver_cost) / T,
            "err_tip_mm_left": err_global[:, :5].astype(np.float32),
            "err_tip_mm_right": err_global[:, 5:].astype(np.float32),
            "err_palm_mm_left": err_palm[:, 0].astype(np.float32),
            "err_palm_mm_right": err_palm[:, 1].astype(np.float32),
            "err_local_dir_left": local_dir[:, 0].astype(np.float32),
            "err_local_dir_right": local_dir[:, 1].astype(np.float32),
            "err_ddq": err_ddq,
            "err_cam_pos_mm": err_cam_pos_mm,
            "err_cam_rot_deg": err_cam_rot_deg,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', type=str, default='allex',
                        help='Robot config name (configs/<robot_name>.yaml)')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Gaussian sigma for validity-aware keypoint smoothing')
    parser.add_argument('--min_seq_len', type=int, default=16,
                        help='Drop segments shorter than this many frames')
    parser.add_argument('--max_wrist_speed', type=float, default=4.0,
                        help='Wrist speed (m/s) above which a frame is downweighted')
    parser.add_argument('--pad_lens', type=int, nargs='+', default=[224, 1024, 3072],
                        help='IK padding lengths. A segment pads to the shortest that fits, '
                             'and a longer one is dropped')
    parser.add_argument('--wrist_dist_thr', type=float, default=0.9,
                        help='Wrong-hand filter: median wrist-to-camera distance (m)')
    parser.add_argument('--yaw_delta_thr', type=float, default=20.0,
                        help='Wrong-hand filter: guessed-vs-solved yaw delta (deg)')
    parser.add_argument('--cam_drift_thr', type=float, default=1.2,
                        help='Drift filter: max first-to-last camera translation (m)')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    robot_cfg = load_robot_config(args.robot_name)
    metadata = schema_metadata(robot_cfg)

    if osp.isdir(input_dir) or osp.isdir(input_dir + '_chunked'):
        chunked_root = input_dir + '_chunked'
        # Enumerate the chunked clips.
        clip_ids = partition(sorted(
            osp.basename(p) for p in glob.glob(osp.join(chunked_root, 'original', 'annot', '*'))
            if osp.isdir(p)), args.part, key=lambda c: c)
    else:
        assert osp.isfile(input_dir) and input_dir.lower().endswith('.mp4'), \
            (f"--input_dir {input_dir}: expected a clips directory, a path whose "
             f"{input_dir}_chunked directory exists, or one .mp4 file")
        chunked_root = osp.join(osp.dirname(input_dir), 'chunked')
        clip_ids = [osp.basename(input_dir)[:-4]]
    if len(clip_ids) == 0:
        return

    original_annot_dir = osp.join(chunked_root, 'original', 'annot')
    output_annot_dir = osp.join(chunked_root, args.robot_name, 'annot')

    def is_done(clip_id):
        return osp.exists(osp.join(output_annot_dir, f"{clip_id}.done"))

    print(f'Got {len(clip_ids)} clips in total.')
    clip_ids = [c for c in clip_ids if not is_done(c)]
    print(f'Processing {len(clip_ids)} clips (rest already done).')
    if len(clip_ids) == 0:
        return
    os.makedirs(output_annot_dir, exist_ok=True)

    calculator = ActionCalculator(
        robot_cfg=robot_cfg,
        state_slices=robot_cfg.state_slices,
        smooth_sigma=args.smooth_sigma,
        max_wrist_speed=args.max_wrist_speed,
        pad_lens=args.pad_lens,
    )

    for clip_id in tqdm(clip_ids, desc="clips", unit="clip", position=1, leave=True,
                         dynamic_ncols=True, disable=args.no_tqdm):
        clip_annot_dir = osp.join(original_annot_dir, clip_id)
        if not osp.exists(clip_annot_dir):
            mark_done(output_annot_dir, clip_id); continue

        segment_parquets = sorted(p for p in glob.glob(osp.join(clip_annot_dir, '*.parquet'))
                                  if not p.endswith('_narr.parquet'))
        narr_only = sorted(p for p in glob.glob(osp.join(clip_annot_dir, '*_narr.parquet'))
                           if not osp.exists(p[:-len('_narr.parquet')] + '.parquet'))
        assert not narr_only, \
            f"{narr_only[0]} has no <seg>.parquet beside it: stage 7 has not finished {clip_id}"
        if not segment_parquets:
            mark_done(output_annot_dir, clip_id); continue

        clip_output_dir = osp.join(output_annot_dir, clip_id)
        os.makedirs(clip_output_dir, exist_ok=True)
        remove_stale_temps(clip_output_dir)

        # Keep valid existing segments. The validator deletes corrupt ones.
        existing_stems = {
            osp.basename(f)[:-len('.parquet')]
            for f in glob.glob(osp.join(clip_output_dir, '*.parquet'))
            if validate_segment_outputs(annot_path=f)
        }

        for seg_parquet_path in tqdm(segment_parquets, desc=f"{clip_id}", unit="seg",
                                     position=0, leave=False, disable=args.no_tqdm):
            seg_stem = osp.basename(seg_parquet_path)[:-len('.parquet')]
            if seg_stem in existing_stems:
                continue

            rows = [decode_row(r) for r in pq.read_table(seg_parquet_path).to_pylist()]
            if len(rows) == 0:
                continue
            T = len(rows)

            if T < args.min_seq_len:
                tqdm.write(f"  [FILTERED] {seg_stem}: too short ({T} < {args.min_seq_len})")
                continue

            if T > max(args.pad_lens):
                tqdm.write(f"  [FILTERED] {seg_stem}: too long ({T} > {max(args.pad_lens)})")
                continue

            left_kpts = np.zeros((T, 21, 3), dtype=np.float32)
            right_kpts = np.zeros((T, 21, 3), dtype=np.float32)
            left_valid = np.zeros(T, dtype=bool)
            right_valid = np.zeros(T, dtype=bool)

            for t, row in enumerate(rows):
                for side, (kpts, valid) in enumerate(((left_kpts, left_valid),
                                                      (right_kpts, right_valid))):
                    hand = hand_for_side(row, side, require_kpts3d=True)
                    if hand is not None:
                        kpts[t] = np.asarray(hand["kpts3d"], dtype=np.float32).reshape(21, 3)
                        valid[t] = True

            # Missing camera poses get an identity placeholder, masked out below
            cam_poses_raw = [row.get("cam_pose") for row in rows]
            cam_valid = np.array([cp is not None for cp in cam_poses_raw], dtype=bool)
            cam_arr = np.stack([
                np.asarray(cp, dtype=np.float64).reshape(4, 4) if cp is not None
                else np.eye(4, dtype=np.float64)
                for cp in cam_poses_raw
            ])
            left_valid &= cam_valid
            right_valid &= cam_valid

            if not (np.any(left_valid) or np.any(right_valid)):
                continue

            is_drifted, cam_drift = check_camera_drift(
                cam_arr, cam_valid, cam_drift_thr=args.cam_drift_thr)
            if is_drifted:
                tqdm.write(f"  [FILTERED] {seg_stem}: camera drift "
                           f"({cam_drift:.2f}m > {args.cam_drift_thr}m)")
                continue

            result = calculator.process_segment(
                left_kpts, right_kpts, left_valid, right_valid, cam_arr, cam_valid)

            is_outlier, median_wd, yaw_delta = check_wrong_hand_outlier(
                left_kpts, right_kpts, left_valid, right_valid,
                cam_arr, cam_valid, result["optimized_offset_4dof"],
                wrist_dist_thr=args.wrist_dist_thr,
                yaw_delta_thr=args.yaw_delta_thr,
            )
            if is_outlier:
                tqdm.write(f"  [FILTERED] {seg_stem}: wrong-hand outlier "
                           f"(wrist_dist={median_wd:.3f}, yaw_delta={yaw_delta:.1f}deg)")
                continue

            state_eef = result["state_eef"]
            eef_left_dim = robot_cfg.eef_left_dim

            output_rows = []
            for t, row in enumerate(rows):
                output_rows.append({
                    "clip_id": row.get("clip_id"),
                    "frame_id": row.get("frame_id"),
                    "height": row.get("height"),
                    "width": row.get("width"),
                    "intr_model": row.get("intr_model"),
                    "fx": row.get("fx"), "fy": row.get("fy"),
                    "cx": row.get("cx"), "cy": row.get("cy"), "xi": row.get("xi"),
                    "pinhole_fx": row.get("pinhole_fx"), "pinhole_fy": row.get("pinhole_fy"),
                    "pinhole_cx": row.get("pinhole_cx"), "pinhole_cy": row.get("pinhole_cy"),
                    # World pose carried through unchanged. The rebased one is dense
                    # (gaps interpolated) and lives in its own column.
                    "cam_pose": row.get("cam_pose"),
                    "cam_pose_base": result["cam_poses"][t],
                    "narr": row.get("narr"),
                    "language": row.get("language"),
                    "state_qpos": result["state_qpos"][t],
                    "state_eef_left": state_eef[t, :eef_left_dim],
                    "state_eef_right": state_eef[t, eef_left_dim:],
                    "state_mask_left": bool(result["state_mask"][t, 0]),
                    "state_mask_right": bool(result["state_mask"][t, 1]),
                    "retarget_cost": result["retarget_cost"],
                    "err_tip_mm_left": result["err_tip_mm_left"][t],
                    "err_tip_mm_right": result["err_tip_mm_right"][t],
                    "err_palm_mm_left": float(result["err_palm_mm_left"][t]),
                    "err_palm_mm_right": float(result["err_palm_mm_right"][t]),
                    "err_local_dir_left": float(result["err_local_dir_left"][t]),
                    "err_local_dir_right": float(result["err_local_dir_right"][t]),
                    "err_ddq": result["err_ddq"][t],
                    "err_cam_pos_mm": float(result["err_cam_pos_mm"][t]),
                    "err_cam_rot_deg": float(result["err_cam_rot_deg"][t]),
                })

            rows_to_parquet(output_rows, osp.join(clip_output_dir, f"{seg_stem}.parquet"),
                            metadata=metadata)

        mark_done(output_annot_dir, clip_id)


if __name__ == "__main__":
    import signal
    # JAX can SIGABRT tearing down its CUDA context at exit. Work is already on disk by then.
    signal.signal(signal.SIGABRT, lambda *_: (sys.stdout.flush(), sys.stderr.flush(), os._exit(0)))
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

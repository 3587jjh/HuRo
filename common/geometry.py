"""Geometry helpers: bbox crop coverage, pinhole 3D->2D projection, SE(3) conversion and
gap interpolation of pose sequences, and Gaussian-weighted SLERP smoothing of rotations."""
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common.stats import gaussian_kernel


def oob_ratio_from_crop(bbox: np.ndarray, img_size: Tuple[int, int], box_dilate: float = 1.2) -> float:
    """Fraction (0..1) of a hand's square dilated crop that falls outside the image.
    The crop is a square of side max(w,h)*box_dilate centered on the bbox (HAWOR's crop)."""
    # bbox: [x1, y1, x2, y2], img_size: (H, W)
    x1, y1, x2, y2 = map(float, bbox)
    H, W = img_size
    w = x2 - x1
    h = y2 - y1
    assert w > 0 and h > 0

    size = max(w, h)
    L = size * float(box_dilate)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5

    x_min = cx - L * 0.5
    x_max = cx + L * 0.5
    y_min = cy - L * 0.5
    y_max = cy + L * 0.5

    inter_x_min = max(0.0, x_min)
    inter_y_min = max(0.0, y_min)
    inter_x_max = min(float(W), x_max)
    inter_y_max = min(float(H), y_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    full_area = L * L
    out_ratio = 1.0 - (inter_area / full_area)
    return float(np.clip(out_ratio, 0.0, 1.0))


def project_3d_kpts_to_2d(kpts_3d: np.ndarray, img_focal: float, frame_shape: Tuple[int, int],
                          cx: float = None, cy: float = None) -> np.ndarray:
    """Pinhole-project camera-frame 3D keypoints to (non-clipped, float) 2D pixels."""
    H, W = frame_shape
    fx = fy = float(img_focal)
    if cx is None: cx = W / 2.0
    if cy is None: cy = H / 2.0
    X = kpts_3d[:, 0]
    Y = kpts_3d[:, 1]
    Z = kpts_3d[:, 2]
    eps = 1e-8
    Z_safe = np.where(Z == 0, eps, Z)
    u = fx * (X / Z_safe) + cx
    v = fy * (Y / Z_safe) + cy
    return np.stack([u, v], axis=-1)


def wxyz_xyz_to_matrix(wxyz_xyz: np.ndarray) -> np.ndarray:
    """PyRoKi's (7,) [qw,qx,qy,qz, x,y,z] pose -> 4x4 SE(3) matrix."""
    w, x, y, z = wxyz_xyz[:4]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = wxyz_xyz[4:]
    return T


def interpolate_cam_poses(cam_poses: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill invalid camera poses: SLERP/lerp across interior gaps, nearest-valid copy at
    the edges. cam_poses: (T, 4, 4), valid_mask: (T,) bool."""
    if valid_mask.all() or not valid_mask.any():
        return cam_poses

    valid_idx = np.where(valid_mask)[0]
    cam_poses = cam_poses.copy()

    first_valid = valid_idx[0]
    if first_valid > 0:
        cam_poses[:first_valid] = cam_poses[first_valid]
    last_valid = valid_idx[-1]
    if last_valid < len(cam_poses) - 1:
        cam_poses[last_valid + 1:] = cam_poses[last_valid]

    for gap_start_i in range(len(valid_idx) - 1):
        i0 = valid_idx[gap_start_i]
        i1 = valid_idx[gap_start_i + 1]
        if i1 - i0 <= 1:
            continue

        rots = Rotation.concatenate([
            Rotation.from_matrix(cam_poses[i0, :3, :3]),
            Rotation.from_matrix(cam_poses[i1, :3, :3]),
        ])
        slerp = Slerp([0.0, 1.0], rots)
        t0 = cam_poses[i0, :3, 3]
        t1 = cam_poses[i1, :3, 3]

        for j in range(i0 + 1, i1):
            alpha = (j - i0) / (i1 - i0)
            cam_poses[j, :3, :3] = slerp([alpha])[0].as_matrix()
            cam_poses[j, :3, 3] = (1 - alpha) * t0 + alpha * t1
            cam_poses[j, 3, :] = [0, 0, 0, 1]

    return cam_poses


def gaussian_slerp_smoothing(rot_mats: np.ndarray, sigma: float = 2.0,
                             kernel_size: int = 9) -> np.ndarray:
    """Gaussian-weighted SLERP low-pass of a rotation-matrix sequence. (N,3,3) -> (N,3,3)."""
    if len(rot_mats) == 0:
        return rot_mats
    if kernel_size % 2 != 1:
        raise ValueError("Kernel size must be odd")

    half_k = kernel_size // 2
    N = len(rot_mats)
    quats = Rotation.from_matrix(rot_mats).as_quat()

    # hemisphere correction
    quats_fixed = [quats[0]]
    for i in range(1, N):
        q = quats[i]
        if np.dot(q, quats_fixed[-1]) < 0:
            q = -q
        quats_fixed.append(q)
    quats_fixed = np.array(quats_fixed)

    weights = gaussian_kernel(kernel_size, sigma)
    smoothed_rots = []
    for i in range(N):
        start = max(0, i - half_k)
        end = min(N, i + half_k + 1)
        local_quats = quats_fixed[start:end]
        local_weights = weights[half_k - (i - start): half_k + (end - i)]
        local_weights = local_weights / local_weights.sum()

        r_avg = Rotation.from_quat(local_quats[0])
        for j in range(1, len(local_quats)):
            r_next = Rotation.from_quat(local_quats[j])
            current_w = local_weights[j] / (local_weights[:j + 1].sum())
            r_avg = Slerp([0, 1], Rotation.concatenate([r_avg, r_next]))([current_w])[0]
        smoothed_rots.append(r_avg.as_matrix())

    return np.stack(smoothed_rots)

"""Camera-model helpers: intrinsics validation, ensemble selection, and MEI/UCM
undistortion (fx, fy, cx, cy, xi). Requires cv2.omnidir (opencv-contrib builds only)."""
import numpy as np
import cv2

from common.stats import mad_filter_median


def is_valid_intrinsics_light(frame_shape, intr, pp_tol=0.30, fhat_bounds=(0.1,8.0), ar_bounds=(0.33,3.0), xi_bounds=(-0.01,3.5)):
    H,W = frame_shape
    fx,fy,cx,cy,xi = map(float, intr)
    if not np.isfinite([fx,fy,cx,cy,xi]).all(): return False
    if fx<=0 or fy<=0: return False
    if abs(cx-W/2)/W > pp_tol or abs(cy-H/2)/H > pp_tol: return False
    fhat = 0.5*(fx/W + fy/H)
    if not (fhat_bounds[0] <= fhat <= fhat_bounds[1]): return False
    r = fx/max(fy,1e-6)
    if not (ar_bounds[0] <= r <= ar_bounds[1]): return False
    if not (xi_bounds[0] <= xi <= xi_bounds[1]): return False
    return True


def need_undistort(intrinsics, frame_shape, thr_px=0.5, xi_eps=1e-6, xi_neg_eps=0.01, border_frac=0.02):
    H, W = frame_shape
    fx, fy, cx, cy, xi = map(float, intrinsics)
    if xi < 0.0:
        if -xi > xi_neg_eps: return None
        return False
    if xi < xi_eps: return False

    xi = np.array([[float(xi)]], dtype=np.float64)
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], np.float64)
    D = np.zeros((1, 4), np.float64)
    R = np.eye(3, dtype=np.float64)
    m1, m2 = cv2.omnidir.initUndistortRectifyMap(
        K, D, xi, R, K, (W, H), cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    if not (np.isfinite(m1).all() and np.isfinite(m2).all()): return None

    b = int(round(max(H, W) * border_frac))
    ys = slice(b, H - b if H - b > b else H)
    xs = slice(b, W - b if W - b > b else W)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32),
                    np.arange(H, dtype=np.float32))
    disp = np.sqrt((m1 - X)**2 + (m2 - Y)**2)[ys, xs]
    p99_px = float(np.percentile(disp, 99.0))
    return p99_px >= float(thr_px)


def build_undistort_maps(frame_shape, intrinsics, k=0.5, auto=False, tol=0.01, k_min=0.15):
    """Build cv2.remap maps that undistort a MEI/UCM frame to a pinhole frame.
    Returns (map1, map2, k_used, intrinsics_new = [fx, fy, cx, cy, 0])."""
    fx, fy, cx, cy, xi = map(float, intrinsics)
    xi = np.array([[float(xi)]], dtype=np.float64)
    H, W = frame_shape

    K  = np.array([[fx, 0,  cx],
                   [0,  fy, cy],
                   [0,   0,  1]], np.float64)
    D  = np.zeros((1, 4), np.float64)
    R  = np.eye(3, dtype=np.float64)

    def _is_ok(k_try, margin=1.0):
        Knew = K.copy()
        Knew[0,0] *= k_try
        Knew[1,1] *= k_try
        Knew[0,2], Knew[1,2] = W/2.0, H/2.0
        m1, m2 = cv2.omnidir.initUndistortRectifyMap(
            K, D, xi, R, Knew, (W, H), cv2.CV_32FC1,
            cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        if not (np.isfinite(m1).all() and np.isfinite(m2).all()): return False
        x_valid = (m1 >= margin) & (m1 <= (W - 1 - margin))
        y_valid = (m2 >= margin) & (m2 <= (H - 1 - margin))
        valid  = x_valid & y_valid
        bottom_valid = valid[-1, :] # (W,)
        return bottom_valid.all()

    if auto:
        lo, hi = max(1e-3, k_min - tol), 2.50
        best = hi
        while (hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            if _is_ok(mid): best = hi = mid
            else: lo = mid
        k_used = float(best)
    else:
        k_used = float(k)
    if not _is_ok(k_used) or k_used < k_min:
        raise ValueError('Invalid intrinsics or extreme FOV (k too small)')

    Knew = K.copy()
    Knew[0,0] *= k_used
    Knew[1,1] *= k_used
    Knew[0,2], Knew[1,2] = W/2.0, H/2.0
    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K, D, xi, R, Knew, (W, H), cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    intrinsics_new = np.array([Knew[0,0], Knew[1,1], Knew[0,2], Knew[1,2], 0.0], np.float64)
    return map1, map2, k_used, intrinsics_new


def undistort_apply(frame, map1, map2):
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _stack_rows5(intrs):
    rows = []
    for v in intrs:
        if v is None: continue
        a = np.asarray(v, dtype=float)
        if a.ndim == 0: continue
        a = a.reshape(-1)
        if a.size != 5: continue
        if np.isfinite(a).all(): rows.append(a)
    if not rows: return np.empty((0, 5), dtype=float)
    return np.vstack(rows).astype(np.float64)


def early_stop_intrinsics(
    intrs, width, height,
    min_k=3,
    fx_fy_rel_tol=0.06,
    c_rel_tol_x=0.03,
    c_rel_tol_y=0.04,
    xi_abs_tol=0.10
):
    intrs = _stack_rows5(intrs)
    if intrs.shape[0] < min_k: return False, None
    ref = np.median(intrs[-min_k:], axis=0)

    dfx = fx_fy_rel_tol * max(abs(ref[0]), 1e-6)
    dfy = fx_fy_rel_tol * max(abs(ref[1]), 1e-6)
    dcx = c_rel_tol_x * float(width)
    dcy = c_rel_tol_y * float(height)
    dxi = xi_abs_tol

    diff = np.abs(intrs - ref)
    mask = (
        (diff[:,0] <= dfx) &
        (diff[:,1] <= dfy) &
        (diff[:,2] <= dcx) &
        (diff[:,3] <= dcy) &
        (diff[:,4] <= dxi)
    )
    if mask.sum() < min_k:
        return False, None

    cluster = intrs[mask]
    med = np.median(cluster, axis=0)
    ok_ff = (
        np.max(np.abs(cluster[:,0]-med[0])) <= fx_fy_rel_tol * max(abs(med[0]), 1e-6) and
        np.max(np.abs(cluster[:,1]-med[1])) <= fx_fy_rel_tol * max(abs(med[1]), 1e-6)
    )
    ok_c = (
        np.max(np.abs(cluster[:,2]-med[2])) <= c_rel_tol_x * float(width) and
        np.max(np.abs(cluster[:,3]-med[3])) <= c_rel_tol_y * float(height)
    )
    ok_xi = np.max(np.abs(cluster[:,4]-med[4])) <= xi_abs_tol
    return (ok_ff and ok_c and ok_xi), (med.astype(float) if (ok_ff and ok_c and ok_xi) else None)


def intrinsics_selection(intrs):
    # --- MAD outlier cutting + median ---
    intrs = _stack_rows5(intrs)
    return mad_filter_median(intrs)


def pinhole_from_row(row, where="row"):
    """(fx, fy, cx, cy) of the undistorted frames, from the row's pinhole_* columns. A table
    written without those columns gets the pinhole re-derived from its raw camera model."""
    if row.get("pinhole_fx") is not None:
        pinhole = [row["pinhole_fx"], row["pinhole_fy"], row["pinhole_cx"], row["pinhole_cy"]]
    else:
        frame_shape = (int(row["height"]), int(row["width"]))
        raw = [row["fx"], row["fy"], row["cx"], row["cy"], row["xi"]]
        need_undist = need_undistort(raw, frame_shape)
        assert need_undist is not None, f"{where}: intrinsics {raw} cannot be undistorted"
        pinhole = build_undistort_maps(frame_shape, raw, auto=True)[3] if need_undist else raw
    fx, fy, cx, cy = (float(v) for v in pinhole[:4])
    assert np.isfinite([fx, fy, cx, cy]).all() and fx > 0 and fy > 0, \
        f"{where}: invalid pinhole {pinhole}"
    return fx, fy, cx, cy

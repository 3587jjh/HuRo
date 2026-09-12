# Stage 1 — camera intrinsics (droidcalib, with anycalib fallback).
# Reads raw clips and writes per-clip JSON
# {fx, fy, cx, cy, xi, H, W, model} ({} if no valid estimate).
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage1_annot_intrinsics.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import av
import cv2
cv2.setNumThreads(0) # prevent corruption with av
import os
import glob
import argparse
import json
import kornia
from tqdm import tqdm
import sys
from pathlib import Path

# repo root on sys.path so `common` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import setup_submodule_imports, partition, DROIDCALIB_WEIGHTS, ANYCALIB_WEIGHTS
from common.camera import need_undistort, is_valid_intrinsics_light, early_stop_intrinsics, intrinsics_selection

setup_submodule_imports()
from droid import Droid
from anycalib import AnyCalib

NO_TQDM = False


def log(msg):
    if NO_TQDM: print(msg)
    else: tqdm.write(msg)


def get_args_default():
    global NO_TQDM
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--motion_size', type=int, default=60)
    parser.add_argument('--motion_min_gap', type=int, default=5)
    parser.add_argument('--lap_top_ratio', type=float, default=0.3)
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()
    assert args.motion_size > 0 and args.motion_min_gap > 0
    if args.no_tqdm: NO_TQDM = True
    return args


def get_args_droidcalib():
    args = argparse.Namespace(
        opt_intr=True,
        camera_model="mei",
        weights=str(DROIDCALIB_WEIGHTS),
        buffer=1024,
        image_size_target=[517, 384],
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, stereo=False, disable_vis=True
    )
    return args


def frame_stream_all_droidcalib(path_mp4, image_size):
    with av.open(path_mp4, "r") as reader:
        s = reader.streams.video[0]
        total = s.frames or (int(float(s.duration * s.time_base) * 30 + 0.5) if s.duration and s.time_base else None) # assume 30 fps
        for i, frame in enumerate(tqdm(reader.decode(video=0), total=total, desc=f"detecting motion frames",
            unit="frame", position=0, leave=False, dynamic_ncols=True, disable=NO_TQDM)):
            frame = frame.to_ndarray(format="bgr24") # (H,W,3), uint8, BGR
            if i == 0:
                wd, ht = image_size
                h0, w0, _ = frame.shape
                scale = np.sqrt((ht * wd) / (h0 * w0))
                h1 = int(h0 * scale)
                w1 = int(w0 * scale)
                h2, w2 = h1-h1%8, w1-w1%8

            frame = cv2.resize(frame, (w1, h1))
            frame = frame[:h2, :w2]
            frame = torch.as_tensor(frame).permute(2, 0, 1).contiguous().to('cuda')
            yield frame[None]


def frame_stream_droidcalib(frames, image_size, mei=False):
    wd, ht = image_size
    h0, w0, _ = frames[0].shape
    fx = fy = (w0 + h0) / 2.0
    cx, cy = w0 / 2.0, h0 / 2.0

    scale = np.sqrt((ht * wd) / (h0 * w0))
    h1 = int(h0 * scale)
    w1 = int(w0 * scale)
    h2, w2 = h1-h1%8, w1-w1%8

    size_factor = [(w2 / w0), (h2 / h0)]
    intrinsics = torch.tensor(
        [fx*size_factor[0], fy*size_factor[1], cx*size_factor[0], cy*size_factor[1]] + ([0.0] if mei else []),
        dtype=torch.float32
    )
    for t, frame in enumerate(frames):
        frame = cv2.resize(frame, (w1, h1))
        frame = frame[:h2, :w2]
        frame = torch.as_tensor(frame).permute(2, 0, 1).contiguous()
        yield t, frame[None], intrinsics.clone(), size_factor


def motion_filter(path_mp4):
    args_droidcalib = get_args_droidcalib()
    is_moving = []
    laps = []
    droid = None

    for t, frame in enumerate(frame_stream_all_droidcalib(path_mp4, args_droidcalib.image_size_target)):
        # frame: (1,3,H,W) uint8 tensor, BGR, cuda
        if droid is None:
            args_droidcalib.image_size = [frame.shape[2], frame.shape[3]]
            droid = Droid(args_droidcalib)
        is_moving.append(droid.filterx.track(t, frame, custom=True))

        gray = kornia.color.bgr_to_grayscale(frame.float())
        lap = kornia.filters.laplacian(gray, kernel_size=3).var().item()
        laps.append(lap)

    del droid
    torch.cuda.empty_cache()
    return np.array(is_moving), np.array(laps)


def thin_mask_by_mingap(mask, min_gap):
    mask = np.asarray(mask, dtype=bool).ravel()
    if not mask.any() or min_gap <= 1: return mask
    ones = np.flatnonzero(mask)
    kept = []
    last = -10**9
    for idx in ones:
        if idx - last >= min_gap:
            kept.append(idx)
            last = idx
    thin = np.zeros_like(mask, dtype=bool)
    thin[np.asarray(kept, dtype=int)] = True
    return thin


def select_windows_motion_cover(mask, motion_size):
    mask = np.asarray(mask, dtype=bool).ravel()
    N = mask.size
    if N == 0 or not mask.any(): return []

    ones = np.flatnonzero(mask)
    out = []
    i = 0

    while i < ones.size:
        remaining = ones.size - i
        if remaining < motion_size:
            if remaining < motion_size // 2: break
            j = i + remaining - 1
        else:
            j = i + motion_size - 1
        s = int(ones[i])
        e = int(ones[j]) + 1
        out.append((s, e))
        i = j + 1
    return out


def subsample_windows_uniform(windows, max_windows):
    """windows: sorted [(s, e), ...]."""
    if max_windows is None or max_windows <= 0:
        return windows
    if len(windows) <= max_windows:
        return windows

    idxs = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
    idxs = sorted(set(int(i) for i in idxs))
    return [windows[i] for i in idxs]


def iter_video_windows_droidcalib(path_mp4, mask, motion_size, windows=None):
    if windows is None: windows = select_windows_motion_cover(mask, motion_size)  # [(s,e), ...]
    if not windows: return

    with av.open(path_mp4, "r") as reader:
        w = 0
        s, e = windows[w]
        buf = []

        with tqdm(total=len(windows), desc=f"windows", unit="win",
            position=0, leave=False, dynamic_ncols=True, disable=NO_TQDM) as wpbar:
            for idx, frame in enumerate(reader.decode(video=0)):
                if idx == e:
                    if len(buf) > 0:
                        wpbar.update(1)
                        yield buf
                    buf = []
                    w += 1
                    if w >= len(windows): break
                    s, e = windows[w]

                if s <= idx < e and mask[idx]:
                    img = frame.to_ndarray(format="bgr24")  # (H,W,3), uint8, BGR
                    buf.append(img)

            if len(buf) > 0:
                wpbar.update(1)
                yield buf


def iter_video_frames_anycalib(path_mp4, is_moving, laps, lap_top_ratio, lap_min_size):

    static_indices = np.flatnonzero(~is_moving)
    num_static = len(static_indices)
    yield_mask = np.zeros_like(is_moving, dtype=bool)
    num_chosen = 0

    n_select = int(num_static * lap_top_ratio)
    n_select = max(n_select, min(lap_min_size, num_static))
    if n_select > 0:
        vals = laps[static_indices]
        vals = np.nan_to_num(vals, nan=-np.inf)
        top_local = np.argpartition(vals, -n_select)[-n_select:]
        yield_mask = np.zeros_like(is_moving, dtype=bool)
        yield_mask[static_indices[top_local]] = True
        num_chosen = yield_mask.sum()

    log(f"total: {len(is_moving)} | static: {num_static} | selected: {num_chosen}")
    if num_chosen == 0: return

    with av.open(path_mp4, "r") as reader:
        s = reader.streams.video[0]
        with tqdm(total=num_chosen, desc=f"frames", unit="frame", position=0, leave=False, dynamic_ncols=True, disable=NO_TQDM) as wpbar:
            for idx, frame in enumerate(reader.decode(video=0)):
                if yield_mask[idx]:
                    img = frame.to_ndarray(format="rgb24") # (H,W,3), uint8, RGB
                    img = torch.from_numpy(img).to(device='cuda', dtype=torch.float32).permute(2,0,1).contiguous() / 255.0
                    wpbar.update(1)
                    yield img


def pred_droidcalib(frames):
    # frames: list[np.ndarray(H,W,3), BGR, uint8], len=motion_size
    args_droidcalib = get_args_droidcalib()
    droid = None
    for (t, frame, intrinsics, sf) in frame_stream_droidcalib(frames, args_droidcalib.image_size_target, mei=True):
        if droid is None:
            args_droidcalib.image_size = [frame.shape[2], frame.shape[3]]
            droid = Droid(args_droidcalib)
        droid.track(t, frame, intrinsics=intrinsics)
    _, intr = droid.terminate()
    del droid
    torch.cuda.empty_cache()
    intr[0:4:2] /= sf[0]
    intr[1:4:2] /= sf[1]
    return intr


def pred_anycalib(model, frame):
    # frame: torch (3,H,W), RGB, [0,1]
    intr = model.predict(frame, cam_id="ucm")["intrinsics"].detach().to("cpu").numpy()
    return intr


def calc_intrinsics_droidcalib(path_mp4, is_moving, motion_size, max_windows=20, min_k=3):
    """Windowed droidcalib estimation. Returns (intr, (H, W)). intr is None when no
    stable estimate emerges (the caller falls back to anycalib)."""
    intrs = []
    num_hit = 0

    with av.open(path_mp4, "r") as reader:
        s = reader.streams.video[0]
        W = s.codec_context.width or s.width
        H = s.codec_context.height or s.height
        assert W and H

    windows = select_windows_motion_cover(is_moving, motion_size)
    num_windows = len(windows)
    if num_windows < min_k:
        log(f"too few windows.")
        return None, (H, W)

    windows_used = subsample_windows_uniform(windows, max_windows)
    for i, frames in enumerate(iter_video_windows_droidcalib(path_mp4, is_moving, motion_size, windows=windows_used)):
        try:
            intr = pred_droidcalib(frames)
            if not is_valid_intrinsics_light((H, W), intr):
                log(f"<window {i+1}> invalid intrinsics")
                continue
            intrs.append(intr) # np array (5,), original image size
            log("<window {}> fx = {:.1f}, fy = {:.1f} | cx = {:.1f}, cy = {:.1f} | xi (mei) = {:.3f}".format(i+1, *intr))

            stop, intr_est = early_stop_intrinsics(intrs, W, H, min_k=min_k)
            if stop:
                num_hit += 1
                log(f"stability hit {num_hit}/2")
                if num_hit >= 2:
                    log("early stop: intrinsics stabilized.")
                    return intr_est, (H, W)
        except Exception:
            log(f'<window {i+1}> unstable intrinsics prediction')
    if num_windows == min_k and num_hit == 1:
        log('exceptional rule.')
        return intr_est, (H, W)
    return None, (H, W)


def calc_intrinsics_anycalib(anycalib, path_mp4, is_moving, laps, lap_top_ratio, lap_min_size=50):
    with av.open(path_mp4, "r") as reader:
        s = reader.streams.video[0]
        W = s.codec_context.width or s.width
        H = s.codec_context.height or s.height
        assert W and H

    intrs = []
    for i, frame in enumerate(iter_video_frames_anycalib(path_mp4, is_moving, laps, lap_top_ratio, lap_min_size=lap_min_size)):
        try:
            intr = pred_anycalib(anycalib, frame)
            if not is_valid_intrinsics_light((H, W), intr): continue
            intrs.append(intr) # np array (5,), original image size
        except Exception:
            continue
    intr = intrinsics_selection(intrs)
    return intr, (H, W)


def main():
    args = get_args_default()

    if os.path.isdir(args.input_dir):
        paths_mp4 = sorted(glob.glob(os.path.join(args.input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part)
        if len(paths_mp4) == 0: return
        output_dir = os.path.normpath(args.input_dir) + '_intr'
        os.makedirs(output_dir, exist_ok=True)
    else:
        assert os.path.isfile(args.input_dir)
        assert args.input_dir.lower().endswith('.mp4')
        paths_mp4 = [args.input_dir]
        output_dir = os.path.join(os.path.dirname(args.input_dir), 'intr')
        os.makedirs(output_dir, exist_ok=True)

    log(f'\n#######\n\033[2;32mGot {len(paths_mp4)} mp4 files in total.\033[0m')
    paths_mp4_new = []
    for path_mp4 in paths_mp4:
        if not os.path.exists(os.path.join(output_dir, f"{os.path.basename(path_mp4)[:-4]}.json")):
            paths_mp4_new.append(path_mp4)
    log(f'\033[2;32mSkipping {len(paths_mp4) - len(paths_mp4_new)} mp4 files (already processed)\033[0m\n#######\n')
    paths_mp4 = paths_mp4_new
    if len(paths_mp4) == 0: return

    # anycalib is loaded on the first clip that actually falls back to it
    anycalib = None

    for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1, leave=True, dynamic_ncols=True, disable=NO_TQDM):
        clip_id = os.path.basename(path_mp4)[:-4]
        log(f'\033[2;36mprocessing {clip_id}.mp4 ...\033[0m')

        with torch.inference_mode():
            is_moving, laps = motion_filter(path_mp4)

            is_moving_thin = thin_mask_by_mingap(is_moving, args.motion_min_gap)
            intr, frame_shape = calc_intrinsics_droidcalib(path_mp4, is_moving_thin, args.motion_size)
            model_name = 'droidcalib'

            if intr is None:
                log('cannot find stable intrinsics. fallback to anycalib ...')
                if anycalib is None:
                    anycalib = AnyCalib(model_id="anycalib_gen", local_path=str(ANYCALIB_WEIGHTS)).cuda().eval()
                intr, frame_shape = calc_intrinsics_anycalib(anycalib, path_mp4, is_moving, laps, args.lap_top_ratio)
                model_name = 'anycalib'

        if intr is not None:
            need_undist = need_undistort(intr, frame_shape)
            log(f"\n------ {model_name} ------")
            log(f"resolution | {frame_shape[1]} x {frame_shape[0]}")
            log("intrinsics | fx = {:.1f}, fy = {:.1f} | cx = {:.1f}, cy = {:.1f} | xi (mei) = {:.3f}".format(*intr))
            log(f"need_undist: {need_undist}")
            log("------------------------\n")
        else:
            log('cannot find any valid intrinsics')

        out_path = os.path.join(output_dir, f"{clip_id}.json")
        tmp_path = out_path + ".tmp"
        data = {}
        if intr is not None:
            keys = ['fx', 'fy', 'cx', 'cy', 'xi']
            data = dict(zip(keys, map(float, intr)))
            assert frame_shape
            H,W = frame_shape
            data['H'], data['W'] = H,W
            data['model'] = model_name
        with open(tmp_path, "w", encoding="utf-8") as f: json.dump(data, f)
        os.replace(tmp_path, out_path)


if __name__ == '__main__':
    main()

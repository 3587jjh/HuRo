# Stage 4 — 3D hand keypoints, MANO parameters and hand masks (HAWOR).
# Reads stage 3's shards and writes the same rows + per-hand kpts3d / wrist_rot / finger_rot /
# hand_mask and a null cam_pose for stage 5 to fill.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage4_annot_hand.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import os, os.path as osp, glob, argparse
import numpy as np
import torch
import av
from tqdm import tqdm
import sys
from pathlib import Path

# repo root on sys.path so `common` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition, HAWOR_CKPT, setup_hawor_imports
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.geometry import oob_ratio_from_crop, project_3d_kpts_to_2d
from common.io import ParquetReader, write_clip_chunks, mark_done

setup_hawor_imports()
from scripts.scripts_test_video.hawor_video import load_hawor, estimate_3d_hand_keypoints
from hawor.utils.process import get_mano_faces_closed
from lib.vis.renderer import Renderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--thresh_oob', type=float, default=0.10)
    parser.add_argument('--chunk_size', type=int, default=5000, help='frames per parquet chunk')
    parser.add_argument('--gap_fill', type=int, default=3, help='Max consecutive missing frames to bridge (sequence ends at gap_fill+1)')
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    if osp.isdir(input_dir):
        paths_mp4 = sorted(glob.glob(osp.join(input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part)
        if len(paths_mp4) == 0: return
        contact_dir = input_dir + '_contact_refined'
        output_dir = input_dir + '_hand'
    else:
        assert osp.isfile(input_dir)
        assert input_dir.lower().endswith('.mp4')
        paths_mp4 = [input_dir]
        base_dir = osp.dirname(input_dir)
        contact_dir = osp.join(base_dir, 'contact_refined')
        output_dir = osp.join(base_dir, 'hand')

    def is_done(path_mp4):
        clip_id = osp.basename(path_mp4)[:-4]
        return osp.exists(osp.join(output_dir, f"{clip_id}.done"))

    print(f'Got {len(paths_mp4)} mp4 files in total.')
    paths_mp4_new = []
    for path_mp4 in paths_mp4:
        if not is_done(path_mp4):
            paths_mp4_new.append(path_mp4)
    print(f'Skipping {len(paths_mp4) - len(paths_mp4_new)} mp4 files (already processed)')
    paths_mp4 = paths_mp4_new
    if len(paths_mp4) == 0: return
    os.makedirs(output_dir, exist_ok=True)

    hawor, _ = load_hawor(str(HAWOR_CKPT))
    hawor = hawor.to('cuda')
    hawor.eval()

    # MANO faces with wrist closure
    faces_right, faces_left = get_mano_faces_closed()
    faces_right_t = torch.from_numpy(faces_right.astype(np.int64)).cuda()
    faces_left_t = torch.from_numpy(faces_left.astype(np.int64)).cuda()
    mano_faces_t = {0: faces_left_t, 1: faces_right_t}  # side -> faces tensor
    verts_color = (torch.tensor([0, 0, 255, 255]) / 255).unsqueeze(0).cuda()  # (1, 4)

    seq_len = args.seq_len
    assert seq_len > 2 and seq_len % 2 == 0
    stride = (seq_len * 3) // 4 # overlap ratio = 0.25

    with torch.inference_mode():
        for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1, leave=True, dynamic_ncols=True):
            clip_id = osp.basename(path_mp4)[:-4]

            # read stage 3's refined-contact shards (intrinsics carried as columns)
            try: hos_reader = ParquetReader(contact_dir, clip_id)
            except FileNotFoundError: mark_done(output_dir, clip_id); continue
            contact_ids = hos_reader.get_frame_ids()
            if not contact_ids: mark_done(output_dir, clip_id); continue

            # intrinsics from carried columns. Undistort -> pinhole (matches detection space)
            meta = hos_reader.get(contact_ids[0])
            intr = np.array([meta['fx'], meta['fy'], meta['cx'], meta['cy'], meta['xi']], dtype=np.float64)
            image_h, image_w = int(meta['height']), int(meta['width'])
            frame_shape = (image_h, image_w)
            need_undist = need_undistort(intr, frame_shape)
            if need_undist is None: mark_done(output_dir, clip_id); continue
            if need_undist:
                try: map1, map2, _, intr = build_undistort_maps(frame_shape, intr, auto=True)
                except Exception: mark_done(output_dir, clip_id); continue
            img_focal = 0.5 * (intr[0] + intr[1])
            img_cx, img_cy = float(intr[2]), float(intr[3])

            # mask renderer (identity camera, pytorch3d rasterizer)
            mask_renderer = Renderer(image_w, image_h, img_focal, 'cuda',
                                     bin_size=128, max_faces_per_bin=20000)
            cam_R = torch.eye(3).unsqueeze(0).cuda()
            cam_T = torch.zeros(1, 3).cuda()
            K_real = torch.tensor([[img_focal, 0, img_cx],
                                   [0, img_focal, img_cy],
                                   [0, 0, 1]], dtype=torch.float32).unsqueeze(0).cuda()
            mask_cameras, mask_lights = mask_renderer.create_camera_from_cv(cam_R, cam_T, K=K_real)
            n_mask_pixels = image_h * image_w

            # frame_id -> hand_idx -> accumulators (averaged across overlapping windows)
            frame2kpts3d = defaultdict(lambda: defaultdict(lambda: {"sum": None, "cnt": 0}))
            frame2mano = defaultdict(lambda: defaultdict(lambda: {"orient_sum": None, "pose_sum": None, "cnt": 0}))
            frame2hand_mask = defaultdict(dict)  # frame_id -> hand_idx -> packed bitarray (OR-merged)

            def _new_side_state():
                return {"active": False, "seq_id": 0, "total_len": 0, "frames": [], "imgs": [],
                        "bboxes": [], "hand_indices": [], "next_start_idx": 0,
                        "last_start_fid": None, "gap": 0}
            side_states = {0: _new_side_state(), 1: _new_side_state()}  # 0 left, 1 right

            def _clear_side_buffer(side):
                st = side_states[side]
                st["frames"].clear(); st["imgs"].clear(); st["bboxes"].clear(); st["hand_indices"].clear()
                st["next_start_idx"] = 0; st["last_start_fid"] = None; st["gap"] = 0

            def _start_new_sequence_if_needed(side):
                st = side_states[side]
                if not st["active"]:
                    st["active"] = True
                    st["seq_id"] += 1
                    st["total_len"] = 0
                    _clear_side_buffer(side)

            def _run_hawor_window(side, frame_ids, imgs, bboxes, hand_indices):
                # skip window if accumulated gaps made its temporal span too wide
                max_span = (seq_len - 1) + args.gap_fill
                if frame_ids[-1] - frame_ids[0] > max_span:
                    return
                kpts3d_seq, verts3d_seq, wrist_rot_seq, finger_rot_seq = estimate_3d_hand_keypoints(hawor, imgs, bboxes, side, img_focal)
                assert kpts3d_seq.shape[0] == seq_len
                st = side_states[side]
                if frame_ids: st["last_start_fid"] = int(frame_ids[0])

                # correct HAWOR's centered-PP 3D output to real-PP camera frame
                pp_dx = (image_w / 2.0 - img_cx) / img_focal
                pp_dy = (image_h / 2.0 - img_cy) / img_focal
                if pp_dx != 0.0 or pp_dy != 0.0:
                    kpts3d_seq[:, :, 0] += kpts3d_seq[:, :, 2] * pp_dx
                    kpts3d_seq[:, :, 1] += kpts3d_seq[:, :, 2] * pp_dy
                    verts3d_seq[:, :, 0] += verts3d_seq[:, :, 2] * pp_dx
                    verts3d_seq[:, :, 1] += verts3d_seq[:, :, 2] * pp_dy

                for fid, h_idx, kpts in zip(frame_ids, hand_indices, kpts3d_seq):
                    acc = frame2kpts3d[fid][h_idx]
                    kpts = np.asarray(kpts, dtype=np.float32)
                    if acc["sum"] is None: acc["sum"] = kpts.copy(); acc["cnt"] = 1
                    else: acc["sum"] += kpts; acc["cnt"] += 1

                for fid, h_idx, wr, fr in zip(frame_ids, hand_indices, wrist_rot_seq, finger_rot_seq):
                    acc_m = frame2mano[fid][h_idx]
                    wr = np.asarray(wr, dtype=np.float32)
                    fr = np.asarray(fr, dtype=np.float32)
                    if acc_m["orient_sum"] is None:
                        acc_m["orient_sum"] = wr.copy(); acc_m["pose_sum"] = fr.copy(); acc_m["cnt"] = 1
                    else:
                        acc_m["orient_sum"] += wr; acc_m["pose_sum"] += fr; acc_m["cnt"] += 1

                faces_t = mano_faces_t[side]
                verts_t = torch.from_numpy(verts3d_seq).float()  # (T, V, 3)
                for i, (fid, h_idx) in enumerate(zip(frame_ids, hand_indices)):
                    vi = verts_t[[i]].cuda()  # (1, V, 3)
                    _, mask = mask_renderer.render_multiple(
                        vi.unsqueeze(0), faces_t, verts_color, mask_cameras, mask_lights
                    )
                    mask_packed = np.packbits(mask.reshape(-1), bitorder="big")
                    if h_idx not in frame2hand_mask[fid]:
                        frame2hand_mask[fid][h_idx] = mask_packed
                    else:
                        np.bitwise_or(frame2hand_mask[fid][h_idx], mask_packed,
                                      out=frame2hand_mask[fid][h_idx])

            def _run_hawor_if_needed(side):
                st = side_states[side]
                if len(st["frames"]) - st["next_start_idx"] >= seq_len:
                    s_idx = st["next_start_idx"]
                    e_idx = s_idx + seq_len
                    _run_hawor_window(side, st["frames"][s_idx:e_idx], st["imgs"][s_idx:e_idx],
                                      st["bboxes"][s_idx:e_idx], st["hand_indices"][s_idx:e_idx])
                    st["next_start_idx"] += stride

                if len(st["frames"]) > seq_len:
                    st["frames"] = st["frames"][1:]
                    st["imgs"] = st["imgs"][1:]
                    st["bboxes"] = st["bboxes"][1:]
                    st["hand_indices"] = st["hand_indices"][1:]
                    st["next_start_idx"] -= 1

            def _end_sequence(side):
                st = side_states[side]
                if not st["active"]: return
                if len(st["frames"]) >= seq_len:
                    last_start_idx = len(st["frames"]) - seq_len
                    final_first_fid = st["frames"][last_start_idx]
                    if st.get("last_start_fid") != final_first_fid:
                        _run_hawor_window(side, st["frames"][last_start_idx:last_start_idx + seq_len],
                                          st["imgs"][last_start_idx:last_start_idx + seq_len],
                                          st["bboxes"][last_start_idx:last_start_idx + seq_len],
                                          st["hand_indices"][last_start_idx:last_start_idx + seq_len])
                st["active"] = False
                st["total_len"] = 0
                _clear_side_buffer(side)

            def _end_sequence_all():
                _end_sequence(0)
                _end_sequence(1)

            def _gap_or_end(side):
                """Increment gap counter for an active side. End sequence if gap exceeds threshold."""
                st = side_states[side]
                if not st["active"]:
                    return
                st["gap"] += 1
                if st["gap"] > args.gap_fill:
                    _end_sequence(side)

            with av.open(path_mp4, "r") as reader:
                s = reader.streams.video[0]
                total = s.frames or (int(float(s.duration * s.time_base) * 30 + 0.5) if s.duration and s.time_base else None)  # assume 30 fps

                for frame_id, frame in enumerate(tqdm(reader.decode(video=0), total=total, desc=f"{clip_id}",
                    unit="frame", position=0, leave=False, dynamic_ncols=True, disable=args.no_tqdm)):
                    if frame_id == 0:
                        im_bgr = frame.to_ndarray(format='bgr24')
                        assert frame_shape == im_bgr.shape[:2]
                    if frame_id not in contact_ids:
                        _gap_or_end(0)
                        _gap_or_end(1)
                        continue

                    row = hos_reader.get(frame_id)
                    hands = row['hands']
                    assert len(hands) > 0

                    selected_hands = {0: None, 1: None}  # side -> (hand_idx, bbox)
                    for idx, h in enumerate(hands):
                        side, side_conf = h['side'], h['side_conf']
                        assert side in (0,1)
                        if side_conf < 0.5: continue # hand side not guaranteed

                        bbox = np.asarray(h['box'], dtype=np.float32)
                        oob = oob_ratio_from_crop(bbox, frame_shape, box_dilate=1.2) # hawor's crop
                        if oob > args.thresh_oob: continue # edge hand

                        if selected_hands[side] is not None:
                            raise AssertionError(f"{clip_id} | frame {frame_id} | multiple valid hands for side {side}")
                        selected_hands[side] = (idx, bbox)
                    if selected_hands[0] is None and selected_hands[1] is None:
                        _gap_or_end(0)
                        _gap_or_end(1)
                        continue

                    im_bgr = frame.to_ndarray(format='bgr24')
                    if need_undist: im_bgr = undistort_apply(im_bgr, map1, map2)

                    for side in (0,1):
                        cand = selected_hands[side]
                        if cand is None:
                            _gap_or_end(side)
                            continue
                        hand_idx, bbox = cand

                        _start_new_sequence_if_needed(side)
                        st = side_states[side]
                        st["gap"] = 0
                        st["frames"].append(frame_id)
                        st["imgs"].append(im_bgr)
                        st["bboxes"].append(bbox)
                        st["hand_indices"].append(hand_idx)
                        st["total_len"] += 1
                        _run_hawor_if_needed(side)
                _end_sequence_all()

            chunk_idx = 0
            rows = []
            for contact_id in sorted(contact_ids):
                row = hos_reader.get(contact_id)
                if row is None: continue
                hands = row.get('hands')
                if not hands: continue
                kpts_for_frame = frame2kpts3d.get(contact_id, {})
                masks_for_frame = frame2hand_mask.get(contact_id, {})

                new_hands = []
                for idx, h in enumerate(hands):
                    h_new = dict(h)
                    acc = kpts_for_frame.get(idx)  # {"sum": ..., "cnt": ...} or None
                    if not acc or acc["cnt"] == 0:
                        kpts3d = None
                    else:
                        kpts3d = (acc["sum"] / acc["cnt"]).astype(np.float32)
                        # sam2 can't take out-of-range prompt points -> require all projected kpts in-image
                        kpts2d = project_3d_kpts_to_2d(kpts3d, img_focal, frame_shape, cx=img_cx, cy=img_cy)
                        assert kpts2d is not None and np.isfinite(kpts2d).all()
                        x = kpts2d[:, 0]
                        y = kpts2d[:, 1]
                        if not ((x >= 0).all() and (x < image_w).all() and (y >= 0).all() and (y < image_h).all()):
                            kpts3d = None

                    hand_mask = None
                    mask_packed = masks_for_frame.get(idx)
                    if mask_packed is not None:
                        hand_mask = np.unpackbits(mask_packed, bitorder="big")[:n_mask_pixels].reshape(frame_shape).astype(bool)

                    # MANO rotations follow kpts3d validity
                    mano_for_frame = frame2mano.get(contact_id, {})
                    acc_m = mano_for_frame.get(idx)
                    if not acc_m or acc_m["cnt"] == 0 or kpts3d is None:
                        wrist_rot = None
                        finger_rot = None
                    else:
                        wrist_rot = (acc_m["orient_sum"] / acc_m["cnt"]).astype(np.float32)
                        finger_rot = (acc_m["pose_sum"] / acc_m["cnt"]).astype(np.float32)

                    h_new['kpts3d'] = kpts3d
                    h_new['wrist_rot'] = wrist_rot
                    h_new['finger_rot'] = finger_rot
                    h_new['hand_mask'] = hand_mask
                    new_hands.append(h_new)
                new_row = dict(row)
                new_row['hands'] = new_hands
                new_row['cam_pose'] = None
                rows.append(new_row)

                if len(rows) == args.chunk_size:
                    write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
                    chunk_idx += 1
                    rows.clear()

            if rows: write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
            mark_done(output_dir, clip_id)


if __name__ == "__main__":
    main()

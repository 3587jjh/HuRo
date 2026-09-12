# Stage 3 — track-consistent hand sides (BoT-SORT): pins one side per hand track.
# Reads stage 2's shards and rewrites them with the refined sides, same schema.
#
#   python pipeline/stage3_annot_contact_refine.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')
import os, os.path as osp, glob, argparse, logging
import numpy as np
import cv2
import av
from tqdm import tqdm
from collections import defaultdict
from types import SimpleNamespace
import sys
from pathlib import Path

# repo root on sys.path so `common` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.io import ParquetReader, write_clip_chunks, mark_done


def create_tracker(frame_rate=30):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)

    class BOTSORT_CUSTOM(BOTSORT):
        def update(self, results, img=None, feats=None):
            super().update(results, img=img, feats=feats)
            return np.asarray(
                [t.result for t in self.tracked_stracks
                 if t.state == TrackState.Tracked],
                dtype=np.float32,
            )

    tracker_args = SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=0.5,
        track_low_thresh=0.5,
        new_track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        gmc_method="sparseOptFlow",
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
        model="auto",
    )
    return BOTSORT_CUSTOM(tracker_args, frame_rate=frame_rate)


# Mimic YOLO's Results class so BoT-SORT can consume our detections.
class MockResults:
    def __init__(self, boxes_xyxy, scores, classes):
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes_xyxy.size == 0:
            self._boxes_xyxy = boxes_xyxy.reshape(0, 4)
        else:
            self._boxes_xyxy = boxes_xyxy.reshape(-1, 4)
        self._scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self._classes = np.asarray(classes, dtype=np.int32).reshape(-1)

    @property
    def xyxy(self):
        return self._boxes_xyxy

    @property
    def xywh(self):
        if self._boxes_xyxy.shape[0] == 0:
            return self._boxes_xyxy
        x1, y1, x2, y2 = self._boxes_xyxy.T
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def conf(self):
        return self._scores

    @property
    def cls(self):
        return self._classes

    def __len__(self):
        return self._boxes_xyxy.shape[0]

    def __getitem__(self, idx):
        return MockResults(
            self._boxes_xyxy[idx],
            self._scores[idx],
            self._classes[idx],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--min_valid_len', type=int, default=16)
    parser.add_argument('--chunk_size', type=int, default=5000, help='frames per parquet chunk')
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    if osp.isdir(input_dir):
        paths_mp4 = sorted(glob.glob(osp.join(input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part)
        if len(paths_mp4) == 0: return
        contact_dir = input_dir + '_contact'
        output_dir = input_dir + '_contact_refined'
    else:
        assert osp.isfile(input_dir)
        assert input_dir.lower().endswith('.mp4')
        paths_mp4 = [input_dir]
        base_dir = osp.dirname(input_dir)
        contact_dir = osp.join(base_dir, 'contact')
        output_dir = osp.join(base_dir, 'contact_refined')

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

    for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1, leave=True, dynamic_ncols=True):
        clip_id = osp.basename(path_mp4)[:-4]

        # read stage 2's contact shards (intrinsics carried as columns)
        try: hos_reader = ParquetReader(contact_dir, clip_id)
        except FileNotFoundError: mark_done(output_dir, clip_id); continue
        contact_ids = hos_reader.get_frame_ids()
        if not contact_ids: mark_done(output_dir, clip_id); continue

        # intrinsics from the carried columns. Rebuild undistort maps (deterministic)
        meta = hos_reader.get(contact_ids[0])
        intr = np.array([meta['fx'], meta['fy'], meta['cx'], meta['cy'], meta['xi']], dtype=np.float64)
        image_h, image_w = int(meta['height']), int(meta['width'])
        frame_shape = (image_h, image_w)
        need_undist = need_undistort(intr, frame_shape)
        if need_undist is None: mark_done(output_dir, clip_id); continue
        if need_undist:
            try: map1, map2, _, _ = build_undistort_maps(frame_shape, intr, auto=True)
            except Exception: mark_done(output_dir, clip_id); continue

        ############################################################################################
        tracker = create_tracker(frame_rate=30)  # assume 30 fps
        track_stats = {} # tid -> { 'frame_ids': [...], 'hand_indices': [...], 'right_probs': [...] }
        tid_offset = 0

        with av.open(path_mp4, "r") as reader:
            s = reader.streams.video[0]
            total = s.frames or (int(float(s.duration * s.time_base) * 30 + 0.5) if s.duration and s.time_base else None)  # assume 30 fps

            for frame_id, frame in enumerate(tqdm(reader.decode(video=0), total=total, desc=f"{clip_id}",
                unit="frame", position=0, leave=False, dynamic_ncols=True, disable=args.no_tqdm)):

                im_bgr = frame.to_ndarray(format='bgr24')
                if frame_id == 0: assert frame_shape == im_bgr.shape[:2]
                if need_undist: im_bgr = undistort_apply(im_bgr, map1, map2)

                row = hos_reader.get(frame_id)
                if (row is None) or (not row.get('hands')) or (not row['hands']):
                    results = MockResults([], [], [])
                    _ = tracker.update(results, img=im_bgr)
                    continue

                hands = row['hands']
                boxes = [h['box'] for h in hands]
                scores = [h['conf'] for h in hands]
                classes = [0] * len(hands)
                results = MockResults(boxes, scores, classes)
                try: tracks = tracker.update(results, img=im_bgr)
                except Exception:
                    tracker = create_tracker(frame_rate=30)
                    tid_offset += 1000000
                    continue
                if tracks is None or len(tracks) == 0: continue

                for trk in tracks:
                    x1, y1, x2, y2, tid, _, _, det_idx = trk
                    tid = int(tid) - 1 + tid_offset
                    det_idx = int(det_idx)
                    assert 0 <= det_idx < len(hands)

                    h = hands[det_idx]
                    side = int(h['side']) # 0: left, 1: right
                    side_conf = float(h['side_conf'])
                    conf = float(h['conf'])
                    p_right = side_conf if side==1 else 1-side_conf

                    stats = track_stats.setdefault(tid, {'frame_ids': [], 'hand_indices': [], 'right_probs': [], 'confs': [], 'xs': []})
                    stats['frame_ids'].append(frame_id)
                    stats['hand_indices'].append(det_idx)
                    stats['right_probs'].append(p_right)
                    stats['confs'].append(conf)
                    stats['xs'].append(0.5 * (h['box'][0] + h['box'][2]))

        ############################################################################################
        track_info = {}  # tid -> dict(...)
        for tid, stats in track_stats.items():
            right_probs = np.asarray(stats['right_probs'])
            confs = np.asarray(stats['confs'])
            xs = np.asarray(stats['xs'])

            w_sum = confs.sum()
            assert w_sum > 0
            mean_p_right = float((right_probs * confs).sum() / w_sum)
            if mean_p_right >= 0.5: side = 1; side_conf = mean_p_right # right
            else: side = 0; side_conf = 1.0 - mean_p_right # left
            mean_x = float((xs * confs).sum() / w_sum)

            track_info[tid] = {
                'mean_p_right': mean_p_right,
                'side': side,
                'side_conf': side_conf,
                'frames': list(stats['frame_ids']),
                'hand_indices': list(stats['hand_indices']),
                'mean_x': mean_x,
            }

        # side conflict handling
        frame_sets = {tid: set(info['frames']) for tid, info in track_info.items()}
        def evidence(info):
            return info['side_conf'] * len(info['frames'])
        sorted_tids = sorted(track_info.keys(), key=lambda t: evidence(track_info[t]), reverse=True)

        winners = set()
        losers = set()
        for tid in sorted_tids:
            info = track_info[tid]
            side = info['side']
            frames = frame_sets[tid]
            conflict = False

            if len(frames) < args.min_valid_len: losers.add(tid); continue
            mean_x = info['mean_x']
            geom_side = 1 if mean_x >= image_w * 0.5 else 0
            if geom_side != side: losers.add(tid); continue

            for w_tid in winners:
                w_info = track_info[w_tid]
                if w_info['side'] != side: continue
                if not (frames & frame_sets[w_tid]): continue
                conflict = True
                break
            if conflict: losers.add(tid)
            else: winners.add(tid)

        frame2updates = defaultdict(dict) # frame_id -> {hand_idx: (new_side, new_side_conf)}
        # Winners first: the strongest track claims a side. A later track with that side
        # in the same frame is flagged.
        order = [t for t in sorted_tids if t not in losers] \
              + [t for t in sorted_tids if t in losers]
        frame_side_taken = defaultdict(set)  # frame_id -> {side already claimed}
        for tid in order:
            info = track_info[tid]
            side = info['side']
            side_conf = info['side_conf']
            if tid in losers:
                # Take the side a co-occurring winner leaves free, falling back to flipping
                # this track's own label when the winners disagree or there are none.
                w_sides = {track_info[w]['side'] for w in winners
                           if frame_sets[w] & frame_sets[tid]}
                new_side = 1 - w_sides.pop() if len(w_sides) == 1 else 1 - side
                if new_side != side: side_conf = 1 - side_conf
                side = new_side
            for f_id, h_idx in zip(info['frames'], info['hand_indices']):
                conf = side_conf
                if conf >= 0.5:
                    # downstream keeps one hand per side per frame. side_conf < 0.5 = not guaranteed
                    if side in frame_side_taken[f_id]: conf = min(conf, 0.49)
                    else: frame_side_taken[f_id].add(side)
                frame2updates[f_id][h_idx] = (side, conf)

        ############################################################################################
        chunk_idx = 0
        rows = []
        for contact_id in sorted(contact_ids):
            row = hos_reader.get(contact_id)
            if row is None: continue
            hands = row['hands']
            if not hands: continue
            update_map = frame2updates.get(contact_id)
            if not update_map: continue

            new_hands = []
            for idx, h in enumerate(hands):
                upd = update_map.get(idx)
                if upd is None: continue
                new_side, new_side_conf = upd
                h_new = dict(h)
                h_new['side'] = int(new_side)
                h_new['side_conf'] = float(new_side_conf)
                new_hands.append(h_new)
            if not new_hands: continue
            new_row = dict(row)
            new_row['hands'] = new_hands
            rows.append(new_row)

            if len(rows) == args.chunk_size:
                write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
                chunk_idx += 1
                rows.clear()
        if rows: write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
        mark_done(output_dir, clip_id)


if __name__ == "__main__":
    main()

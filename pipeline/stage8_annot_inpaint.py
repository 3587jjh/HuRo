# Stage 8 — arm inpainting (ProPainter).
# Reads stage 7's chunked output (the parquet carries the arm_mask) and writes
# "<seg>_inpainted.mp4" next to each source segment. No parquet is written.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage8_annot_inpaint.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import glob, os, os.path as osp, sys, argparse
import numpy as np, cv2, av
import torch
from tqdm import tqdm
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition, setup_propainter_imports, PROPAINTER_WEIGHTS_DIR
from common.io import decode_mask_bytes
from common.video import save_video_task, validate_video_pair

setup_propainter_imports()
from inference_propainter_custom import ProPainterInference

# IO executor for async video saving
io_executor = ThreadPoolExecutor(max_workers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    if osp.isdir(input_dir):
        chunked_dir = osp.join(input_dir + '_chunked', 'original')
        # List the clips in stage 7's chunked output.
        clip_ids = partition(sorted(
            osp.basename(p) for p in glob.glob(osp.join(chunked_dir, 'annot', '*'))
            if osp.isdir(p)), args.part)
    else:
        assert osp.isfile(input_dir) and input_dir.lower().endswith('.mp4')
        chunked_dir = osp.join(osp.dirname(input_dir), 'chunked', 'original')
        clip_ids = [osp.basename(input_dir)[:-4]]
    if len(clip_ids) == 0:
        return

    # Stage 7's per-segment videos + parquets. Inpainted output goes alongside the videos.
    original_video_dir = osp.join(chunked_dir, 'video')
    original_annot_dir = osp.join(chunked_dir, 'annot')
    output_video_dir = original_video_dir

    def is_done(clip_id):
        return osp.exists(osp.join(output_video_dir, f"{clip_id}_inpaint.done"))

    def mark_done(clip_id):
        done = osp.join(output_video_dir, f"{clip_id}_inpaint.done")
        tmp = done + ".tmp"
        with open(tmp, "wb"):
            pass
        os.replace(tmp, done)

    print(f'Got {len(clip_ids)} clips in total.')
    clip_ids = [c for c in clip_ids if not is_done(c)]
    print(f'Processing {len(clip_ids)} clips (rest already done).')
    if len(clip_ids) == 0:
        return
    os.makedirs(output_video_dir, exist_ok=True)

    propainter = ProPainterInference(model_dir=str(PROPAINTER_WEIGHTS_DIR))

    with torch.inference_mode():
        for clip_id in tqdm(clip_ids, desc="clips", unit="clip", position=1, leave=True,
                             dynamic_ncols=True, disable=args.no_tqdm):
            clip_video_dir = osp.join(original_video_dir, clip_id)
            clip_annot_dir = osp.join(original_annot_dir, clip_id)
            if not osp.exists(clip_video_dir):
                mark_done(clip_id); continue

            # Source segment videos (exclude already-inpainted outputs)
            segment_videos = sorted(p for p in glob.glob(osp.join(clip_video_dir, '*.mp4'))
                                    if not p.endswith('_inpainted.mp4'))
            if not segment_videos:
                mark_done(clip_id); continue

            # Skip valid existing inpainted segments. Corrupt ones are deleted by the validator.
            existing_stems = set()
            for inpainted_path in glob.glob(osp.join(clip_video_dir, '*_inpainted.mp4')):
                stem = osp.basename(inpainted_path).replace('_inpainted.mp4', '')
                if validate_video_pair(inpainted_path, osp.join(clip_video_dir, stem + '.mp4')):
                    existing_stems.add(stem)

            io_futures = []
            for seg_video_path in tqdm(segment_videos, desc=f"{clip_id} segments", unit="seg",
                                       position=0, leave=False, disable=args.no_tqdm):
                seg_name = osp.basename(seg_video_path)[:-4]   # e.g. "000010_000040"
                if seg_name in existing_stems:
                    continue
                output_path = osp.join(clip_video_dir, seg_name + "_inpainted.mp4")

                # frame_id -> arm_mask (None for gap frames, which become a zero mask below)
                rows = pq.read_table(osp.join(clip_annot_dir, seg_name + ".parquet"),
                                     columns=["frame_id", "arm_mask"]).to_pylist()
                fid_to_arm_mask = {r["frame_id"]: (decode_mask_bytes(r["arm_mask"])
                                                   if r["arm_mask"] is not None else None)
                                   for r in rows}

                frames, masks = [], []
                orig_h, orig_w = None, None
                with av.open(seg_video_path, "r") as reader:
                    for local_fid, frame in enumerate(reader.decode(video=0)):
                        im_bgr = frame.to_ndarray(format="bgr24")
                        if orig_h is None:
                            orig_h, orig_w = im_bgr.shape[:2]
                        frames.append(im_bgr)
                        masks.append(fid_to_arm_mask[local_fid])

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

                inpainted = propainter.infer(proc_frames, proc_masks, subvideo_length=80)

                output_frames = []
                for img in inpainted:
                    if img.shape[1] != orig_w or img.shape[0] != orig_h:
                        img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                    output_frames.append(img)

                io_futures.append(io_executor.submit(save_video_task, output_path, output_frames,
                                                     30.0, (orig_w, orig_h)))
            for fut in io_futures:
                fut.result()
            mark_done(clip_id)


if __name__ == "__main__":
    main()

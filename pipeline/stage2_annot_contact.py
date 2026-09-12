# Stage 2 — per-frame hand detection (100DoH Faster R-CNN).
# Detects hands (box, conf, side) per undistorted frame -> chunked Parquet, with the stage-1
# raw intrinsics carried as columns. Frames with no detected hand are dropped.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage2_annot_contact.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
#
# The frame preprocessing, the detector class list and the box decoding are adapted from
# 100DoH's demo.py, MIT License, Copyright (c) 2020 Dandan Shan.
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')
import os, glob, json, argparse
import numpy as np
import torch
import cv2
import av
import random
from tqdm import tqdm
import sys
from pathlib import Path

# repo root on sys.path so `common` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import partition, CONTACT_CFG, CONTACT_WEIGHTS
from common.camera import need_undistort, build_undistort_maps, undistort_apply
from common.io import write_clip_chunks, mark_done

# 100DoH model package with its CUDA ops, pip-installed from submodules/100doh/lib
from model.utils.config import cfg, cfg_from_file
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet


# 100DoH's compiled CUDA ops may lack this GPU's arch ("no kernel image") -> swap in torchvision equivalents
def _patch_100doh_cuda_ops():
    from torchvision.ops import nms as _tv_nms, roi_align as _tv_roi_align, roi_pool as _tv_roi_pool
    import model.roi_layers as _rl

    try:
        _dummy = torch.zeros(1, 4, device='cuda')
        _rl.nms(_dummy, torch.zeros(1, device='cuda'), 0.5)
        return False  # works fine, no patch needed
    except RuntimeError as e:
        if 'no kernel image' not in str(e):
            raise
        print("[WARN] 100DoH CUDA ops incompatible with this GPU, patching with torchvision equivalents")
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass  # sync may also raise from the async CUDA error

    import model.rpn.proposal_layer as _pl
    import model.faster_rcnn.faster_rcnn as _frcnn

    _rl.nms = _tv_nms
    _pl.nms = _tv_nms

    # wrap torchvision functional ops to match the nn.Module interface
    class _TVROIAlign(torch.nn.Module):
        def __init__(self, output_size, spatial_scale, sampling_ratio):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale
            self.sampling_ratio = sampling_ratio
        def forward(self, input, rois):
            return _tv_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    class _TVROIPool(torch.nn.Module):
        def __init__(self, output_size, spatial_scale):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale
        def forward(self, input, rois):
            return _tv_roi_pool(input, rois, self.output_size, self.spatial_scale)

    _rl.ROIAlign = _TVROIAlign
    _rl.ROIPool = _TVROIPool
    _frcnn.ROIAlign = _TVROIAlign
    _frcnn.ROIPool = _TVROIPool
    return True

_patch_100doh_cuda_ops()
from model.roi_layers import nms  # import after potential patch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def get_image_blob(im_bgr):
    im = im_bgr.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    H, W = im.shape[:2]
    im_min, im_max = min(H, W), max(H, W)

    processed, scales = [], []
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_min)
        if np.round(im_scale * im_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_max)
        resized = cv2.resize(im, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed.append(resized)
        scales.append(im_scale)

    blob = im_list_to_blob(processed)
    return blob, np.array(scales, dtype=np.float32)


# Detector classes. The checkpoint is a ResNet-101 trained with this 3-class head.
CLASSES = np.asarray(['__background__', 'targetobject', 'hand'])
HAND_CLASS_IDX = 2


def build_model():
    model = resnet(CLASSES, 101, pretrained=False, class_agnostic=False)
    model.create_architecture()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh_hand', type=float, default=0.5) # if modified, tracker args (stage 3) should also be tuned
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--chunk_size', type=int, default=5000, help='frames per parquet chunk')
    parser.add_argument('--no_tqdm', action='store_true')
    args = parser.parse_args()

    cfg_from_file(str(CONTACT_CFG))
    cfg.USE_GPU_NMS = True

    if os.path.isdir(args.input_dir):
        paths_mp4 = sorted(glob.glob(os.path.join(args.input_dir, '*.mp4')))
        paths_mp4 = partition(paths_mp4, args.part)
        if len(paths_mp4) == 0: return
        output_dir = os.path.normpath(args.input_dir) + '_contact'
        os.makedirs(output_dir, exist_ok=True)
    else:
        assert os.path.isfile(args.input_dir)
        assert args.input_dir.lower().endswith('.mp4')
        paths_mp4 = [args.input_dir]
        output_dir = os.path.join(os.path.dirname(args.input_dir), 'contact')
        os.makedirs(output_dir, exist_ok=True)

    print(f'Got {len(paths_mp4)} mp4 files in total.')
    paths_mp4_new = []
    for path_mp4 in paths_mp4:
        if not os.path.exists(os.path.join(output_dir, f"{os.path.basename(path_mp4)[:-4]}.done")):
            paths_mp4_new.append(path_mp4)
    print(f'Skipping {len(paths_mp4) - len(paths_mp4_new)} mp4 files (already processed)')
    paths_mp4 = paths_mp4_new
    if len(paths_mp4) == 0: return

    model = build_model()
    print('Loading checkpoint:', str(CONTACT_WEIGHTS))
    ckpt = torch.load(str(CONTACT_WEIGHTS), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if 'pooling_mode' in ckpt: cfg.POOLING_MODE = ckpt['pooling_mode']
    cfg.CUDA = True
    model.cuda()
    model.eval()

    stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    means= torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    stds, means = stds.cuda(), means.cuda()

    with torch.inference_mode():
        for path_mp4 in tqdm(paths_mp4, desc="clips", unit="clip", position=1, leave=True, dynamic_ncols=True):
            clip_id = os.path.basename(path_mp4)[:-4]

            im_data  = torch.FloatTensor(1)
            im_info  = torch.FloatTensor(1)
            gt_boxes = torch.FloatTensor(1)
            num_boxes= torch.LongTensor(1)
            box_info = torch.FloatTensor(1)
            im_data  = im_data.cuda(); im_info = im_info.cuda()
            gt_boxes = gt_boxes.cuda(); num_boxes = num_boxes.cuda()
            box_info = box_info.cuda()

            chunk_idx = 0
            rows = []

            # read stage 1 intrinsics, and undistort frames if the MEI model warrants it
            path_intr = os.path.join(os.path.dirname(path_mp4)+'_intr', os.path.basename(path_mp4)[:-4]+'.json')
            if not os.path.exists(path_intr):
                path_intr = os.path.join(os.path.dirname(path_mp4), 'intr', os.path.basename(path_mp4)[:-4]+'.json')
            assert os.path.exists(path_intr)
            with open(path_intr, 'r') as f: data = json.load(f)
            if not data:
                mark_done(output_dir, clip_id)
                continue
            intr = np.array([data['fx'], data['fy'], data['cx'], data['cy'], data['xi']], dtype=np.float64)
            image_h, image_w = int(data['H']), int(data['W'])
            frame_shape = (image_h, image_w)
            need_undist = need_undistort(intr, frame_shape)
            if need_undist is None:
                mark_done(output_dir, clip_id)
                continue
            if need_undist:
                try: map1, map2, _, _ = build_undistort_maps(frame_shape, intr, auto=True)
                except Exception:
                    mark_done(output_dir, clip_id)
                    continue
            # raw intrinsics carried forward per row (clip-level constant)
            fx_f, fy_f, cx_f, cy_f, xi_f = (float(v) for v in intr)

            with av.open(path_mp4, "r") as reader:
                total = (s:=reader.streams.video[0]).frames or (int(float(s.duration * s.time_base) * 30 + 0.5) if s.duration and s.time_base else None) # assume 30 fps video
                for frame_id, frame in enumerate(tqdm(reader.decode(video=0), total=total, desc=f"{clip_id}",
                    unit="frame", position=0, leave=False, dynamic_ncols=True, disable=args.no_tqdm)):

                    im_bgr = frame.to_ndarray(format='bgr24')
                    if frame_id == 0: assert frame_shape == im_bgr.shape[:2]
                    if need_undist: im_bgr = undistort_apply(im_bgr, map1, map2)

                    blob, scales = get_image_blob(im_bgr)
                    assert len(scales) == 1
                    im_info_np = np.array([[blob.shape[1], blob.shape[2], scales[0]]], dtype=np.float32)

                    im_data_pt = torch.from_numpy(blob).permute(0,3,1,2).contiguous().pin_memory()
                    im_info_pt = torch.from_numpy(im_info_np).contiguous().pin_memory()
                    im_data.resize_(im_data_pt.size()).copy_(im_data_pt, non_blocking=True)
                    im_info.resize_(im_info_pt.size()).copy_(im_info_pt, non_blocking=True)
                    gt_boxes.resize_(1,1,5).zero_()
                    num_boxes.resize_(1).zero_()
                    box_info.resize_(1,1,5).zero_()

                    rois, cls_prob, bbox_pred, *_ , loss_list = model(im_data, im_info, gt_boxes, num_boxes, box_info)
                    scores = cls_prob.data.squeeze(0)           # (N, C)
                    boxes  = rois.data[:, :, 1:5].squeeze(0)    # (N, 4)

                    # left/right head: P(side==Right) per proposal
                    lr_prob = torch.sigmoid(loss_list[2][0].detach()).squeeze(0).float()  # (N,1)
                    lr = (lr_prob > 0.5).float()                                          # (N,1) 0=L, 1=R
                    side_conf = torch.where(lr.bool(), lr_prob, 1.0 - lr_prob)            # (N,1)

                    if cfg.TEST.BBOX_REG:
                        deltas = bbox_pred.data
                        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                            deltas = deltas.view(-1,4)*stds + means
                            deltas = deltas.view(1,-1,4*scores.size(1))
                        pred_boxes = bbox_transform_inv(boxes.unsqueeze(0), deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1).squeeze(0)
                    else:
                        pred_boxes = boxes.repeat(1, scores.size(1))

                    pred_boxes = pred_boxes / scales[0]

                    # hand-class NMS (the detector's object class is not consumed downstream)
                    cls_scores = scores[:, HAND_CLASS_IDX]
                    keep_idx = torch.nonzero(cls_scores > args.thresh_hand).view(-1)
                    if keep_idx.numel() == 0: continue
                    cls_boxes = pred_boxes[keep_idx][:, HAND_CLASS_IDX*4:(HAND_CLASS_IDX+1)*4]
                    cls_scores_kept = cls_scores[keep_idx]
                    order = torch.sort(cls_scores_kept, descending=True)[1]
                    keep_nms = nms(cls_boxes[order, :], cls_scores_kept[order], cfg.TEST.NMS).view(-1)

                    # per hand: [x1, y1, x2, y2, conf, side(0=L,1=R), side_conf]
                    hand_dets = torch.cat((
                        cls_boxes[order][keep_nms],
                        cls_scores_kept[order][keep_nms].unsqueeze(1),
                        lr[keep_idx][order][keep_nms],
                        side_conf[keep_idx][order][keep_nms],
                    ), 1).detach().cpu().numpy()

                    hands = []
                    for h in hand_dets.astype(np.float32, copy=False):
                        hands.append({
                            "box":   [float(h[0]), float(h[1]), float(h[2]), float(h[3])],
                            "conf":  float(h[4]),
                            "side":  int(h[5]),
                            "side_conf": float(h[6]),
                        })

                    row = {
                        "clip_id": clip_id,
                        "frame_id": int(frame_id),
                        "height":  int(image_h),
                        "width":   int(image_w),
                        "intr_model": data['model'],
                        "hands":   hands,
                        "fx": fx_f, "fy": fy_f, "cx": cx_f, "cy": cy_f, "xi": xi_f,
                    }
                    rows.append(row)
                    if len(rows) == args.chunk_size:
                        write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
                        chunk_idx += 1
                        rows.clear()
                if rows: write_clip_chunks(clip_id, rows, output_dir, chunk_idx)
            mark_done(output_dir, clip_id)


if __name__ == '__main__':
    main()

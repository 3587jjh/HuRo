"""Stage-6 model wrappers: detectron2 ViTDet-H person detector + SAM2 video predictor.
detectron2 / sam2 / hydra are imported lazily inside the constructors."""
# The predictor wrapper is adapted from detectron2's DefaultPredictor, Apache-2.0,
# Copyright (c) Facebook, Inc. and its affiliates. The mask propagation loop is adapted from
# SAM 2's video predictor example, Apache-2.0, Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch

from common.paths import (DETECTRON_CFG, DETECTRON_WEIGHTS,
                          SAM2_CONFIG_DIR, SAM2_MODEL_CFG, SAM2_WEIGHTS)


class DetectorDetectron2:
    """Cascade Mask R-CNN ViTDet-H person detector (lazy-config predictor)."""

    def __init__(self, score_thresh=0.25):
        import detectron2.data.transforms as T
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate
        from omegaconf import OmegaConf

        cfg = LazyConfig.load(str(DETECTRON_CFG))
        cfg.train.init_checkpoint = str(DETECTRON_WEIGHTS)
        for predictor in cfg.model.roi_heads.box_predictors:
            predictor.test_score_thresh = score_thresh

        self._T = T
        self.model = instantiate(cfg.model)
        DetectionCheckpointer(self.model).load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))
        mapper = instantiate(cfg.dataloader.test.mapper)
        self.aug = mapper.augmentations
        self.input_format = mapper.image_format
        assert self.input_format in ("RGB", "BGR"), self.input_format
        self.model.eval().cuda()

    def __call__(self, image_bgr):
        T = self._T
        with torch.no_grad():
            img = image_bgr[:, :, ::-1] if self.input_format == "RGB" else image_bgr
            h, w = img.shape[:2]
            x = self.aug(T.AugInput(img)).apply_image(img)
            x = torch.as_tensor(x.astype("float32").transpose(2, 0, 1))
            return self.model([{"image": x, "height": h, "width": w}])[0]

    def get_bboxes(self, image_bgr):
        """Person bounding boxes (N,4) + scores (N,): COCO class 0 with score > 0.5."""
        inst = self(image_bgr)["instances"]
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)
        return inst.pred_boxes.tensor[keep].cpu().numpy(), inst.scores[keep].cpu().numpy()


class DetectorSam2:
    """SAM2 video predictor: bidirectional multi-object mask propagation over in-memory
    BGR frames (requires the sam2 in-memory-frames patch)."""

    def __init__(self):
        from sam2.build_sam import build_sam2_video_predictor
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        self.device = "cuda"
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(SAM2_CONFIG_DIR), version_base=None):
            self.video_predictor = build_sam2_video_predictor(
                SAM2_MODEL_CFG, str(SAM2_WEIGHTS), device=self.device)

    def segment_video_from_arrays_bidir_multi(self, frames, seeds, output_bboxes=None):
        """frames: ordered list of (H,W,3) BGR uint8 images.
        seeds: {obj_id: {"frame_idx", "bbox", "points"}} or {obj_id: {"frame_idx", "mask"}}.
        Returns (fwd, rev), each {obj_id: {local_frame_idx: (H,W) bool}}."""
        frame_masks_fwd = {obj_id: {} for obj_id in seeds}
        frame_masks_rev = {obj_id: {} for obj_id in seeds}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.video_predictor.init_state(video_path=frames)
            self.video_predictor.reset_state(state)

            for obj_id, seed in seeds.items():
                idx = seed["frame_idx"]
                mask = seed.get("mask")
                if mask is not None:
                    self.video_predictor.add_new_mask(
                        state, frame_idx=int(idx), obj_id=int(obj_id), mask=mask)
                else:
                    bbox, points = seed["bbox"], seed["points"]
                    if bbox is None or np.all(bbox) == 0:
                        self.video_predictor.add_new_points_or_box(
                            state, frame_idx=int(idx), obj_id=int(obj_id),
                            points=np.array(points), labels=np.ones(len(points)))
                    else:
                        self.video_predictor.add_new_points_or_box(
                            state, frame_idx=int(idx), obj_id=int(obj_id),
                            box=np.array(bbox), points=np.array(points), labels=np.ones(len(points)))

            video_segments_fwd = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in \
                    self.video_predictor.propagate_in_video(state, reverse=False):
                video_segments_fwd[out_frame_idx] = {
                    oid: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, oid in enumerate(out_obj_ids)}
            video_segments_rev = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in \
                    self.video_predictor.propagate_in_video(state, reverse=True):
                video_segments_rev[out_frame_idx] = {
                    oid: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, oid in enumerate(out_obj_ids)}
        assert output_bboxes is None
        torch.cuda.empty_cache()

        for frame_idx, obj_dict in video_segments_fwd.items():
            for oid, m in obj_dict.items():
                frame_masks_fwd.setdefault(int(oid), {})[frame_idx] = m[0].astype(bool)
        for frame_idx, obj_dict in video_segments_rev.items():
            for oid, m in obj_dict.items():
                frame_masks_rev.setdefault(int(oid), {})[frame_idx] = m[0].astype(bool)
        return frame_masks_fwd, frame_masks_rev

# Robotization Pipeline

The pipeline converts egocentric human videos into robotized episodes for VLA pretraining. It
retargets the human hand motion into the target robot's joint trajectory, and it removes the
human arms from the frames and renders that robot in their place. Both conversions require
annotations that raw video does not carry, so the pipeline estimates them from the frames first:
the camera parameters, the 3D hand motion, and a language instruction for each manipulation
segment.

## The flow

**Stages 1 to 8 (`stage*_annot_*.py`) operate on the human video alone.** Stage 1 estimates the
camera intrinsics, which stages 2 onward use to undistort the frames to a pinhole view. Stages 2
and 3 detect the hands and assign each one a side, stage 4 recovers the 3D hand pose, and stage 5
the metric, gravity-aligned camera trajectory. Stage 6 segments the human arms and stage 8
inpaints them out, leaving the cleaned scene the robot is rendered onto. Between the two, stage 7
splits the clip into manipulation segments and generates one language instruction per segment
with a VLM. Every stage after stage 7 operates on segments rather than clips.

**Stages 9 and 10 (`stage*_robot_*.py`) introduce the robot.** Stage 9 fits the robot's arms and
hands to the human hand motion and produces the joint trajectory. Stage 10 renders that robot
onto stage 8's cleaned frames.

**Stage 11 writes the dataset.** A segment becomes one LeRobot episode, with the overlay video
as the observation, the retargeted joints as the state and action, and stage 7's instruction as
the language.

The stages write Parquet tables of their own, all sharing one schema. The data format reference
in [`examples/README.md`](../examples/README.md) documents that schema and the LeRobot dataset.

**Allex is the only robot configured here.** Only stages 9 to 11 depend on the target robot, so
an annotated clip directory can be retargeted to another without repeating stages 1 to 8.
[`configs/README.md`](../configs/README.md) documents what adding a robot involves.

## The stages

| # | Script | Backend | Output |
|---|--------|---------|--------|
| 1 | `stage1_annot_intrinsics.py` | DROIDCalib, AnyCalib fallback | per-clip camera intrinsics JSON (`fx, fy, cx, cy, xi, H, W, model`) |
| 2 | `stage2_annot_contact.py` | 100DoH Faster R-CNN, hand class only | per-frame hand boxes and sides, and the first Parquet table |
| 3 | `stage3_annot_contact_refine.py` | BoT-SORT tracking | the same hands, with the sides made consistent along each track |
| 4 | `stage4_annot_hand.py` | HAWOR | per-frame 3D hand keypoints, MANO rotations and hand masks |
| 5 | `stage5_annot_extrinsics.py` | DROID-SLAM, MoGe-2, GeoCalib | per-frame camera pose, metric-scaled and gravity-aligned (4x4 `T_cam2world`, OpenCV convention) |
| 6 | `stage6_annot_segment.py` | Detectron2 ViTDet-H, SAM 2 | per-frame arm mask, and the number of people detected, which stage 7 uses to reject multi-person segments |
| 7 | `stage7_annot_narr.py` | Qwen3.5 VLM | per-segment undistorted videos and Parquet tables, plus the `language` instruction used by stage 11 |
| 8 | `stage8_annot_inpaint.py` | ProPainter | per-segment arm-removed videos, no Parquet table |
| 9 | `stage9_robot_retarget.py` | PyRoKi IK (JAX) | per-segment robot joint angles and wrist poses, the camera pose in the robot's base frame, and IK diagnostics |
| 10 | `stage10_robot_overlay.py` | Isaac Sim | per-segment overlay videos, the robot rendered onto stage 8's cleaned frames |
| 11 | `stage11_lerobot_convert.py` | none | a LeRobot V2.0 dataset, one episode per segment |

## Subsystems

| Directory | Holds |
|---|---|
| `pipeline/segmentation/` | stage 6 model wrappers: ViTDet-H person detection and SAM 2 |
| `pipeline/captioning/` | stage 7 VLM subsystem: captioning, prompts, caption validation, rejection tracker |
| `pipeline/retargeting/` | stage 9 IK subsystem: the two-stage solver and its forward-kinematics helpers |
| `pipeline/overlay/` | stage 10 Isaac Sim renderer, including the URDF import |

# Robotization Pipeline

The pipeline converts egocentric human videos into robotized episodes for VLA pretraining. It
retargets the human hand motion into the target robot's joint trajectory, and it removes the
human arms from the frames and renders that robot in their place. Both conversions require
annotations that raw video does not carry, so the pipeline estimates them from the frames first.

## The flow

**Stages 1 to 7 (`pipeline/stage*_annot_*.py`) operate on the human video alone.** Stage 1
estimates the camera intrinsics, which stages 2 to 6 use to undistort the frames to a pinhole
view. Stages 2 and 3 detect the hands and assign each one a side, stage 4 recovers the 3D hand
pose, and stage 5 the metric, gravity-aligned camera trajectory. Stage 6 cuts manipulation
segments out of the clip and generates one language instruction per segment with a VLM. It
drops a segment without a message when it detects three or more people in it or rejects its
instruction. Stage 7 masks the human arms and inpaints them out, leaving the cleaned scene the
robot is rendered onto. Every stage from stage 7 on operates on segments rather than clips.

**Stages 8 and 9 (`pipeline/stage*_robot_*.py`) introduce the robot.** Stage 8 fits the robot's
arms and hands to the human hand motion and produces the joint trajectory. Stage 9 renders that
robot onto stage 7's cleaned frames.

**Stage 10 writes the dataset.** A segment becomes one LeRobot episode, with the overlay
video as the observation, the retargeted joints as the state and action, and stage 6's
instruction as the language.

Stages 2 to 9 write Parquet tables of their own, all sharing one schema. The data format reference
in [`examples/README.md`](../examples/README.md) documents that schema and the LeRobot dataset.

**Allex is the only robot configured here.** Only stages 8 to 10 depend on the target robot, so
annotated clips can be retargeted to another robot without repeating stages 1 to 7.
[`configs/README.md`](../configs/README.md) documents how to add a robot.

## The stages

| # | Script | Backend | Output |
|---|--------|---------|--------|
| 1 | `stage1_annot_intrinsics.py` | DroidCalib, AnyCalib fallback | per-clip JSON of `fx, fy, cx, cy, xi, H, W, model` |
| 2 | `stage2_annot_contact.py` | 100DoH Faster R-CNN, hand class only | per-frame hand boxes and sides, and the first Parquet table |
| 3 | `stage3_annot_contact_refine.py` | BoT-SORT tracking | the same hands, with the sides made consistent along each track |
| 4 | `stage4_annot_hand.py` | HAWOR | per-frame 3D hand keypoints, MANO rotations and hand masks |
| 5 | `stage5_annot_extrinsics.py` | DROID-SLAM, MoGe-2, GeoCalib | per-frame camera pose, metric-scaled and gravity-aligned (4x4 cam-to-world, OpenCV convention) |
| 6 | `stage6_annot_narr.py` | ViTDet-H, Qwen3.5-9B | per-segment undistorted videos, and Parquet tables that carry the `language` instruction |
| 7 | `stage7_annot_inpaint.py` | ViTDet-H, SAM 2, ProPainter | per-segment arm masks and inpainted videos |
| 8 | `stage8_robot_retarget.py` | PyRoKi IK (JAX) | per-segment robot joint angles and wrist poses, the camera pose relative to the robot base, and IK diagnostics |
| 9 | `stage9_robot_overlay.py` | Isaac Sim | per-segment overlay videos of the robot on the cleaned frames |
| 10 | `stage10_lerobot_convert.py` | none | a LeRobot V2.0 dataset, one episode per segment |

## Subsystems

| Directory | Holds |
|---|---|
| `pipeline/segmentation/` | model wrappers for stages 6 and 7: ViTDet-H person detection and SAM 2 |
| `pipeline/captioning/` | stage 6 VLM subsystem: captioning, prompts, caption validation, rejection tracker |
| `pipeline/retargeting/` | stage 8 IK subsystem: the two-step solver and its forward-kinematics helpers |
| `pipeline/overlay/` | stage 9 Isaac Sim renderer, including the URDF import |

# Data format

A pipeline run reads the videos in a directory `<clips>` and writes every output beside it.
Its stages, numbered as in [`pipeline/README.md`](../pipeline/README.md), record their
annotations in **Parquet tables**, and the last stage converts those tables into a **LeRobot
V2.0 dataset** for policy training.

## The Parquet tables

### Where they are written

Stage 1 writes one JSON of intrinsics per clip under `<clips>_intr`. Stages 2 to 5 write
each clip as shards, `{clip_id}_NN.parquet`, in one directory per stage: `<clips>_contact`,
`_contact_refined`, `_hand` and `_extr`. Stage 6 cuts manipulation segments out of each
clip. From then on each segment has its own files under `<clips>_chunked`, with one
`<robot>` tree per target robot:

```
<clips>_chunked/
  original/video/<clip_id>/<start>_<end>.mp4            stage 6   the segment's frames
  original/video/<clip_id>/<start>_<end>_inpainted.mp4  stage 7   the same frames, arms removed
  original/annot/<clip_id>/<start>_<end>_narr.parquet   stage 6   the human annotations
  original/annot/<clip_id>/<start>_<end>.parquet        stage 7   the same table + the arm masks
  <robot>/annot/<clip_id>/<start>_<end>.parquet         stage 8   the robot trajectory
  <robot>/overlay/video/<clip_id>/<start>_<end>.mp4     stage 9   the rendered robot
  <robot>/overlay/annot/<clip_id>/<start>_<end>.parquet stage 9   stage 8's table + a render flag
```

`<start>` and `<end>` are inclusive frame indices in the clip. Stages 2 to 9 write a `.done`
marker per finished clip. A rerun skips the marked clips. A dropped clip, such as one without a
detected hand, gets a marker but no output.

### The columns

Every table uses the one schema in [`common/io.py`](../common/io.py). Each row is one video
frame. The pose columns use the axes and units that [Coordinate frames](#coordinate-frames)
defines. Annotations from stage 2 on are made on the undistorted frames.

| column | type | stage |
|---|---|---|
| `clip_id` | string | 2 |
| `frame_id` | int32, the clip's frame index, restarted at 0 per segment by stage 6 | 2 |
| `height`, `width` | int32, the input video's resolution | 2 |
| `intr_model` | string, `droidcalib` or `anycalib` (fallback), the method stage 1 used | 2 |
| `fx`, `fy`, `cx`, `cy`, `xi` | float64, stage 1's raw intrinsics | 2 |
| `pinhole_*` | float64, `fx`, `fy`, `cx`, `cy` of the undistorted frames | 2 |
| `hands` | list of the per-hand struct below | 2 |
| `cam_pose` | 4x4 cam-to-world, OpenCV, metric-scaled and gravity-aligned | 5 |
| `n_person_det` | int32, the number of people detected, only on sampled frames | 6 |
| `narr` | struct of `think`, `left`, `right`, `bimanual` | 6 |
| `language` | string, `narr` merged to one instruction | 6 |
| `arm_mask` | bit-packed (H,W) bool | 7 |
| `cam_pose_base` | 4x4 cam-to-robot-base, OpenCV | 8 |
| `state_qpos` | float32, the robot's joint angles, ordered by `state_qpos_joint_names` below | 8 |
| `state_eef_left`, `state_eef_right` | float32, wrist position (3) + first two rows of its rotation matrix (6) + hand joint angles | 8 |

The `hands` struct, one entry per detected hand:

| field | type | stage |
|---|---|---|
| `box` | float32[4], `[x1,y1,x2,y2]` | 2 |
| `conf` | float32, detector confidence | 2 |
| `side` | int8, 0 left and 1 right | 2, rewritten by 3 |
| `side_conf` | float32 | 2, rewritten by 3 |
| `kpts3d` | float32[21,3], MediaPipe order, camera coordinate frame | 4 |
| `wrist_rot` | float32[3], axis-angle | 4 |
| `finger_rot` | float32[15,3], axis-angle | 4 |
| `hand_mask` | bit-packed (H,W) bool | 4, set to null by 6 |

Stages 8 and 9 also write the diagnostic columns below. None of them enters the LeRobot dataset:

| column | type | stage |
|---|---|---|
| `state_mask_left`, `state_mask_right` | bool, false where the IK has no target for that side's hand. The state there follows from the frames around it, or is the neutral pose if that side has no target in the whole segment | 8 |
| `retarget_cost` | float32, the segment's solver cost divided by its frame count, written on every row | 8 |
| `err_tip_mm_*`, `err_palm_mm_*`, `err_local_dir_*` | per-side IK residuals | 8 |
| `err_ddq` | float32, per-joint second difference | 8 |
| `err_cam_pos_mm`, `err_cam_rot_deg` | how far the solved camera sits from stage 5's | 8 |
| `overlay_valid` | bool, false where Isaac Sim failed to render the frame | 9 |

### The robot metadata

From stage 8 on, each table carries schema metadata, `pq.read_table(...).schema.metadata`, which
names the robot and its joints, so the table can be read without the robot's config:

| key | holds |
|---|---|
| `robot_name` | the config the table was written from |
| `state_qpos_joint_names` | the joint names, in `state_qpos` order |
| `state_eef_layout` | `per_side:wrist_pos3_rot6d6_hand_joints` |
| `state_eef_left_hand_joint_names`, `state_eef_right_hand_joint_names` | the joints after each wrist pose |
| `state_eef_frame` | `robot_base_frame` |
| `cam_pose_convention` | `opencv_cam_to_base` |

### Reading them

`examples/read_parquet.py` prints one segment's schema metadata, the columns each table filled,
and one decoded row:

```bash
python examples/read_parquet.py examples/clips_chunked    # after a run over examples/clips/
```

Decode each row with `decode_row` from `common/io.py`, as the script does. On disk, masks are
bit-packed, and an absent per-hand array is stored as a list of NaNs.

## Coordinate frames

Positions are in metres and joint angles in radians, in the tables and in the dataset alike.

The robot's wrist poses and camera pose, `state_eef_*` and `cam_pose_base` in the tables, are
given in the robot base frame, whose axes are x forward, y left and z up. The camera's own axes
are **OpenCV: +x right, +y down, +z forward**.

Each wrist rotation is the orientation of the MANO wrist frame, which follows the hand's anatomy:

```
along   = wrist            -> middle-finger MCP      (down the hand)
across  = pinky MCP        -> index MCP              (across the knuckles, toward the thumb side)
normal  = along x across                             (palmar on a left hand, dorsal on a right)
```

| axis | left hand | right hand | in words |
|---|---|---|---|
| **+x** | `+along` | `−along` | **left:** wrist→fingers · **right:** fingers→wrist |
| **+y** | `−normal` | `+normal` | out of the **back of the hand** (dorsal), both hands |
| **+z** | `+across` | `+across` | toward the **thumb side**, both hands |

The wrist position, `state_eef_*[0:3]`, is the origin of the robot's wrist link. For Allex it is
the wrist pitch joint.
[`configs/README.md`](../configs/README.md#the-mano-wrist-frames--eef_link_names) draws these
axes on a person's hands and on Allex's hands, and `configs/check_frames.py` checks a robot's
wrist link against them.

## The LeRobot dataset

Stage 10 turns each segment of stage 9's tables into one episode of a LeRobot dataset. It
writes the dataset under `<clips>_lerobot/<robot>/<HxW>/`. A LeRobot dataset holds videos of
a single resolution, so clips of different aspect ratios end up in separate datasets. For
Allex an episode holds:

| feature | dim | from the tables |
|---|---|---|
| `observation.state` | 48 | `state_qpos`, reordered as `meta/info.json` names it |
| `action` | 48 | `observation.state` of the next frame, with the last frame repeated |
| `observation.state_eef_{left,right}` | 9 | the wrist pose, the first nine numbers of `state_eef_*` |
| `action_eef_{left,right}` | 9 | the same wrist pose, of the next frame |
| `observation.state_cam_frame` | 9 | `cam_pose_base` as position (3) + first two rows of the rotation matrix (6) |
| `observation.images.zed_left` | — | the overlay video |
| `task_index`, `annotation.human.coarse_action` | 1 | index into `meta/tasks.jsonl`, whose task string is the segment's `language` |

`meta/modality.json` splits the joint vector into the arms, the hands, the neck and the waist,
and all four carry motion.

> [!WARNING]
> **Stage 10 deliberately writes no `meta/stats.json`**, the standard LeRobot file of
> normalisation statistics. Their values depend on the training setup, such as the datasets and
> resolutions merged into one corpus and the trainer's normalisation. Fill it in before training,
> following the format of a V2.0 dataset such as
> [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht/blob/v2.0/meta/stats.json).

`examples/load_lerobot.py` reads a dataset as a PyTorch `Dataset` and prints its layout:

```bash
python examples/load_lerobot.py examples/clips_lerobot/allex/192x342    # after a run over examples/clips/
```

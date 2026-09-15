<div align="center">

<h1 align="center">HuRo:<br>Robotizing Human Videos for Scalable VLA Pretraining</h1>

<p align="center">
    <a href='https://arxiv.org/abs/2609.10706'><img src='https://img.shields.io/badge/arXiv-2609.10706-b31b1b?logo=arxiv&logoColor=white' alt='arXiv'></a>
    <a href='https://3587jjh.github.io/HuRo/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data_coming_soon-lightgrey' alt='Dataset'>
    <a href='#-license'><img src='https://img.shields.io/badge/License-Apache--2.0-orange' alt='License'></a>
</p>

<p align="center"><img src="docs/teaser.png" width="100%" alt="Overview of HuRo pretraining for real-world manipulation"></p>

<div align="justify">

**HuRo** converts egocentric human video into robot-aligned observations and action trajectories
for vision-language-action (VLA) pretraining. The pipeline retargets the human hand motion into
robot joint trajectories, and it removes the human arms from the frames and renders the robot in
their place. Intermediate signals a source does not provide are estimated, so videos at different
annotation levels are converted into a common format. Using this pipeline we construct the **HuRo
dataset**, comprising about **630K robotized episodes** and **142M processed frames** from five
human-video sources. Across four real-world manipulation tasks, increasing the robotized
pretraining scale improves overall completion after finetuning from **51.5% to 80.3%** and
out-of-distribution completion under spatial and visual shifts from **34.9% to 72.2%**.

</div>

<br>

</div>

---

## 📢 News & Updates

- **[2026-09-11]** 🚀 Code released.
- **[2026-09-10]** 🚀 Paper on arXiv.
- **[2026-09-09]** 🚀 Project page live.

---

## 📑 Contents

- [📢 News & Updates](#-news--updates)
- [🏭 Robotization Pipeline](#-robotization-pipeline)
- [🔧 Setup](#-setup)
- [⚡ Run](#-run)
- [🎬 Input video](#-input-video)
- [🤖 Target robot](#-target-robot)
- [📊 Data format](#-data-format)
- [📥 HuRo Dataset](#-huro-dataset)
- [📄 License](#-license)
- [📝 Citation](#-citation)

## 🏭 Robotization Pipeline

> [!NOTE]
> **This repository releases HuRo's robotization pipeline**, from raw video
> to a LeRobot V2.0 dataset.

![raw egocentric video, the arms removed, and the robot rendered in their place](docs/pipeline_example.jpg)

HuRo's robotization pipeline converts **raw, untrimmed egocentric video** of everyday activity
into robotized episodes for VLA pretraining. It first estimates the camera and hand annotations,
identifies the manipulation segments within the untrimmed clip, and assigns a language instruction
to each. Then **action conversion** retargets the human hand motion into the target robot's joint
trajectory, and **visual conversion** removes the visible human embodiment and renders that robot
onto the cleaned scene. A segment becomes one episode of the target embodiment, comprising the
robotized video, the retargeted states and actions, and a language instruction, formatted as a
LeRobot V2.0 dataset. See [`pipeline/README.md`](pipeline/README.md) for the pipeline in detail,
and the data format reference in [`examples/README.md`](examples/README.md) for the outputs a
run writes and how to read them.

## 🔧 Setup

The pipeline requires Linux, an NVIDIA GPU with at least 24 GB of VRAM and a driver supporting
CUDA 12.8, and git. The robot overlay also requires a GPU with RT cores.
[`setup/README.md`](setup/README.md) covers the installation and how to download the
off-the-shelf models the pipeline uses.

## ⚡ Run

```bash
./run_pipeline.sh
```

The script runs all ten stages in order over a directory of clips. Edit the settings block at
the top of it to configure a run. `INPUT_DIR` defaults to `examples/clips/`, and those two clips
already meet the requirements in [Input video](#-input-video), so the script runs as it stands.
All stages are resumable.

The outputs are written beside the input directory, as `<clips>_intr`, `_contact`,
`_contact_refined`, `_hand`, `_extr`, `_chunked` and `_lerobot`.
[`examples/README.md`](examples/README.md) documents what each of them holds.

Each stage can also be run on its own:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python pipeline/stage<N>_<phase>_<name>.py \
    --input_dir <clip-dir> --part <a>/<b> --no_tqdm

CUDA_VISIBLE_DEVICES=0 python pipeline/stage1_annot_intrinsics.py \
    --input_dir examples/clips --part 2/4 --no_tqdm
```

Every stage accepts those three flags. The retargeting, overlay and LeRobot conversion stages
additionally accept `--robot_name`, which defaults to `allex` (`configs/allex.yaml`). See
[Target robot](#-target-robot) for adding a custom robot.

> [!NOTE]
> **The released code does not read annotations that a source dataset provides.** It estimates
> the camera geometry, hand poses and language from the video itself. To start at stage N with
> provided annotations, write what stage N-1 would write in the format that
> [`examples/README.md`](examples/README.md) documents, with the `.done` markers included.
> Without the markers, a stage can treat a clip as dropped.

## 🎬 Input video

`INPUT_DIR` (in `run_pipeline.sh`) points at a directory of `.mp4` files. Untrimmed video is
fine, because the pipeline finds its own segments. A file of roughly **30 seconds to 30 minutes**
is the length to aim for. A very short clip can hold too little camera motion for a good
calibration. A longer clip takes more CPU memory.

Three properties of the video matter:

- **Egocentric.** The whole pipeline is designed around egocentric video.
- **30 fps.** The pipeline's internal parameters assume that rate.
- **A 256-pixel shorter side.** A larger input also runs, but decoding the video and resizing
  its frames for the off-the-shelf models take longer, without a noticeable gain in quality.

Fisheye and rectilinear video are both accepted as they are. A clip whose camera calibration fails
is dropped without an error, and an empty `<clips>_intr/<clip>.json` is the record of that.

Convert each video to 30 fps and a 256-pixel shorter side with:

```bash
ffmpeg -i raw.mp4 -vf "fps=30,scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'" \
    -c:v libx264 -crf 18 -an clips/raw_30fps_256.mp4
```

## 🤖 Target robot

**Allex is the only robot configured here.** Adding another robot is mostly configuration. The
retargeting and the overlay read one YAML file and the URDF it names, and they need no code
change. The annotations before the retargeting do not depend on the robot either, so an existing
`<clips>_chunked` tree can be converted for another robot. See
[`configs/README.md`](configs/README.md) for the full procedure.

## 📊 Data format

The **Parquet tables** carry the annotations, and the **LeRobot V2.0 dataset** holds the
robotized episodes. See [`examples/README.md`](examples/README.md) for both formats in full.

Two reader scripts come with them, one per format, and each is the reference implementation for
its own format. Run them on the outputs of the example clips:

```bash
python examples/read_parquet.py examples/clips_chunked                  # the Parquet tables
python examples/load_lerobot.py examples/clips_lerobot/allex/192x342    # the LeRobot episodes
```

## 📥 HuRo Dataset

**Coming soon.**

## 📄 License

HuRo's own code is Apache-2.0 ([`LICENSE`](LICENSE)).
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists the licence of every
off-the-shelf component, what this repository redistributes of each, and the
attribution for the EPIC-KITCHENS material.

> [!WARNING]
> **This pipeline cannot be run commercially.** Its dependencies impose the restriction.

## 📝 Citation

```bibtex
@misc{jeong2026huro,
      title={HuRo: Robotizing Human Videos for Scalable VLA Pretraining},
      author={Jinho Jeong and Se June Joo and Jaehyun Kang and Dongyun Kim and Yena Kim and Hanjung Kim and Seon Joo Kim},
      year={2026},
      eprint={2609.10706},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.10706},
}
```

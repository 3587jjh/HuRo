# The EgoDex subset

The HuRo dataset includes 274,719 episodes robotized from **EgoDex** — 43.6 % of the corpus, its
largest single source. None of them are published.

Two separate reasons, and it is worth keeping them apart: we may not publish it, and we could not
hand you the exact thing even if we could.

## 1. Licence

EgoDex is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). The NoDerivatives
term forbids *sharing* adapted material, and robotized frames are adapted material by any reading.
Whether the derived annotation tables are adapted material is genuinely unsettled, and we do not
rely on the favourable reading. So nothing is published: not the video, not the annotations.

## 2. The subset is not reproducible from this repository

**The HuRo EgoDex subset was not produced by the pipeline in this repository.** Two departures:

- **Camera and hand annotations came from EgoDex itself**, not from stages 1–5. EgoDex ships ARKit
  ground truth from Apple Vision Pro — intrinsics, per-frame camera pose, 68 joint transforms and
  language — which is better than anything the pipeline can estimate from the video. The released
  code deliberately ignores what a source provides, which is right for Ego4D and EPIC-KITCHENS and
  wrong for EgoDex.
- **Arm segmentation and inpainting were done with separate, earlier code**, not by the stage 7 in
  this repository. That code is not part of the released pipeline, so the exact `arm_mask` and
  arm-removed frames the paper trained on cannot be regenerated here.

The consequence is blunt: **running this repository over EgoDex produces a comparable dataset, not
the paper's dataset.** Numbers from it are not directly comparable with the paper's.

### How this differs from the other subsets

The Ego4D, EPIC-KITCHENS, Egocentric-10K and EgoVerse subsets *were* produced by the released
stack, and their annotation tables are published. That is not the same as bit-reproducibility, and
the distinction matters:

| | EgoDex | the published subsets |
|---|---|---|
| What produced the tables | **different code**, outside this repository | this repository, at an earlier commit |
| Re-running the pipeline gives you | a comparable dataset | the same pipeline's output, save for stage 8, which is not bit-reproducible on GPU and settles on an equally valid solution of its own |

Each published dataset card states its own reproducibility position. Here the gap is a matter of
different code entirely; there it is a matter of solver version.

## 3. What you can do instead

Treat your own copy of [EgoDex](https://github.com/apple/ml-egodex) as ordinary egocentric video
and run the pipeline as documented in the [main README](../README.md). Nothing extra is required:

```bash
./run_pipeline.sh    # INPUT_DIR=clips
```

Re-encode first — EgoDex ships 1080p 30 fps and the pipeline wants a 256-pixel shorter side. Frame
count is unchanged, so HDF5 row `n` still corresponds to video frame `n`:

```bash
ffmpeg -i <task>/<i>.mp4 -vf "fps=30,scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'" \
    -c:v libx264 -crf 18 -an clips/<task>_<i>.mp4
```

Camera and hand annotations then come from DroidCalib / HaWoR / DROID-SLAM rather than from ARKit.
That is the main difference from the paper's subset, and it costs quality as well as compute:
stages 1–5 are the expensive half of the pipeline, and ARKit's ground truth is better than what
they estimate.

> [!NOTE]
> **A converter that reads EgoDex's HDF5 annotations directly would skip stages 1–5 and use the
> ARKit ground truth instead.** No such converter is part of this repository, and this document
> does not describe one. If you write your own, two things are easy to get wrong: ARKit's origin
> frame is **per-session**, a stationary frame set when the recording began — consistent within an
> episode and meaningless across episodes, so rebase to the episode's first camera pose. And
> Apple's README warns that the 2D reprojections of the 3D annotations do not exactly match the
> hands in the video, because the RGB stream is synthesized from several Vision Pro cameras; that
> shows up as a small, consistent offset between the rendered robot and the scene, and it is
> inherent to EgoDex rather than a retargeting bug.

## Licence of this document

Nothing in this document contains EgoDex material. Your use of the EgoDex videos and annotations is
governed by [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) and Apple's
terms. Datasets you generate by running this pipeline over them are yours to use within those
terms, subject to the same limits the HuRo pipeline imposes — see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

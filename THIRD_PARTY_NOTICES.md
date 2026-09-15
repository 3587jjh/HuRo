# Third-party notices

HuRo's pipeline runs on the off-the-shelf components below, and the last column says what this
repository redistributes of each. The rest comes from upstream. Each patch in
`submodules_patches/` is a diff against one of them, with that project's licence in its header.

## Components

| Component | Code | Weights | Distributed here |
|---|---|---|---|
| [DroidCalib](https://github.com/boschresearch/DroidCalib) | AGPL-3.0 | AGPL-3.0 | Patch |
| [lietorch](https://github.com/princeton-vl/lietorch) | BSD-3-Clause | — | Patch |
| [AnyCalib](https://github.com/javrtg/AnyCalib) | Apache-2.0 | Apache-2.0 | Patch |
| [100DoH detector](https://github.com/ddshan/hand_object_detector) | MIT | none stated | Patch, adapted code, weights mirror |
| [ultralytics](https://github.com/ultralytics/ultralytics) (BoT-SORT) | AGPL-3.0 | — | No |
| [HAWOR](https://github.com/ThunderVVV/HaWoR) | CC BY-NC-ND 4.0 | CC BY-NC-ND 4.0 | Patch |
| [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | BSD-3-Clause | BSD-3-Clause | Patch |
| [MoGe-2](https://github.com/microsoft/MoGe) | MIT | MIT | No |
| [GeoCalib](https://github.com/cvg/GeoCalib) | Apache-2.0 | CC BY 4.0 | Patch |
| [Detectron2](https://github.com/facebookresearch/detectron2) ViTDet-H | Apache-2.0 | CC BY-SA 3.0 | Patch, adapted code |
| [SAM 2](https://github.com/facebookresearch/sam2) | Apache-2.0 | Apache-2.0 | Patch, adapted code |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache-2.0 | Apache-2.0 | No |
| [ProPainter](https://github.com/sczhou/ProPainter) | S-Lab 1.0, non-commercial | S-Lab 1.0 | Patch |
| [PyRoKi](https://github.com/chungmin99/pyroki) | MIT | — | Adapted code |
| [jaxls](https://github.com/brentyi/jaxls) | MIT | — | No |
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 5.1 | NVIDIA Omniverse licence | — | No |
| [MANO](https://mano.is.tue.mpg.de) / [smplx](https://github.com/vchoutas/smplx) | research only | research only | No |
| [ALLEX model](https://github.com/wirobotics-rih/allex_model) | BSD-3-Clause | meshes: evaluation only | Patch |

## EPIC-KITCHENS-100

`examples/clips/`, `docs/pipeline_example.jpg`, `docs/teaser.png` and the left panel of
`docs/wrist_frames.png` derive from EPIC-KITCHENS-100, © Damen et al.,
<https://epic-kitchens.github.io/>, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and provided without warranties
or conditions of any kind. Every one of them is modified from the original material.

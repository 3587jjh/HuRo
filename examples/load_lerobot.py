"""Minimal PyTorch loader for a stage-11 LeRobot dataset, and the reference for reading one.
Feature conventions:

  observation.state / action   (J,) joint angles, J = 48 for allex.
                               action[t] = state[t+1], and the last frame repeats.
  *_eef_{left,right}           (9,) wrist pose in the robot base frame:
                               position xyz + first two ROWS of the rotation matrix
  observation.state_cam_frame  (9,) camera pose, same base frame and 3+6 layout, OpenCV axes
  meta/info.json               per-dimension names for every vector feature above
  meta/modality.json           which slice of observation.state / action is which body part

Running this file prints that layout for a dataset.

  python examples/load_lerobot.py examples/clips_lerobot/allex/192x342
"""
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset

VECTOR_FEATURES = [
    "observation.state", "action",
    "observation.state_eef_left", "observation.state_eef_right",
    "action_eef_left", "action_eef_right",
    "observation.state_cam_frame",
]


class HuRoLeRobotDataset(Dataset):
    """Frame-level dataset: each item is one timestep with its video frame, vectors and task.

    Values come out as stored. No meta/stats.json is written, so fill it in before training."""

    def __init__(self, root):
        self.root = Path(root)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = info["fps"]
        self.chunk_size = info["chunks_size"]
        self.video_key = next(k for k in info["features"] if k.startswith("observation.images."))
        # Per-dimension names and declared widths. modality.json names the slices of each vector.
        self.feature_names = {f: info["features"][f]["names"] for f in VECTOR_FEATURES}
        self._widths = {f: info["features"][f]["shape"][0] for f in VECTOR_FEATURES}
        modality_path = self.root / "meta" / "modality.json"
        self.modality = json.loads(modality_path.read_text()) if modality_path.is_file() else {}

        tasks = [json.loads(l) for l in (self.root / "meta" / "tasks.jsonl").read_text().splitlines()]
        self.task_by_index = {t["task_index"]: t["task"] for t in tasks}

        episodes = [json.loads(l) for l in (self.root / "meta" / "episodes.jsonl").read_text().splitlines()]
        self.episode_lengths = [e["length"] for e in episodes]
        self.index = [(ep_i, t) for ep_i, n in enumerate(self.episode_lengths) for t in range(n)]

        self._cache_ep = -1  # one-episode cache: sequential reads decode each video once
        self._cache = None

    def __len__(self):
        return len(self.index)

    def _episode_paths(self, ep_i):
        chunk = ep_i // self.chunk_size
        return (self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_i:06d}.parquet",
                self.root / "videos" / f"chunk-{chunk:03d}" / self.video_key / f"episode_{ep_i:06d}.mp4")

    def _load_episode(self, ep_i):
        if self._cache_ep == ep_i:
            return self._cache
        import pyarrow.parquet as pq
        pq_path, mp4_path = self._episode_paths(ep_i)
        tbl = pq.read_table(pq_path)
        vectors = {f: np.asarray(tbl[f].to_pylist(), dtype=np.float32) for f in VECTOR_FEATURES}
        for f, v in vectors.items():
            assert v.shape[1] == self._widths[f], \
                f"{f} is {v.shape[1]}-D in episode {ep_i}, meta/info.json declares {self._widths[f]}"
        task_index = tbl["task_index"][0].as_py()
        with av.open(str(mp4_path)) as container:
            frames = [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]
        assert len(frames) == tbl.num_rows, f"video/parquet frame mismatch in episode {ep_i}"
        self._cache_ep, self._cache = ep_i, (vectors, frames, task_index)
        return self._cache

    def __getitem__(self, i):
        ep_i, t = self.index[i]
        vectors, frames, task_index = self._load_episode(ep_i)
        item = {f: torch.from_numpy(vectors[f][t]) for f in VECTOR_FEATURES}
        item["image"] = torch.from_numpy(frames[t].copy())  # (H, W, 3) uint8 RGB
        item["task"] = self.task_by_index[task_index]
        item["episode_index"] = ep_i
        item["frame_index"] = t
        return item


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    ds = HuRoLeRobotDataset(root)
    n_ep = len(ds.episode_lengths)
    print(f"{root}: {n_ep} episodes, {len(ds)} frames, fps {ds.fps}")

    item = ds[0]
    for k in VECTOR_FEATURES:
        v = item[k]
        print(f"  {k:>30} {tuple(v.shape)} range [{v.min():+.3f}, {v.max():+.3f}]")
    print(f"  {'image':>30} {tuple(item['image'].shape)} {item['image'].dtype}")
    print(f"  task: {item['task']!r}")

    # What those vectors hold: modality.json's slices, resolved through info.json's names.
    # `action` repeats the same slices, so only the observation side is worth printing.
    if ds.modality:
        print("\nlayout (meta/modality.json over meta/info.json names)")
        for group, spec in ds.modality.get("state", {}).items():
            key, a, b = spec["original_key"], spec["start"], spec["end"]
            names = ds.feature_names.get(key)
            if names is None:
                continue
            span = names[a:b]
            shown = f"{span[0]} .. {span[-1]}" if len(span) > 2 else ", ".join(span)
            print(f"  {key:28} [{a:2}:{b:2}] {group:18} {shown}")
        print("  action holds the same slices. Its names drop the state suffix, and the wrist "
              "poses live under action_eef_{left,right}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8)  # keep episode-sequential order for the frame cache
    batch = next(iter(loader))
    print(f"batch: state {tuple(batch['observation.state'].shape)}, "
          f"image {tuple(batch['image'].shape)}, tasks {len(batch['task'])}")


if __name__ == "__main__":
    main()

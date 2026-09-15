# Stage 9: robot overlay (Isaac Sim). Renders the retargeted robot into each segment's
# arm-removed video. Reads stage 8's parquet + stage 7's inpainted video. Writes the overlay
# video + the parquet with a per-frame overlay_valid column.
#
#   CUDA_VISIBLE_DEVICES=0 python pipeline/stage9_robot_overlay.py \
#       --input_dir /path/to/clips --part 1/1 --no_tqdm
import subprocess

# Fail loudly before importing Isaac Sim: a node without a usable GPU renders black frames.
_gpu_check = subprocess.run(["nvidia-smi"], capture_output=True)
if _gpu_check.returncode != 0:
    raise RuntimeError("GPU not available: nvidia-smi failed.")

import ctypes
import faulthandler
import json
import os
import os.path as osp
import signal
import sys
import tempfile
import time

# Isaac Sim is proprietary. Setting OMNI_KIT_ACCEPT_EULA=Y accepts NVIDIA's licence, so nothing
# in this repository sets it. run_pipeline.sh checks for it and stops without it.
# setup/docker_run.sh forwards the value already in the environment.
if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() not in ("y", "yes", "1"):
    raise RuntimeError(
        "stage 9 renders through Isaac Sim, licensed under the NVIDIA Omniverse License "
        "Agreement (site-packages/isaacsim/LICENSE.txt). Read it, then set "
        "OMNI_KIT_ACCEPT_EULA=Y to accept."
    )

# Process-level hang recovery: a supervisor relaunches the worker when its heartbeat file goes
# stale (exit 86 = self-detected hang). Segments whose outputs are already on disk are skipped.
_HEARTBEAT_ENV = "HURO_S9_HEARTBEAT"
_HANG_EXIT_CODE = 86
_HANG_TIMEOUT = float(os.environ.get("HURO_S9_HANG_TIMEOUT", 1200))
_MAX_RESTARTS = int(os.environ.get("HURO_S9_MAX_RESTARTS", 3))
_PR_SET_PDEATHSIG = 1   # prctl option from <linux/prctl.h>


def _touch_heartbeat():
    """Mark progress for the supervisor and re-arm the faulthandler stack dump."""
    heartbeat = os.environ.get(_HEARTBEAT_ENV)
    if heartbeat:
        try:
            os.utime(heartbeat, None)
        except OSError:
            pass
    faulthandler.dump_traceback_later(max(60.0, _HANG_TIMEOUT - 120.0), exit=False)


def _supervise() -> int:
    fd, heartbeat = tempfile.mkstemp(prefix="huro_s9_heartbeat_")
    os.close(fd)
    env = dict(os.environ, **{_HEARTBEAT_ENV: heartbeat})
    # SIGTERM and SIGHUP leave through the finally below, as Ctrl+C does.
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda s, _frame: sys.exit(128 + s))
    libc = ctypes.CDLL(None, use_errno=True)
    child = None
    try:
        for attempt in range(1, _MAX_RESTARTS + 2):
            os.utime(heartbeat, None)
            # No terminal signal reaches the worker in its own session. The kernel kills it when
            # the supervisor dies.
            child = subprocess.Popen([sys.executable] + sys.argv, env=env,
                                     start_new_session=True,
                                     preexec_fn=lambda: libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL))
            returncode = None
            while returncode is None:
                try:
                    returncode = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if time.time() - os.path.getmtime(heartbeat) > _HANG_TIMEOUT:
                        print(f"Error: no progress for {_HANG_TIMEOUT:.0f}s. Isaac Sim likely "
                              f"hung. Killing the run (attempt {attempt}/{_MAX_RESTARTS + 1}).",
                              flush=True)
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                        break
            if returncode is not None and returncode < 0:
                sig = -returncode
                print(f"Error: Isaac Sim crashed (signal {sig}, {signal.strsignal(sig)}).",
                      "Rerun with OVERLAY_NO_SUPPRESS=1 to see its output, and read the newest",
                      "log under site-packages/isaacsim/kit/logs.", flush=True)
            if returncode is not None and returncode != _HANG_EXIT_CODE:
                return returncode
            if returncode == _HANG_EXIT_CODE:
                print(f"Relaunching after a self-detected hang "
                      f"(attempt {attempt}/{_MAX_RESTARTS + 1}).", flush=True)
        print("Error: the stage kept hanging after every restart.", flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        # A worker left running would keep writing beside a restarted run.
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        os.unlink(heartbeat)


if __name__ == "__main__" and _HEARTBEAT_ENV not in os.environ:
    sys.exit(_supervise())


# Where the Vulkan loader looks for driver manifests when none is named explicitly.
_VULKAN_ICD_DIRS = (
    "/usr/local/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d",
    "/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d",
    osp.expanduser("~/.local/share/vulkan/icd.d"),
)


def _check_gpu_rendering():
    """Check Vulkan GPU rendering. Clear breakage is fatal. Other problems warn."""
    errors, warnings_ = [], []

    if os.environ.get("HURO_SKIP_GPU_RENDER_CHECK"):
        return

    try:
        ctypes.CDLL("libvulkan.so.1")
    except OSError as e:
        errors.append(f"Cannot load libvulkan.so.1: {e}")

    # An explicitly named driver manifest must exist. Otherwise just look for any.
    icd_paths = [p for p in (os.environ.get("VK_DRIVER_FILES", "")
                             or os.environ.get("VK_ICD_FILENAMES", "")).split(":") if p]
    if icd_paths:
        for icd_path in icd_paths:
            if not os.path.isfile(icd_path):
                errors.append(f"Vulkan driver file does not exist: {icd_path}")
    else:
        icd_paths = [osp.join(d, f) for d in _VULKAN_ICD_DIRS if osp.isdir(d)
                     for f in sorted(os.listdir(d)) if f.endswith(".json")]
        if not icd_paths:
            warnings_.append(f"No Vulkan driver manifest found under {', '.join(_VULKAN_ICD_DIRS)}")

    # 0-byte driver stubs shadowing the real library kill rendering silently.
    named_libs, loaded_libs = 0, 0
    for icd_path in icd_paths:
        if not os.path.isfile(icd_path):
            continue
        try:
            with open(icd_path) as f:
                lib_path = json.load(f).get("ICD", {}).get("library_path", "")
        except (json.JSONDecodeError, OSError):
            continue
        if not lib_path:
            continue
        named_libs += 1
        if os.path.isfile(lib_path) and os.path.getsize(lib_path) == 0:
            errors.append(f"Vulkan driver library {lib_path} is a 0-byte stub ({icd_path})")
            continue
        try:
            ctypes.CDLL(lib_path)
            loaded_libs += 1
        except OSError as e:
            warnings_.append(f"Vulkan driver '{lib_path}' from {icd_path} did not load: {e}")

    # A manifest failing to load only warns if another loads. None loading is an error.
    if named_libs and not loaded_libs and not errors:
        errors.append("No Vulkan driver library named by a manifest could be loaded. The "
                      "graphics driver userspace is missing (check NVIDIA_DRIVER_CAPABILITIES)")

    # Missing OptiX denoiser weights only cause shutdown log noise.
    if not any(os.path.isfile(p) for p in ("/usr/share/nvidia/nvoptix.bin",
                                           "/usr/lib/nvidia/nvoptix.bin")):
        warnings_.append("OptiX denoiser weights (nvoptix.bin) not found. Isaac Sim logs "
                         "errors at shutdown")

    for msg in warnings_:
        print(f"Warning: {msg}", flush=True)

    if errors:
        raise RuntimeError(
            "GPU rendering environment check failed. Isaac Sim would produce black frames.\n"
            "Errors:\n  " + "\n  ".join(errors) + "\n"
            "Fix the Vulkan driver setup, or set HURO_SKIP_GPU_RENDER_CHECK=1 to bypass these checks."
        )


_check_gpu_rendering()

import warnings
warnings.filterwarnings('ignore')

import contextlib
import glob
import logging
import argparse
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.camera import pinhole_from_row
from common.io import decode_row, mark_done, remove_stale_temps, rows_to_parquet
from common.paths import overlay_usd, partition
from common.robot_config import load_robot_config, schema_metadata
from common.video import save_video_task, validate_segment_outputs

logging.getLogger("pipeline.overlay").setLevel(logging.ERROR)

# Exit with the hang code if importing pipeline.overlay stalls.
_import_timer = threading.Timer(900, lambda: (
    print("Error: overlay import timed out after 900s.", flush=True),
    os._exit(_HANG_EXIT_CODE),
))
_import_timer.start()
from pipeline.overlay import (
    RobotOverlayProcessor,
    calculate_camera_params_from_intrinsics,
    check_joint_coverage,
    create_joint_mapping,
    usd_robot_config,
)
_import_timer.cancel()
_touch_heartbeat()

# The overlay processor's tqdm goes to stderr, which Isaac Sim's log capture treats as errors.
import pipeline.overlay.processor as _overlay_processor
_overlay_processor.tqdm = lambda iterable, **kw: iterable


def _suppress_isaac_output():
    """Send stdout+stderr to /dev/null. Returns the saved fds for _restore_output."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    return stdout_fd, stderr_fd


def _restore_output(fds):
    stdout_fd, stderr_fd = fds
    os.dup2(stdout_fd, 1)
    os.close(stdout_fd)
    os.dup2(stderr_fd, 2)
    os.close(stderr_fd)


@contextlib.contextmanager
def _render_timeout(timeout_seconds, label):
    """Exit the process if the block does not finish in time. A hung render never recovers."""
    timer = threading.Timer(timeout_seconds, lambda: (
        print(f"Error: {label} timed out after {timeout_seconds:.0f}s, likely hung.", flush=True),
        os._exit(_HANG_EXIT_CODE),
    ))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', type=str, default='allex',
                        help='Robot config name (configs/<robot_name>.yaml)')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--per_frame_timeout', type=float, default=1.0,
                        help='Per-frame render budget in seconds. A segment gets '
                             'max(300, n_frames * this)')
    parser.add_argument('--stagger_interval', type=float, default=15.0,
                        help='Seconds each successive --part waits before starting, so several '
                             'partitions do not initialize Isaac Sim at the same moment')
    args = parser.parse_args()
    input_dir = osp.normpath(args.input_dir)

    robot_cfg = load_robot_config(args.robot_name)
    metadata = schema_metadata(robot_cfg)
    hide_links = set(robot_cfg.config["overlay"].get("hide_link_names") or [])

    if osp.isdir(input_dir) or osp.isdir(input_dir + '_chunked'):
        chunked_root = input_dir + '_chunked'
        # Enumerate stage 8's retargeted clips.
        clip_ids = partition(sorted(
            osp.basename(p) for p in glob.glob(
                osp.join(glob.escape(osp.join(chunked_root, args.robot_name, 'annot')), '*'))
            if osp.isdir(p)), args.part, key=lambda c: c)
    else:
        assert osp.isfile(input_dir) and input_dir.lower().endswith('.mp4'), \
            f"--input_dir {input_dir}: expected a clip directory, a path whose _chunked " \
            f"sibling exists, or one .mp4 clip"
        chunked_root = osp.join(osp.dirname(input_dir), 'chunked')
        clip_ids = [osp.basename(input_dir)[:-4]]
    if not clip_ids:
        return

    robot_base = osp.join(chunked_root, args.robot_name)
    state_annot_dir = osp.join(robot_base, 'annot')                     # stage 8
    inpaint_video_dir = osp.join(chunked_root, 'original', 'video')     # stage 7
    output_annot_dir = osp.join(robot_base, 'overlay', 'annot')
    output_video_dir = osp.join(robot_base, 'overlay', 'video')

    def is_done(clip_id):
        return osp.exists(osp.join(output_annot_dir, f"{clip_id}.done"))

    print(f'Got {len(clip_ids)} clips in total.')
    clip_ids = [c for c in clip_ids if not is_done(c)]
    print(f'Processing {len(clip_ids)} clips (rest already done).')
    if not clip_ids:
        return
    for clip_id in clip_ids:
        stage8_done = osp.join(state_annot_dir, f"{clip_id}.done")
        if not osp.exists(stage8_done):
            raise RuntimeError(f"{stage8_done} does not exist: stage 8 has not finished {clip_id}, "
                               f"or its marker is missing")
    os.makedirs(output_annot_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)

    processor = None
    smoke_tested = False

    for clip_id in tqdm(clip_ids, desc="clips", unit="clip", position=1, leave=True,
                         dynamic_ncols=True, file=sys.stdout, disable=args.no_tqdm):
        clip_annot_dir = osp.join(state_annot_dir, clip_id)
        if not osp.exists(clip_annot_dir):
            mark_done(output_annot_dir, clip_id); continue

        segment_parquets = sorted(glob.glob(osp.join(glob.escape(clip_annot_dir), '*.parquet')))
        if not segment_parquets:
            mark_done(output_annot_dir, clip_id); continue

        clip_out_annot_dir = osp.join(output_annot_dir, clip_id)
        clip_out_video_dir = osp.join(output_video_dir, clip_id)
        os.makedirs(clip_out_annot_dir, exist_ok=True)
        os.makedirs(clip_out_video_dir, exist_ok=True)
        remove_stale_temps(clip_out_annot_dir)   # temps of processes that have exited
        remove_stale_temps(clip_out_video_dir)

        # Keep segments whose parquet and video are both intact. The validator deletes the rest.
        existing_stems = set()
        for stem in {osp.basename(f)[:-len('.parquet')]
                     for f in glob.glob(osp.join(glob.escape(clip_out_annot_dir), '*.parquet'))} | \
                    {osp.basename(f)[:-len('.mp4')]
                     for f in glob.glob(osp.join(glob.escape(clip_out_video_dir), '*.mp4'))}:
            if validate_segment_outputs(
                annot_path=osp.join(clip_out_annot_dir, f'{stem}.parquet'),
                video_path=osp.join(clip_out_video_dir, f'{stem}.mp4'),
            ):
                existing_stems.add(stem)

        # Isaac Sim is initialized once. A resolution change only resizes the render product.
        intr_cols = ["height", "width", "fx", "fy", "cx", "cy", "xi",
                     "pinhole_fx", "pinhole_fy", "pinhole_cx", "pinhole_cy"]
        present = set(pq.ParquetFile(segment_parquets[0]).schema_arrow.names)
        first_row = decode_row(pq.read_table(
            segment_parquets[0], columns=[c for c in intr_cols if c in present]).to_pylist()[0])
        clip_h, clip_w = int(first_row["height"]), int(first_row["width"])
        fx, fy, cx, cy = pinhole_from_row(first_row)
        clip_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        clip_cam_params = calculate_camera_params_from_intrinsics(
            intrinsics=clip_K, extrinsics=np.eye(4, dtype=np.float64),
            width=clip_w, height=clip_h, is_world_to_camera=False, verbose=False,
        )

        if processor is None:
            _init_timer = threading.Timer(900, lambda: (
                print("Error: Isaac Sim init timed out after 900s, likely hung.", flush=True),
                os._exit(_HANG_EXIT_CODE),
            ))
            _init_timer.start()
            processor = RobotOverlayProcessor(
                robot_path=robot_cfg.urdf_path,
                usd_cache_path=str(overlay_usd(args.robot_name)),
                camera_params=clip_cam_params,
                render_width=clip_w, render_height=clip_h,
                robot_prim_path=robot_cfg.config["overlay"].get("prim_path", "/World/Robot"),
                hide_links=hide_links,
            )
            _init_timer.cancel()
            _touch_heartbeat()

            # PhysX sets the DOF order. Read it from the loaded robot, not the config.
            usd_cfg = usd_robot_config(robot_cfg, processor.renderer.joint_names)
            missing = check_joint_coverage(usd_cfg)
            assert not missing, f"joints absent from the {usd_cfg.name} articulation: {missing}"
            processor.joint_mapping = create_joint_mapping(usd_cfg)
            print(f"Articulation: {usd_cfg.total_dofs} DOFs, driving {usd_cfg.actuated_dofs} "
                  f"from state_qpos + {len(usd_cfg.mimic_joints)} mimic")
        else:
            processor.update_camera_and_resolution(clip_cam_params, clip_w, clip_h)

        for seg_parquet_path in tqdm(segment_parquets, desc=f"{clip_id}", unit="seg",
                                     position=0, leave=False, disable=args.no_tqdm,
                                     file=sys.stdout):
            _touch_heartbeat()
            seg_stem = osp.basename(seg_parquet_path)[:-len('.parquet')]
            if seg_stem in existing_stems:
                continue

            rows = [decode_row(r) for r in pq.read_table(seg_parquet_path).to_pylist()]
            if not rows:
                continue
            n_frames = len(rows)

            states = np.stack([np.asarray(r["state_qpos"], dtype=np.float32) for r in rows])
            cam_poses = np.stack([np.asarray(r["cam_pose_base"], dtype=np.float64).reshape(4, 4)
                                  for r in rows])
            if np.isnan(states).all():
                continue
            assert np.isfinite(states).all(), f"Non-finite values in states for {seg_stem}"

            h, w = int(rows[0]["height"]), int(rows[0]["width"])
            fx, fy, cx, cy = pinhole_from_row(rows[0])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            inpaint_path = osp.join(inpaint_video_dir, clip_id, f'{seg_stem}_inpainted.mp4')
            if not osp.exists(inpaint_path):
                raise FileNotFoundError(f"{inpaint_path} is missing: stage 7 has not finished {clip_id}")

            frames_rgb = []
            with av.open(inpaint_path, "r") as reader:
                for frame in reader.decode(video=0):
                    frames_rgb.append(frame.to_ndarray(format='rgb24'))

            if len(frames_rgb) != n_frames:
                print(f"Warning: frame count mismatch for {seg_stem}: "
                      f"video={len(frames_rgb)}, parquet={n_frames}")
                n_frames = min(len(frames_rgb), n_frames)

            # Snap base to origin between segments only. Per-frame re-posing causes jitter.
            processor.set_robot_base_pose(np.zeros(3, dtype=np.float64),
                                          np.array([1.0, 0.0, 0.0, 0.0]))

            # The semantic annotator needs warm-up flushes: frame 0 is rendered twice and dropped.
            warmup_cfg = processor.joint_mapping(states[0])
            warmup_cam = calculate_camera_params_from_intrinsics(
                intrinsics=K, extrinsics=cam_poses[0], width=w, height=h,
                is_world_to_camera=False, verbose=False,
            )
            with _render_timeout(300, f"warmup for {seg_stem}"):
                _fds = _suppress_isaac_output()
                try:
                    processor.renderer.update_camera_pose(warmup_cam, update_intrinsics=True)
                    for _ in range(2):
                        processor.renderer.create_overlay(
                            robot_cfg=warmup_cfg, video_frame=frames_rgb[0], alpha=1.0)
                finally:
                    _restore_output(_fds)

            processor.renderer.reset_instability_tracking()
            seg_timeout = max(300, n_frames * args.per_frame_timeout)
            with _render_timeout(seg_timeout, f"render {seg_stem} ({n_frames} frames)"):
                overlay_frames_rgb = processor.process_frames(
                    frames=frames_rgb[:n_frames],
                    robot_trajectories=states[:n_frames],
                    camera_poses=cam_poses[:n_frames],
                    camera_intrinsics=K,
                    is_camera_to_world=True,
                    alpha=1.0,
                    # Per-frame heartbeat so long segments never read as a hang
                    progress_cb=lambda _frame_idx: _touch_heartbeat(),
                )

            failed_indices = {fi for fi in processor.renderer.failed_frame_indices
                              if fi < n_frames}
            overlay_valid = np.ones(n_frames, dtype=bool)
            for fi in failed_indices:
                overlay_valid[fi] = False

            if failed_indices:
                print(f"Warning: {len(failed_indices)} render failures in {seg_stem} "
                      f"(frames {sorted(failed_indices)}) -> overlay_valid=False")

            # Every frame of the first segment failed to render: Vulkan is broken despite the
            # startup checks. Stop.
            if not smoke_tested:
                smoke_tested = True
                if len(failed_indices) == n_frames:
                    raise RuntimeError(
                        f"GPU rendering smoke test failed: {len(failed_indices)}/{n_frames} "
                        f"frames of the first segment {seg_stem} failed to render, so Vulkan "
                        f"rendering is not working despite the startup checks passing."
                    )

            out_rows = []
            for t in range(n_frames):
                row = dict(rows[t])
                row["overlay_valid"] = bool(overlay_valid[t])
                out_rows.append(row)
            rows_to_parquet(out_rows, osp.join(clip_out_annot_dir, f"{seg_stem}.parquet"),
                            metadata=metadata)

            overlay_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in overlay_frames_rgb]
            save_video_task(osp.join(clip_out_video_dir, f"{seg_stem}.mp4"),
                            overlay_bgr, 30.0, (w, h))

        _touch_heartbeat()
        mark_done(output_annot_dir, clip_id)

    if processor is not None:
        processor.cleanup()


if __name__ == "__main__":
    # Stagger by partition so several parts of the same job do not initialize Isaac Sim at once.
    _part_str = next((sys.argv[i + 1] for i, a in enumerate(sys.argv)
                      if a == '--part' and i + 1 < len(sys.argv)), '1/1')
    _stagger = next((float(sys.argv[i + 1]) for i, a in enumerate(sys.argv)
                     if a == '--stagger_interval' and i + 1 < len(sys.argv)), 15.0)
    _wait = (int(_part_str.split('/')[0]) - 1) * _stagger
    if _wait > 0:
        print(f"Staggering Isaac Sim init: waiting {_wait:.0f}s", flush=True)
        time.sleep(_wait)
    sys.stdout.reconfigure(line_buffering=True)   # the log is often a pipe or a file
    # The renderer points fd 2 at /dev/null while it runs, so an error is printed to a copy of it.
    _stderr_fd = os.dup(2)
    try:
        main()
    except Exception:
        import traceback
        with os.fdopen(_stderr_fd, "w") as err:
            traceback.print_exc(file=err)
        # Isaac Sim can crash while the interpreter shuts down after an error.
        os._exit(1)

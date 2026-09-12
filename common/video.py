"""Video I/O helpers: per-segment H.264 encode and output-file validation."""
import os
from fractions import Fraction
import av
import cv2
import pyarrow.parquet as pq


def save_video_task(save_path, frames, fps, size):
    """Encode BGR frames to MP4 (H.264/yuv420p, crf=16, 1s GOP). `size` is (width, height)."""
    container = av.open(save_path, mode='w')
    stream = container.add_stream('libx264', rate=Fraction(fps))
    stream.width = size[0]
    stream.height = size[1]
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '16', 'preset': 'medium', 'g': str(int(fps))}

    for frame_bgr in frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def validate_segment_outputs(annot_path, video_path=None):
    """Validate a segment's output files, deleting corrupt/orphaned ones. True iff all valid.
    Parquet: footer readable. MP4 (if given): readable, frame count == parquet num_rows."""
    has_annot = os.path.exists(annot_path)
    has_video = video_path is not None and os.path.exists(video_path)

    if not has_annot and not has_video:
        return False

    if video_path is not None:
        if has_annot and not has_video:
            os.remove(annot_path); return False
        if has_video and not has_annot:
            os.remove(video_path); return False

    try:
        num_rows = pq.read_metadata(annot_path).num_rows
    except Exception:
        os.remove(annot_path)
        if has_video:
            os.remove(video_path)
        return False

    if has_video:
        try:
            with av.open(video_path, "r") as r:
                n_frames = r.streams.video[0].frames or 0
            if n_frames != num_rows:
                os.remove(video_path); os.remove(annot_path); return False
        except Exception:
            os.remove(video_path); os.remove(annot_path); return False

    return True


def validate_video_pair(video_path, ref_video_path):
    """Validate a derived mp4 against its source by frame count, deleting it if invalid.
    Never touches `ref_video_path`. Returns True iff valid."""
    if not os.path.exists(video_path):
        return False
    if not os.path.exists(ref_video_path):
        os.remove(video_path); return False
    try:
        with av.open(ref_video_path, "r") as r:
            ref_frames = r.streams.video[0].frames or 0
    except Exception:
        return False
    try:
        with av.open(video_path, "r") as r:
            n_frames = r.streams.video[0].frames or 0
        if n_frames != ref_frames:
            os.remove(video_path); return False
    except Exception:
        os.remove(video_path); return False
    return True

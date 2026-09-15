"""Video I/O helpers: per-segment H.264 encode and output-file validation."""
import os
from fractions import Fraction
import av
import cv2
import pyarrow.parquet as pq
from common.io import atomic_path


def count_frames(path):
    """Frame count of the first video stream: the count the container stores, or the decoded
    count when it stores none (fragmented MP4, for example)."""
    with av.open(path, "r") as reader:
        stream = reader.streams.video[0]
        return stream.frames or sum(1 for _ in reader.decode(video=0))


def save_video_task(save_path, frames, fps, size):
    """Encode BGR frames to MP4 (H.264/yuv420p, crf=16, 1s GOP) through a temp name and a rename.
    `size` is (width, height), both even."""
    if size[0] % 2 or size[1] % 2:
        raise ValueError(f"{save_path}: size {size[0]}x{size[1]} has an odd side. "
                         "H.264 yuv420p needs an even width and height.")
    tmp_path = atomic_path(save_path)
    try:
        container = av.open(tmp_path, mode='w', format='mp4')
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
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    os.replace(tmp_path, save_path)


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
            n_frames = count_frames(video_path)
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
        ref_frames = count_frames(ref_video_path)
    except Exception:
        return False
    try:
        n_frames = count_frames(video_path)
        if n_frames != ref_frames:
            os.remove(video_path); return False
    except Exception:
        os.remove(video_path); return False
    return True

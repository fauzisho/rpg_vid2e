#!/usr/bin/env python3
"""
Build an event-camera-style dataset from a video: grayscale frames + timestamps.txt,
then ESIM events saved as NPZ + JSON metadata.

Requires: ffmpeg/ffprobe on PATH, and esim_py installed (see repo README).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def probe_video(path: Path) -> tuple[float, int | None]:
    """Return (avg_fps, nb_frames or None if unknown)."""
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    parts = [x.strip() for x in out.strip().split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected ffprobe output: {out!r}")
    # csv order: r_frame_rate, avg_frame_rate, nb_frames
    rate_s = parts[0] or parts[1]
    nb = parts[2] if len(parts) > 2 else ""
    num, den = rate_s.split("/")
    fps = float(num) / float(den)
    n_frames: int | None = None
    if nb and nb != "N/A":
        try:
            n_frames = int(nb)
        except ValueError:
            n_frames = None
    return fps, n_frames


def extract_grayscale_frames(video: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "%08d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            "format=gray",
            "-vsync",
            "0",
            pattern,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    files = sorted(frames_dir.glob("*.png"))
    return len(files)


def write_timestamps(path: Path, n: int, fps: float) -> None:
    dt = 1.0 / fps
    lines = [f"{i * dt:.9f}\n" for i in range(n)]
    path.write_text("".join(lines))


@dataclass
class EventDatasetResult:
    """Paths and stats after building events.npz from video."""

    npz_path: Path
    meta_path: Path
    timestamps_path: Path
    num_events: int
    fps: float
    num_frames: int


def build_event_dataset(
    video: Path,
    out: Path,
    *,
    cp: float = 0.2,
    cn: float = 0.2,
    refractory: float = 0.0,
    log_eps: float = 1e-3,
    use_log: bool = True,
    keep_frames: bool = False,
) -> EventDatasetResult:
    """
    Extract frames, run ESIM, write events.npz + meta.json + timestamps.txt.
    """
    import numpy as np
    import esim_py

    video = video.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    fps, probed_n = probe_video(video)
    frames_dir = out / "frames"

    n = extract_grayscale_frames(video, frames_dir)
    if probed_n is not None and probed_n != n:
        print(
            f"Warning: ffprobe reported {probed_n} frames but ffmpeg extracted {n}; using {n}.",
            file=sys.stderr,
        )
    ts_path = out / "timestamps.txt"
    write_timestamps(ts_path, n, fps)

    esim = esim_py.EventSimulator(cp, cn, refractory, log_eps, use_log)
    events = esim.generateFromFolder(str(frames_dir), str(ts_path))

    npz_path = out / "events.npz"
    np.savez(
        npz_path,
        x=events[:, 0].astype(np.int32),
        y=events[:, 1].astype(np.int32),
        t=events[:, 2].astype(np.float64),
        p=events[:, 3].astype(np.float64),
    )

    meta = {
        "source_video": str(video),
        "fps": fps,
        "num_frames": n,
        "num_events": int(events.shape[0]),
        "contrast_threshold_pos": cp,
        "contrast_threshold_neg": cn,
        "refractory_period_s": refractory,
        "log_eps": log_eps,
        "use_log": use_log,
        "events_file": "events.npz",
        "timestamps_file": "timestamps.txt",
        "format": "columns x,y,t(seconds),polarity in NPZ",
        "snn_ready": True,
        "snn_note": "Load with numpy.load: arrays x,y,t(sec),p (float polarity ±1). "
        "Reshape or bin for spike tensors per your SNN framework.",
    }
    meta_path = out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if not keep_frames:
        for f in frames_dir.glob("*.png"):
            f.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass

    return EventDatasetResult(
        npz_path=npz_path,
        meta_path=meta_path,
        timestamps_path=ts_path,
        num_events=int(events.shape[0]),
        fps=fps,
        num_frames=n,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Video → ESIM event dataset (frames + events.npz)")
    ap.add_argument("--video", "-v", type=Path, required=True, help="Input video (e.g. seq1/video.mp4)")
    ap.add_argument("--out", "-o", type=Path, required=True, help="Output directory for dataset")
    ap.add_argument("--cp", type=float, default=0.2, help="Positive contrast threshold")
    ap.add_argument("--cn", type=float, default=0.2, help="Negative contrast threshold")
    ap.add_argument("--refractory", type=float, default=0.0, help="Refractory period (seconds)")
    ap.add_argument("--log-eps", type=float, default=1e-3, help="log_eps for ESIM")
    ap.add_argument("--no-log", action="store_true", help="Disable log intensity")
    ap.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted PNGs under out/frames (default: delete after events)",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    r = build_event_dataset(
        video,
        args.out,
        cp=args.cp,
        cn=args.cn,
        refractory=args.refractory,
        log_eps=args.log_eps,
        use_log=not args.no_log,
        keep_frames=args.keep_frames,
    )
    print(f"Wrote {r.npz_path} ({r.num_events} events)")
    print(f"Meta: {r.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

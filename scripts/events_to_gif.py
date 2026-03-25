#!/usr/bin/env python3
"""
Render an animated GIF from events.npz (x, y, t, p): each frame shows events in a time slice.
Requires: numpy, pillow (pip install pillow)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Install pillow: pip install pillow") from e


def frame_rgb(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    scale: int = 50,
) -> np.ndarray:
    """H×W×3 uint8, red = +1, green = -1 (same style as esim_py/tests/plot_virtual_events.py)."""
    pos = p > 0
    neg = p < 0
    flat_pos = np.zeros(height * width, dtype=np.uint32)
    flat_neg = np.zeros(height * width, dtype=np.uint32)
    if np.any(pos):
        np.add.at(flat_pos, (x[pos].astype(np.int64) + y[pos].astype(np.int64) * width), 1)
    if np.any(neg):
        np.add.at(flat_neg, (x[neg].astype(np.int64) + y[neg].astype(np.int64) * width), 1)
    r = np.clip(flat_pos.reshape(height, width) * scale, 0, 255).astype(np.uint8)
    g = np.clip(flat_neg.reshape(height, width) * scale, 0, 255).astype(np.uint8)
    b = np.zeros((height, width), dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def render_events_gif(
    npz_path: Path,
    out_path: Path,
    *,
    frames: int = 40,
    gif_fps: float = 12.0,
    scale: int = 2,
) -> None:
    """Write an animated GIF from events.npz to out_path."""
    d = np.load(npz_path)
    x = d["x"].astype(np.int64)
    y = d["y"].astype(np.int64)
    t = d["t"].astype(np.float64)
    p = d["p"].astype(np.float64)

    t0, t1 = float(t.min()), float(t.max())
    if t1 <= t0:
        raise ValueError("Need a positive time span in events.")

    width = int(x.max()) + 1
    height = int(y.max()) + 1

    edges = np.linspace(t0, t1, frames + 1)
    duration_ms = int(1000 / max(gif_fps, 1e-6))

    pil_frames: list[Image.Image] = []
    for i in range(frames):
        lo, hi = edges[i], edges[i + 1]
        m = (t >= lo) & (t < hi) if i < frames - 1 else (t >= lo) & (t <= hi)
        rgb = frame_rgb(x[m], y[m], p[m], height, width)
        im = Image.fromarray(rgb, mode="RGB")
        if scale > 1:
            w, h = im.size
            im = im.resize((w * scale, h * scale), Image.Resampling.NEAREST)
        pil_frames.append(im)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="events.npz → animated GIF")
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to events.npz (default: example/event_ready/seq1/events.npz)",
    )
    ap.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output .gif path (default: next to npz as events.gif)",
    )
    ap.add_argument(
        "--frames",
        "-n",
        type=int,
        default=40,
        help="Number of time bins (GIF frames)",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Playback speed for GIF (frames per second of the output animation)",
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Integer upscale of each frame (nearest-neighbor), for visibility",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    npz_path = (args.npz or (root / "example/event_ready/seq1/events.npz")).resolve()
    if not npz_path.is_file():
        print(f"Not found: {npz_path}")
        return 1

    out_path = args.out
    if out_path is None:
        out_path = npz_path.with_name("events.gif")
    else:
        out_path = out_path.resolve()

    render_events_gif(
        npz_path,
        out_path,
        frames=args.frames,
        gif_fps=args.fps,
        scale=args.scale,
    )

    d = np.load(npz_path)
    t = d["t"].astype(np.float64)
    x = d["x"]
    t0, t1 = float(t.min()), float(t.max())
    width = int(x.max()) + 1
    height = int(d["y"].max()) + 1
    print(f"Wrote {out_path} ({args.frames} frames, {width}×{height} px, t ∈ [{t0:.4f}, {t1:.4f}] s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

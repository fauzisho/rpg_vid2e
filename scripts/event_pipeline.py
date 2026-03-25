#!/usr/bin/env python3
"""
End-to-end pipeline: input video → SNN-ready event dataset + GIF + 3D Event Stream image.

Steps:
  1. ESIM (esim_py): grayscale frames, timestamps, events.npz, meta.json
  2. Animated GIF from events (time-sliced accumulation)
  3. 3D scatter JPG (Time (ms), x, y) — red/blue polarity

Requires: ffmpeg/ffprobe, esim_py, numpy, pillow, matplotlib.
Optional HTML: pip install plotly

Example:
  python scripts/event_pipeline.py --video example/original/seq1/video.mp4 --out example/event_ready/seq1_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_scripts_on_path() -> Path:
    """Allow importing sibling modules when run as scripts/event_pipeline.py."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return script_dir


def main() -> int:
    _ensure_scripts_on_path()
    from events_3d import load_events, plot_matplotlib, plot_plotly_html, subsample
    from events_to_gif import render_events_gif
    from video_to_event_dataset import build_event_dataset

    ap = argparse.ArgumentParser(
        description="Video → events.npz (SNN-ready) + events.gif + event_stream.jpg",
    )
    ap.add_argument("--video", "-v", type=Path, required=True, help="Input video file")
    ap.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/example/event_ready/<video_stem>)",
    )
    # ESIM
    ap.add_argument("--cp", type=float, default=0.2, help="Positive contrast threshold")
    ap.add_argument("--cn", type=float, default=0.2, help="Negative contrast threshold")
    ap.add_argument("--refractory", type=float, default=0.0, help="Refractory period (s)")
    ap.add_argument("--log-eps", type=float, default=1e-3, help="log_eps for ESIM")
    ap.add_argument("--no-log", action="store_true", help="Disable log intensity")
    ap.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted PNGs under out/frames",
    )
    # GIF
    ap.add_argument("--gif-frames", type=int, default=40, help="Time bins in GIF")
    ap.add_argument("--gif-fps", type=float, default=12.0, help="Playback FPS of GIF")
    ap.add_argument("--gif-scale", type=int, default=2, help="GIF frame upscale")
    # 3D
    ap.add_argument(
        "--max-points-3d",
        type=int,
        default=80_000,
        help="Max events for 3D plot (subsampled if more)",
    )
    ap.add_argument("--seed-3d", type=int, default=0, help="Subsample RNG seed for 3D")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Also write interactive Plotly HTML (e.g. event_stream.html)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    video = args.video.resolve()
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    if args.out is not None:
        out_dir = args.out.resolve()
    else:
        out_dir = (root / "example/event_ready" / video.stem).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== 1/3 Event dataset (ESIM) ===")
    r = build_event_dataset(
        video,
        out_dir,
        cp=args.cp,
        cn=args.cn,
        refractory=args.refractory,
        log_eps=args.log_eps,
        use_log=not args.no_log,
        keep_frames=args.keep_frames,
    )
    print(f"  {r.npz_path} ({r.num_events} events)")

    print("=== 2/3 GIF ===")
    gif_path = out_dir / "events.gif"
    render_events_gif(
        r.npz_path,
        gif_path,
        frames=args.gif_frames,
        gif_fps=args.gif_fps,
        scale=args.gif_scale,
    )
    print(f"  {gif_path}")

    print("=== 3/3 3D Event Stream (JPG) ===")
    x, y, t, p = load_events(r.npz_path)
    x, y, t, p = subsample(x, y, t, p, args.max_points_3d, args.seed_3d)
    jpg_path = out_dir / "event_stream.jpg"
    plot_matplotlib(
        x,
        y,
        t,
        p,
        out_path=jpg_path,
        show=False,
        dpi=args.dpi,
        figsize=(10.0, 8.0),
        jpeg_quality=min(95, max(1, args.jpeg_quality)),
    )
    print(f"  {jpg_path} (plotted {len(t)} events)")

    html_rel: str | None = None
    if args.html is not None:
        html_path = Path(args.html)
        if not html_path.is_absolute():
            html_path = out_dir / html_path
        html_path = html_path.resolve()
        html_rel = html_path.name
        x2, y2, t2, p2 = load_events(r.npz_path)
        x2, y2, t2, p2 = subsample(x2, y2, t2, p2, args.max_points_3d, args.seed_3d)
        try:
            plot_plotly_html(x2, y2, t2, p2, html_path)
        except ImportError:
            print("Plotly not installed; skip --html. pip install plotly", file=sys.stderr)
            return 1

    summary = {
        "source_video": str(video),
        "output_dir": str(out_dir),
        "artifacts": {
            "snn_dataset": {
                "events_npz": "events.npz",
                "meta_json": "meta.json",
                "timestamps_txt": "timestamps.txt",
                "description": "NPZ: x,y,t(seconds),p (±1). Use numpy.load for SNN preprocessing.",
            },
            "visualization": {
                "events_gif": "events.gif",
                "event_stream_jpg": "event_stream.jpg",
            },
        },
        "esim": {
            "cp": args.cp,
            "cn": args.cn,
            "refractory_s": args.refractory,
            "log_eps": args.log_eps,
            "use_log": not args.no_log,
        },
    }
    if html_rel is not None:
        summary["artifacts"]["visualization"]["event_stream_html"] = html_rel

    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"=== Done ===\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

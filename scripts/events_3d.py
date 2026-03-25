#!/usr/bin/env python3
"""
3D "Event Stream" visualization: scatter with Time (ms) on one axis, pixel (x, y) on the
other two — red / blue polarity (ON / OFF style), similar to common event-camera plots.

  - Matplotlib (default): static JPG or PNG, optional interactive window (--show).
  - Plotly (optional): interactive HTML -- install: pip install plotly

Example:
  python scripts/events_3d.py
  python scripts/events_3d.py -o example/event_ready/seq1/event_stream.jpg
  python scripts/events_3d.py --html example/event_ready/seq1/events_3d.html
  python scripts/events_3d.py --show
  python scripts/events_3d.py --html out.html --no-save   # HTML only (needs plotly)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_events(npz_path: Path):
    d = np.load(npz_path)
    return (
        d["x"].astype(np.float64),
        d["y"].astype(np.float64),
        d["t"].astype(np.float64),
        d["p"].astype(np.float64),
    )


def subsample(x, y, t, p, max_points: int, seed: int = 0):
    n = len(t)
    if n <= max_points:
        return x, y, t, p
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx], t[idx], p[idx]


def _style_3d_axes(ax) -> None:
    """Light gray grid / white panes, similar to typical event-stream figures."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("0.85")
        axis.pane.set_linewidth(0.8)
    ax.grid(True, color="0.75", linestyle="-", linewidth=0.4, alpha=0.9)


def plot_matplotlib(
    x,
    y,
    t,
    p,
    out_path: Path | None,
    show: bool,
    dpi: int,
    figsize: tuple[float, float],
    jpeg_quality: int = 92,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d

    # Match common event plots: Time (ms) on X, spatial coordinates on Y and Z.
    # Polarity: red = ON (+1), blue = OFF (−1).
    t_ms = t * 1000.0
    pos = "#c41e1e"
    neg = "#1e4a8c"
    c = np.where(p > 0, pos, neg)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.scatter(
        t_ms,
        x,
        y,
        c=c,
        s=1.2,
        alpha=0.38,
        depthshade=True,
        linewidths=0,
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("x (px)")
    ax.set_zlabel("y (px)")
    ax.set_title("Event Stream")
    _style_3d_axes(ax)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suf = out_path.suffix.lower()
        if suf in (".jpg", ".jpeg"):
            fig.savefig(
                out_path,
                dpi=dpi,
                format="jpeg",
                facecolor="white",
                pil_kwargs={"quality": jpeg_quality, "optimize": True},
            )
        else:
            fig.savefig(out_path, dpi=dpi, facecolor="white")
        print(f"Wrote {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_plotly_html(x, y, t, p, out_path: Path) -> None:
    import plotly.graph_objects as go

    t_ms = t * 1000.0
    pos = p > 0
    traces = []
    if np.any(pos):
        traces.append(
            go.Scatter3d(
                x=t_ms[pos],
                y=x[pos],
                z=y[pos],
                mode="markers",
                name="ON (+1)",
                marker=dict(size=2, color="#c41e1e", opacity=0.4),
            )
        )
    if np.any(~pos):
        traces.append(
            go.Scatter3d(
                x=t_ms[~pos],
                y=x[~pos],
                z=y[~pos],
                mode="markers",
                name="OFF (−1)",
                marker=dict(size=2, color="#1e4a8c", opacity=0.4),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Time (ms)",
            yaxis_title="x (px)",
            zaxis_title="y (px)",
            aspectmode="data",
        ),
        title="Event Stream (drag to rotate)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Wrote {out_path} (open in browser)")


def main() -> int:
    ap = argparse.ArgumentParser(description="3D scatter of events (x, y, t)")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--npz",
        type=Path,
        default=root / "example/event_ready/seq1/events.npz",
        help="Path to events.npz",
    )
    ap.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output image path (.jpg or .png). Default: <npz_dir>/event_stream.jpg",
    )
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write interactive Plotly HTML (requires: pip install plotly)",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=80_000,
        help="Random subsample for speed (full set can freeze the viewer)",
    )
    ap.add_argument("--seed", type=int, default=0, help="Subsampling RNG seed")
    ap.add_argument(
        "--show",
        action="store_true",
        help="Open matplotlib 3D window (rotate with mouse)",
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write image file (use with --html or --show only)",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality 1–95 (only for .jpg/.jpeg output)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--width", type=float, default=10.0, help="Figure width (inches)")
    ap.add_argument("--height", type=float, default=8.0, help="Figure height (inches)")
    args = ap.parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.is_file():
        print(f"Not found: {npz_path}")
        return 1

    x, y, t, p = load_events(npz_path)
    x, y, t, p = subsample(x, y, t, p, args.max_points, args.seed)
    print(f"Plotting {len(t)} events (max_points={args.max_points})")

    if args.out is not None:
        out_img = args.out.resolve()
    else:
        out_img = npz_path.parent / "event_stream.jpg"

    save_img = not args.no_save
    if args.show or save_img:
        plot_matplotlib(
            x,
            y,
            t,
            p,
            out_path=out_img if save_img else None,
            show=args.show,
            dpi=args.dpi,
            figsize=(args.width, args.height),
            jpeg_quality=min(95, max(1, args.jpeg_quality)),
        )

    if args.html is not None:
        try:
            plot_plotly_html(x, y, t, p, args.html.resolve())
        except ImportError:
            print("Plotly not installed. Run: pip install plotly")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

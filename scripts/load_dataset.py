#!/usr/bin/env python3
"""Load an event dataset produced by video_to_event_dataset.py."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "example/event_ready/seq1"


def main() -> None:
    npz_path = DATA / "events.npz"
    meta_path = DATA / "meta.json"

    d = np.load(npz_path)
    x, y, t, p = d["x"], d["y"], d["t"], d["p"]

    print("Arrays (same length = one row per event):")
    print("  x, y  — pixel column, row (int)")
    print("  t     — time in seconds (float)")
    print("  p     — polarity +1 / -1")
    print()
    print(f"num_events: {len(t)}")
    print(f"t range:    {t.min():.6f} … {t.max():.6f} s (duration {t.max() - t.min():.6f} s)")
    print(f"polarity:   +1 → {(p > 0).sum()}   -1 → {(p < 0).sum()}")
    print(f"spatial:    x ∈ [{x.min()}, {x.max()}]   y ∈ [{y.min()}, {y.max()}] → ~{1 + x.max()}×{1 + y.max()} px")

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        print()
        print("meta.json:")
        for k in ("source_video", "fps", "num_frames", "contrast_threshold_pos", "contrast_threshold_neg"):
            if k in meta:
                print(f"  {k}: {meta[k]}")


if __name__ == "__main__":
    main()

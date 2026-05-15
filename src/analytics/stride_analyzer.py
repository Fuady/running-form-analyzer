"""
stride_analyzer.py
──────────────────
Analytics: Compute stride-level metrics from biomechanical feature sequences.

Metrics computed per clip:
  - cadence (estimated steps per minute)
  - stride_length_proxy (normalized, from peak-to-peak ankle displacement)
  - vertical_oscillation_mean (mean hip vertical movement per stride)
  - vertical_oscillation_ratio (oscillation / stride length — efficiency proxy)
  - ground_contact_ratio (fraction of gait cycle in stance, from ankle y)
  - left_right_asymmetry (difference between L/R stride metrics)

Usage:
    python src/analytics/stride_analyzer.py \\
        --features data/processed/features/biomech_features.csv \\
        --output   data/processed/features/stride_metrics.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def estimate_cadence(ankle_y: np.ndarray, fps: float = 30.0) -> float:
    """
    Estimate cadence (steps/min) from ankle vertical position.
    Each stride cycle = two peaks (one per foot).
    """
    if len(ankle_y) < 10:
        return np.nan
    # Peaks in ankle_y correspond to foot lift events
    peaks, _ = find_peaks(-ankle_y, distance=int(fps * 0.2))  # min 0.2s between steps
    if len(peaks) < 2:
        return np.nan
    duration_sec = len(ankle_y) / fps
    steps = len(peaks)
    return float(steps / duration_sec * 60)


def estimate_stride_length(ankle_x: np.ndarray, fps: float = 30.0) -> float:
    """
    Estimate normalized stride length from peak-to-peak ankle x displacement.
    """
    if len(ankle_x) < 10:
        return np.nan
    peaks, _ = find_peaks(ankle_x, distance=int(fps * 0.3))
    if len(peaks) < 2:
        return np.nan
    displacements = np.abs(np.diff(ankle_x[peaks]))
    return float(np.median(displacements))


def compute_vertical_oscillation(hip_y: np.ndarray, fps: float = 30.0) -> float:
    """
    Mean vertical displacement of hip per stride cycle (normalized torso units).
    Lower is better (< 0.08 torso units for efficient runners).
    """
    if len(hip_y) < 10:
        return np.nan
    # Smooth first
    from scipy.signal import savgol_filter
    w = min(11, len(hip_y) - 1 if len(hip_y) % 2 == 0 else len(hip_y))
    if w >= 3 and w % 2 == 1:
        smooth = savgol_filter(hip_y, w, 2)
    else:
        smooth = hip_y
    peaks, _   = find_peaks(-smooth, distance=int(fps * 0.3))
    troughs, _ = find_peaks(smooth,  distance=int(fps * 0.3))
    if len(peaks) == 0 or len(troughs) == 0:
        return float(np.std(hip_y))
    return float(np.mean(np.abs(smooth[peaks].mean() - smooth[troughs].mean())))


def compute_ground_contact_ratio(ankle_y: np.ndarray, fps: float = 30.0) -> float:
    """
    Fraction of frames where ankle y is near its minimum (stance phase proxy).
    """
    if len(ankle_y) < 5:
        return np.nan
    threshold = np.percentile(ankle_y, 30)  # lower 30% = near ground
    return float((ankle_y > threshold).mean())


def compute_stride_metrics(group: pd.DataFrame, fps: float = 30.0) -> dict:
    """Compute all stride metrics for a single clip."""
    group = group.sort_values("frame")

    metrics: dict = {
        "cadence_spm": np.nan,
        "stride_length_proxy": np.nan,
        "vertical_oscillation": np.nan,
        "ground_contact_ratio": np.nan,
        "lr_cadence_asymmetry": np.nan,
        "trunk_lean_mean": np.nan,
        "trunk_lean_std": np.nan,
        "arm_swing_symmetry_mean": np.nan,
        "max_overstride_mean": np.nan,
        "hip_drop_mean": np.nan,
    }

    # Cadence from ankle data in original CSV
    if "cadence_proxy" in group.columns:
        cp = group["cadence_proxy"].dropna().values
        if len(cp) > 5:
            metrics["cadence_spm"] = float(np.mean(cp) * 60 / fps)

    # Stride metrics from biomech features
    if "vertical_oscillation" in group.columns:
        metrics["vertical_oscillation"] = float(group["vertical_oscillation"].dropna().mean())

    for col, key in [
        ("trunk_lean_angle",    "trunk_lean_mean"),
        ("trunk_lean_angle",    "trunk_lean_std"),
        ("arm_swing_symmetry",  "arm_swing_symmetry_mean"),
        ("max_overstride",      "max_overstride_mean"),
        ("hip_drop_angle",      "hip_drop_mean"),
    ]:
        if col in group.columns:
            vals = group[col].dropna()
            if key.endswith("_std"):
                metrics[key] = float(vals.std()) if len(vals) > 1 else 0.0
            else:
                metrics[key] = float(vals.mean()) if len(vals) > 0 else np.nan

    return metrics


def main(args: argparse.Namespace) -> None:
    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features not found: {features_path}")
        sys.exit(1)

    df = pd.read_csv(features_path)
    logger.info(f"Computing stride metrics for {df['video_stem'].nunique()} clips...")

    rows = []
    for video_stem, group in df.groupby("video_stem"):
        form_class = group["form_class"].iloc[0]
        metrics    = compute_stride_metrics(group)
        rows.append({"video_stem": video_stem, "form_class": form_class, **metrics})

    result_df = pd.DataFrame(rows)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    logger.info(f"\n✅ Stride metrics saved: {out_path}")
    logger.info(f"   {len(result_df)} clips × {len(result_df.columns)} metrics")
    logger.info("\nMean values by form class:")
    logger.info(result_df.groupby("form_class")[
        ["trunk_lean_mean", "arm_swing_symmetry_mean", "max_overstride_mean"]
    ].mean().round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stride-level metrics")
    parser.add_argument("--features", default="data/processed/features/biomech_features.csv")
    parser.add_argument("--output",   default="data/processed/features/stride_metrics.csv")
    args = parser.parse_args()
    main(args)

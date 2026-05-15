"""
biomech_features.py
───────────────────
Analytics: Engineer 18 biomechanical features per frame from normalized poses.

Running-specific features:
  Trunk lean angle        — forward tilt from vertical
  Stride angle            — rear leg extension at push-off
  Knee drive angle        — front knee lift
  Hip drop angle          — lateral pelvic tilt during stance
  Arm swing symmetry      — L/R elbow angle difference
  Foot strike position    — foot x relative to hip (overstriding proxy)
  Vertical oscillation    — hip vertical displacement
  Elbow angle             — arm bend during swing
  Head alignment          — head vs trunk axis
  Cadence proxy           — ankle vertical velocity
  + angular velocities for key joints

Usage:
    python src/analytics/biomech_features.py \\
        --keypoints data/processed/keypoints \\
        --labels    data/annotations/form_labels.csv \\
        --output    data/processed/features
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"frame", "timestamp_ms", "pose_detected", "form_class",
             "video_stem", "outcome"}


# ─── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at B formed by A-B-C (degrees)."""
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def angle_from_vertical(p_bottom: np.ndarray, p_top: np.ndarray) -> float:
    """Angle of segment (bottom→top) from vertical (upward), degrees."""
    vec = p_top - p_bottom
    vertical = np.array([0, -1])  # up in image coords
    cos_a = np.dot(vec[:2], vertical) / (np.linalg.norm(vec[:2]) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def get_pt(row: pd.Series, name: str) -> np.ndarray:
    """Extract (x, y) for a landmark; returns NaN array if missing."""
    x = row.get(f"{name}_x", np.nan)
    y = row.get(f"{name}_y", np.nan)
    return np.array([float(x), float(y)])


# ─── PER-FRAME FEATURE COMPUTATION ───────────────────────────────────────────

def compute_features_frame(row: pd.Series) -> dict:
    """Compute all biomechanical features for one frame."""
    f: dict = {}

    # Landmark extraction
    nose    = get_pt(row, "nose")
    l_sh    = get_pt(row, "left_shoulder")
    r_sh    = get_pt(row, "right_shoulder")
    l_el    = get_pt(row, "left_elbow")
    r_el    = get_pt(row, "right_elbow")
    l_wr    = get_pt(row, "left_wrist")
    r_wr    = get_pt(row, "right_wrist")
    l_hip   = get_pt(row, "left_hip")
    r_hip   = get_pt(row, "right_hip")
    l_kn    = get_pt(row, "left_knee")
    r_kn    = get_pt(row, "right_knee")
    l_an    = get_pt(row, "left_ankle")
    r_an    = get_pt(row, "right_ankle")

    mid_hip = (l_hip + r_hip) / 2
    mid_sh  = (l_sh  + r_sh ) / 2

    # ── Trunk lean ────────────────────────────────────────────────────────────
    if not (np.isnan(mid_hip).any() or np.isnan(mid_sh).any()):
        f["trunk_lean_angle"] = angle_from_vertical(mid_hip, mid_sh)
    else:
        f["trunk_lean_angle"] = np.nan

    # ── Knee drive (front knee) ───────────────────────────────────────────────
    # Front leg = whichever knee is higher (lower y in image coords)
    if not np.isnan(l_kn).any() and not np.isnan(r_kn).any():
        if l_kn[1] < r_kn[1]:   # left knee higher
            front_hip, front_kn, front_an = l_hip, l_kn, l_an
            rear_hip,  rear_kn,  rear_an  = r_hip, r_kn, r_an
        else:
            front_hip, front_kn, front_an = r_hip, r_kn, r_an
            rear_hip,  rear_kn,  rear_an  = l_hip, l_kn, l_an

        if not any(np.isnan(v).any() for v in [front_hip, front_kn]):
            f["knee_drive_angle"] = angle_from_vertical(front_hip, front_kn)
        else:
            f["knee_drive_angle"] = np.nan

        # Stride angle — rear leg extension
        if not any(np.isnan(v).any() for v in [mid_hip, rear_kn, rear_an]):
            f["stride_angle"] = angle_3pts(mid_hip, rear_kn, rear_an)
        else:
            f["stride_angle"] = np.nan

        # Rear knee angle
        if not any(np.isnan(v).any() for v in [rear_hip, rear_kn, rear_an]):
            f["rear_knee_angle"] = angle_3pts(rear_hip, rear_kn, rear_an)
        else:
            f["rear_knee_angle"] = np.nan

        # Front knee angle
        if not any(np.isnan(v).any() for v in [front_hip, front_kn, front_an]):
            f["front_knee_angle"] = angle_3pts(front_hip, front_kn, front_an)
        else:
            f["front_knee_angle"] = np.nan
    else:
        f["knee_drive_angle"] = f["stride_angle"] = np.nan
        f["rear_knee_angle"]  = f["front_knee_angle"] = np.nan

    # ── Hip drop ──────────────────────────────────────────────────────────────
    if not (np.isnan(l_hip).any() or np.isnan(r_hip).any()):
        delta_y = abs(l_hip[1] - r_hip[1])
        delta_x = abs(l_hip[0] - r_hip[0]) + 1e-9
        f["hip_drop_angle"] = float(np.degrees(np.arctan(delta_y / delta_x)))
    else:
        f["hip_drop_angle"] = np.nan

    # ── Foot strike position ──────────────────────────────────────────────────
    # Positive = foot ahead of hip (overstriding proxy)
    if not np.isnan(mid_hip).any():
        if not np.isnan(l_an).any():
            f["left_foot_strike_pos"]  = float(l_an[0] - mid_hip[0])
        else:
            f["left_foot_strike_pos"]  = np.nan
        if not np.isnan(r_an).any():
            f["right_foot_strike_pos"] = float(r_an[0] - mid_hip[0])
        else:
            f["right_foot_strike_pos"] = np.nan
        # Max overstride (the foot that's furthest forward)
        vals = [v for v in [f["left_foot_strike_pos"], f["right_foot_strike_pos"]]
                if not np.isnan(v)]
        f["max_overstride"] = float(max(vals)) if vals else np.nan
    else:
        f["left_foot_strike_pos"] = f["right_foot_strike_pos"] = f["max_overstride"] = np.nan

    # ── Arm swing ─────────────────────────────────────────────────────────────
    if not any(np.isnan(v).any() for v in [l_sh, l_el, l_wr]):
        f["left_elbow_angle"] = angle_3pts(l_sh, l_el, l_wr)
    else:
        f["left_elbow_angle"] = np.nan

    if not any(np.isnan(v).any() for v in [r_sh, r_el, r_wr]):
        f["right_elbow_angle"] = angle_3pts(r_sh, r_el, r_wr)
    else:
        f["right_elbow_angle"] = np.nan

    if not np.isnan(f.get("left_elbow_angle", np.nan)) and not np.isnan(f.get("right_elbow_angle", np.nan)):
        f["arm_swing_symmetry"] = abs(f["left_elbow_angle"] - f["right_elbow_angle"])
    else:
        f["arm_swing_symmetry"] = np.nan

    # Arm crossing: wrist x relative to body midline
    mid_x = float(mid_sh[0]) if not np.isnan(mid_sh).any() else np.nan
    if not np.isnan(mid_x):
        f["left_arm_cross"]  = float(l_wr[0] - mid_x) if not np.isnan(l_wr).any() else np.nan
        f["right_arm_cross"] = float(mid_x - r_wr[0]) if not np.isnan(r_wr).any() else np.nan
    else:
        f["left_arm_cross"] = f["right_arm_cross"] = np.nan

    # ── Head alignment ────────────────────────────────────────────────────────
    if not any(np.isnan(v).any() for v in [nose, mid_sh, mid_hip]):
        f["head_alignment"] = angle_from_vertical(mid_sh, nose)
    else:
        f["head_alignment"] = np.nan

    # ── Hip vertical position (oscillation tracked over time in stride_analyzer) ─
    f["hip_height"] = float(mid_hip[1]) if not np.isnan(mid_hip).any() else np.nan

    return f


def smooth(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    if len(series) < window or np.all(np.isnan(series)):
        return series
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(series) - (1 if len(series) % 2 == 0 else 0))
    w = max(w, 3)
    filled = np.nan_to_num(series, nan=float(np.nanmean(series)))
    try:
        return savgol_filter(filled, window_length=w, polyorder=2)
    except Exception:
        return series


def angular_velocity(values: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Frame-by-frame angular velocity (degrees/sec)."""
    vel = np.diff(values) * fps
    return np.concatenate([[0], np.clip(vel, -2000, 2000)])


def process_keypoint_csv(
    csv_path: Path,
    form_class: str,
    video_stem: str,
    fps: float = 30.0,
) -> pd.DataFrame:
    """
    Compute biomechanical features for all frames of a single CSV.
    Returns DataFrame with one row per frame.
    """
    df = pd.read_csv(csv_path)

    feat_rows = []
    for _, row in df.iterrows():
        feats = compute_features_frame(row)
        feats["frame"]        = int(row.get("frame", 0))
        feats["timestamp_ms"] = float(row.get("timestamp_ms", 0))
        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows).sort_values("frame").reset_index(drop=True)

    # Smooth angle columns
    angle_cols = [c for c in feat_df.columns if "angle" in c or "position" in c]
    for col in angle_cols:
        feat_df[col] = smooth(feat_df[col].values)

    # Angular velocities
    for col in angle_cols:
        feat_df[f"{col}_vel"] = angular_velocity(feat_df[col].values, fps)

    # Vertical oscillation (std of hip_height per ~stride cycle)
    feat_df["vertical_oscillation"] = (
        feat_df["hip_height"].rolling(window=15, min_periods=5).std().fillna(0)
    )

    # Cadence proxy: ankle vertical velocity magnitude
    if "left_ankle_y" in df.columns:
        ank_y = df["left_ankle_y"].ffill().values
        feat_df["cadence_proxy"] = np.abs(np.gradient(ank_y)) * fps

    feat_df["form_class"] = form_class
    feat_df["video_stem"] = video_stem

    return feat_df


def main(args: argparse.Namespace) -> None:
    keypoints_dir = Path(args.keypoints)
    labels_path   = Path(args.labels)
    output_dir    = Path(args.output)

    if not keypoints_dir.exists():
        logger.error(f"Keypoints dir not found: {keypoints_dir}")
        sys.exit(1)
    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path}")
        sys.exit(1)

    labels_df  = pd.read_csv(labels_path)
    label_map  = dict(zip(labels_df["video_stem"], labels_df["form_class"]))
    csvs       = list(keypoints_dir.rglob("*_norm.csv"))
    logger.info(f"Processing {len(csvs)} keypoint CSVs...")

    all_dfs = []
    for csv_path in tqdm(csvs, desc="Features"):
        stem = csv_path.stem.replace("_norm", "")
        form_class = label_map.get(stem)
        if form_class is None:
            logger.debug(f"No label for: {stem}")
            continue
        try:
            feat_df = process_keypoint_csv(csv_path, form_class, stem)
            all_dfs.append(feat_df)
        except Exception as e:
            logger.warning(f"Failed {csv_path.name}: {e}")

    if not all_dfs:
        logger.error("No features computed. Check labels and keypoints alignment.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "biomech_features.csv"
    combined.to_csv(out_path, index=False)

    logger.info(f"\n✅ Features saved: {out_path}")
    logger.info(f"   Rows     : {len(combined):,}")
    logger.info(f"   Clips    : {combined['video_stem'].nunique()}")
    logger.info(f"   Columns  : {len(combined.columns)}")
    dist = combined.groupby("form_class")["video_stem"].nunique()
    logger.info(f"   Class distribution:\n{dist.to_string()}")
    logger.info("Next: python src/analytics/eda.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer biomechanical features from keypoints")
    parser.add_argument("--keypoints", default="data/processed/keypoints")
    parser.add_argument("--labels",    default="data/annotations/form_labels.csv")
    parser.add_argument("--output",    default="data/processed/features")
    args = parser.parse_args()
    main(args)

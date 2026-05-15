"""
normalize_poses.py
──────────────────
Data Engineering Step 3: Normalize raw keypoints to be camera/height invariant.

Normalization pipeline per frame:
  1. Hip-center: subtract mid-hip → runner is always at origin
  2. Torso-scale: divide by hip-to-shoulder distance (torso length)
  3. Temporal: forward-fill missing frames, drop leading NaN
  4. Handedness: mirror right-to-left if runner faces left in frame

Results are invariant to:
  - Camera zoom / distance
  - Runner height
  - Horizontal position in frame

Usage:
    python src/data_engineering/normalize_poses.py \\
        --input  data/raw/poses \\
        --output data/processed/keypoints
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

META_COLS = ["frame", "timestamp_ms", "pose_detected", "form_class"]


def detect_runner_direction(df: pd.DataFrame) -> str:
    """
    Heuristic: if left ankle x > right ankle x on average,
    runner is likely moving right → standard orientation.
    Otherwise, runner faces left → mirror.
    """
    la_x = df["left_ankle_x"].dropna()
    ra_x = df["right_ankle_x"].dropna()
    if la_x.empty or ra_x.empty:
        return "right"
    return "right" if float(la_x.mean()) > float(ra_x.mean()) else "left"


def mirror_left_to_right(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror all x coordinates and swap left/right landmark names."""
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith("_x")]
    df[x_cols] = 1.0 - df[x_cols]

    swap_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_heel", "right_heel"),
        ("left_foot_index", "right_foot_index"),
    ]
    for left, right in swap_pairs:
        for coord in ["x", "y", "z", "vis"]:
            lc, rc = f"{left}_{coord}", f"{right}_{coord}"
            if lc in df.columns and rc in df.columns:
                df[lc], df[rc] = df[rc].copy(), df[lc].copy()
    return df


def normalize_frame(row: pd.Series) -> pd.Series:
    """Normalize a single frame: hip-center + torso-scale."""
    row = row.copy()

    # Reference points
    hip_x = (row.get("left_hip_x", np.nan) + row.get("right_hip_x", np.nan)) / 2
    hip_y = (row.get("left_hip_y", np.nan) + row.get("right_hip_y", np.nan)) / 2
    hip_z = (row.get("left_hip_z", np.nan) + row.get("right_hip_z", np.nan)) / 2

    sh_x = (row.get("left_shoulder_x", np.nan) + row.get("right_shoulder_x", np.nan)) / 2
    sh_y = (row.get("left_shoulder_y", np.nan) + row.get("right_shoulder_y", np.nan)) / 2
    sh_z = (row.get("left_shoulder_z", np.nan) + row.get("right_shoulder_z", np.nan)) / 2

    torso = np.sqrt((sh_x-hip_x)**2 + (sh_y-hip_y)**2 + (sh_z-hip_z)**2)
    if np.isnan(torso) or torso < 1e-6:
        return row

    for col in [c for c in row.index if c.endswith("_x")]:
        row[col] = (row[col] - hip_x) / torso
    for col in [c for c in row.index if c.endswith("_y")]:
        row[col] = (row[col] - hip_y) / torso
    for col in [c for c in row.index if c.endswith("_z")]:
        row[col] = (row[col] - hip_z) / torso
    return row


def normalize_csv(csv_path: Path, output_path: Path) -> dict:
    """Normalize a single pose CSV and save result."""
    df = pd.read_csv(csv_path)
    original_len = len(df)

    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]

    # Fill missing frames
    df[coord_cols] = df[coord_cols].ffill().bfill()
    df = df.dropna(subset=["nose_x"]).reset_index(drop=True)
    clean_len = len(df)

    # Mirror if runner faces left
    direction = detect_runner_direction(df)
    mirrored = False
    if direction == "left":
        df = mirror_left_to_right(df)
        mirrored = True

    # Normalize each frame
    feat_cols = [c for c in df.columns if c not in META_COLS]
    norm_rows = [normalize_frame(row) for _, row in df.iterrows()]
    df_norm = pd.DataFrame(norm_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(output_path, index=False)

    return {
        "source": csv_path.name,
        "output": output_path.name,
        "original_frames": original_len,
        "clean_frames": clean_len,
        "drop_rate": round(1 - clean_len / max(original_len, 1), 3),
        "mirrored": mirrored,
    }


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    pose_csvs = list(input_dir.rglob("*_poses.csv"))
    if not pose_csvs:
        logger.error(f"No pose CSVs in {input_dir}")
        sys.exit(1)

    logger.info(f"Normalizing {len(pose_csvs)} pose CSVs...")
    results = []
    for csv_path in tqdm(pose_csvs, desc="Normalizing"):
        rel = csv_path.relative_to(input_dir)
        out_path = output_dir / rel.parent / csv_path.name.replace("_poses.csv", "_norm.csv")
        try:
            meta = normalize_csv(csv_path, out_path)
            results.append(meta)
        except Exception as e:
            logger.warning(f"  Failed: {csv_path.name}: {e}")

    avg_drop = sum(r["drop_rate"] for r in results) / max(len(results), 1)
    mirrored = sum(1 for r in results if r["mirrored"])
    logger.info(f"\n✅ Normalized {len(results)} files")
    logger.info(f"   Avg drop rate   : {avg_drop:.1%}")
    logger.info(f"   Mirrored (L→R)  : {mirrored}")
    logger.info("Next: python src/data_engineering/label_clips.py --auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize MediaPipe pose keypoints")
    parser.add_argument("--input",  default="data/raw/poses",          help="Input pose CSV directory")
    parser.add_argument("--output", default="data/processed/keypoints", help="Output directory")
    args = parser.parse_args()
    main(args)

"""
build_sequences.py
──────────────────
Modeling Step 1: Build fixed-length sliding-window sequences for BiLSTM.

Strategy:
  - For each clip, slide a window of seq_len frames with stride step
  - Each window gets the clip's form_class label
  - Saves X_train/val/test.npy + y_train/val/test.npy + label encoder

Usage:
    python src/modeling/build_sequences.py \\
        --features data/processed/features/biomech_features.csv \\
        --output   data/processed/sequences \\
        --seq-len  30 --step 10
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FORM_CLASSES = ["good_form", "overstriding", "forward_lean", "arm_crossing"]

# Features fed into the LSTM at every timestep
SEQUENCE_FEATURES = [
    "trunk_lean_angle",
    "max_overstride",
    "arm_swing_symmetry",
    "hip_drop_angle",
    "knee_drive_angle",
    "stride_angle",
    "rear_knee_angle",
    "front_knee_angle",
    "head_alignment",
    "vertical_oscillation",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_arm_cross",
    "right_arm_cross",
    "cadence_proxy",
    "trunk_lean_angle_vel",
    "knee_drive_angle_vel",
    "arm_swing_symmetry_vel",
]


def extract_windows(
    group: pd.DataFrame,
    seq_len: int,
    step: int,
    features: list[str],
) -> list[np.ndarray]:
    """Slide a window over a clip and return all valid windows."""
    group = group.sort_values("frame").reset_index(drop=True)
    avail = [f for f in features if f in group.columns]
    data  = group[avail].values.astype(np.float32)

    # Fill NaN with column means
    col_means = np.nanmean(data, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    for j in range(data.shape[1]):
        mask = np.isnan(data[:, j])
        data[mask, j] = col_means[j]

    windows = []
    for start in range(0, len(data) - seq_len + 1, step):
        window = data[start : start + seq_len]
        if window.shape[0] == seq_len:
            # Pad to full feature width if some columns missing
            if window.shape[1] < len(features):
                pad = np.zeros((seq_len, len(features) - window.shape[1]), dtype=np.float32)
                window = np.hstack([window, pad])
            windows.append(window)
    return windows


def build_dataset(
    features_df: pd.DataFrame,
    seq_len: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X, y arrays from all clips."""
    X_list, y_list, stems = [], [], []

    for video_stem, group in features_df.groupby("video_stem"):
        form_class = group["form_class"].iloc[0]
        if form_class not in FORM_CLASSES:
            continue

        windows = extract_windows(group, seq_len, step, SEQUENCE_FEATURES)
        for w in windows:
            X_list.append(w)
            y_list.append(form_class)
            stems.append(video_stem)

    if not X_list:
        logger.error("No sequences built. Check feature file and labels.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    logger.info(f"Built {len(X)} windows from {features_df['video_stem'].nunique()} clips")
    return X, y, stems


def main(args: argparse.Namespace) -> None:
    feat_path  = Path(args.features)
    output_dir = Path(args.output)

    if not feat_path.exists():
        logger.error(f"Features not found: {feat_path}")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    logger.info(f"Loaded {len(df):,} rows | {df['video_stem'].nunique()} clips")

    X, y_raw, stems = build_dataset(df, args.seq_len, args.step)

    # Encode labels
    le = LabelEncoder()
    le.fit(FORM_CLASSES)
    y = le.transform(y_raw).astype(np.int64)

    logger.info(f"X shape : {X.shape}  (windows × frames × features)")
    logger.info(f"Classes : {le.classes_.tolist()}")
    for i, cls in enumerate(le.classes_):
        logger.info(f"  {i} = {cls}: {(y==i).sum()} windows")

    # Stratified split 70/15/15
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, stratify=y_tv, random_state=42)

    # Normalize per feature across time axis
    n_tr, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(n_tr, seq_len, n_feat)
    X_val   = scaler.transform(X_val.reshape(-1, n_feat)).reshape(len(X_val), seq_len, n_feat)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(len(X_test), seq_len, n_feat)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "X_test.npy",  X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy",   y_val)
    np.save(output_dir / "y_test.npy",  y_test)

    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (output_dir / "feature_names.txt").write_text("\n".join(SEQUENCE_FEATURES))

    logger.info(f"\n✅ Sequences saved → {output_dir}")
    logger.info(f"   Train : {X_train.shape}")
    logger.info(f"   Val   : {X_val.shape}")
    logger.info(f"   Test  : {X_test.shape}")
    logger.info("Next: python src/modeling/train_classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LSTM sequences from biomech features")
    parser.add_argument("--features", default="data/processed/features/biomech_features.csv")
    parser.add_argument("--output",   default="data/processed/sequences")
    parser.add_argument("--seq-len",  type=int, default=30,
                        help="Frames per window (default: 30 = 1s at 30fps)")
    parser.add_argument("--step",     type=int, default=10,
                        help="Sliding window step (default: 10)")
    args = parser.parse_args()
    main(args)

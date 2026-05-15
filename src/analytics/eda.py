"""
eda.py
──────
Analytics: Exploratory data analysis for the running form dataset.

Generates 6 publication-quality plots:
  01 — Class distribution (bar + pie)
  02 — Key angle distributions per class (boxplots)
  03 — Trunk lean vs overstride scatter (class separation)
  04 — Feature correlation heatmap
  05 — Temporal angle profiles aligned by frame
  06 — Arm-crossing midline analysis

Usage:
    python src/analytics/eda.py \\
        --features data/processed/features/biomech_features.csv \\
        --stride   data/processed/features/stride_metrics.csv \\
        --output   docs/eda_report
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

CLASS_PALETTE = {
    "good_form":    "#2ecc71",
    "overstriding": "#e74c3c",
    "forward_lean": "#f39c12",
    "arm_crossing": "#9b59b6",
}
CLASS_ORDER = ["good_form", "overstriding", "forward_lean", "arm_crossing"]


# ─── PLOT FUNCTIONS ───────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df.groupby("form_class")["video_stem"].nunique().reindex(CLASS_ORDER).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(counts.index, counts.values,
                       color=[CLASS_PALETTE[c] for c in counts.index],
                       edgecolor="white", width=0.55)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(int(val)), ha="center", fontweight="bold")
    axes[0].set_title("Clips per Form Class"); axes[0].set_ylabel("Number of clips")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].pie(counts.values,
                labels=[c.replace("_", "\n") for c in counts.index],
                colors=[CLASS_PALETTE[c] for c in counts.index],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Form Class Proportions")

    plt.suptitle("Dataset Class Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "01_class_distribution.png")
    plt.close()
    logger.info("Saved: 01_class_distribution.png")


def plot_angle_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    features = [
        ("trunk_lean_angle",   "Trunk Lean Angle (°)"),
        ("max_overstride",     "Max Overstride (torso units)"),
        ("arm_swing_symmetry", "Arm Swing Asymmetry (°)"),
        ("hip_drop_angle",     "Hip Drop Angle (°)"),
        ("knee_drive_angle",   "Knee Drive Angle (°)"),
        ("stride_angle",       "Stride Angle (°)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title) in zip(axes.flatten(), features):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[col, "form_class"]].dropna()
        sub = sub[sub["form_class"].isin(CLASS_ORDER)]
        sns.boxplot(data=sub, x="form_class", y=col, order=CLASS_ORDER,
                    palette=CLASS_PALETTE, ax=ax, width=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Key Biomechanical Features by Form Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "02_angle_boxplots.png")
    plt.close()
    logger.info("Saved: 02_angle_boxplots.png")


def plot_trunk_vs_overstride(df: pd.DataFrame, output_dir: Path) -> None:
    if "trunk_lean_angle" not in df.columns or "max_overstride" not in df.columns:
        return

    # Aggregate to per-clip means
    clip_means = df.groupby(["video_stem", "form_class"])[
        ["trunk_lean_angle", "max_overstride", "arm_swing_symmetry"]
    ].mean().reset_index()
    clip_means = clip_means[clip_means["form_class"].isin(CLASS_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trunk lean vs overstride
    for cls in CLASS_ORDER:
        sub = clip_means[clip_means["form_class"] == cls]
        axes[0].scatter(sub["max_overstride"], sub["trunk_lean_angle"],
                        color=CLASS_PALETTE[cls], label=cls, alpha=0.7, s=50)
    axes[0].set_xlabel("Max Overstride (torso units)")
    axes[0].set_ylabel("Trunk Lean Angle (°)")
    axes[0].set_title("Overstride vs Trunk Lean — Class Separation")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Arm symmetry vs trunk lean
    for cls in CLASS_ORDER:
        sub = clip_means[clip_means["form_class"] == cls]
        axes[1].scatter(sub["trunk_lean_angle"], sub["arm_swing_symmetry"],
                        color=CLASS_PALETTE[cls], label=cls, alpha=0.7, s=50)
    axes[1].set_xlabel("Trunk Lean Angle (°)")
    axes[1].set_ylabel("Arm Swing Asymmetry (°)")
    axes[1].set_title("Trunk Lean vs Arm Asymmetry")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.suptitle("Feature Scatter by Form Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "03_class_scatter.png")
    plt.close()
    logger.info("Saved: 03_class_scatter.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    key_cols = [
        "trunk_lean_angle", "max_overstride", "arm_swing_symmetry",
        "hip_drop_angle", "knee_drive_angle", "stride_angle",
        "rear_knee_angle", "front_knee_angle", "head_alignment",
        "vertical_oscillation",
    ]
    available = [c for c in key_cols if c in df.columns]
    if len(available) < 3:
        return

    clip_means = df.groupby("video_stem")[available].mean()
    corr = clip_means.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix (per-clip means)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "04_correlation_heatmap.png")
    plt.close()
    logger.info("Saved: 04_correlation_heatmap.png")


def plot_temporal_profiles(df: pd.DataFrame, output_dir: Path) -> None:
    """Average angle profile over normalized time for each class."""
    if "trunk_lean_angle" not in df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, form_class in zip(axes.flatten(), CLASS_ORDER):
        sub = df[df["form_class"] == form_class]
        color = CLASS_PALETTE[form_class]

        for col, lbl in [
            ("trunk_lean_angle", "Trunk lean"),
            ("knee_drive_angle", "Knee drive"),
            ("arm_swing_symmetry", "Arm asym."),
        ]:
            if col not in sub.columns:
                continue
            profile_data = []
            for _, grp in sub.groupby("video_stem"):
                vals = grp.sort_values("frame")[col].values
                if len(vals) > 5:
                    profile_data.append(np.interp(
                        np.linspace(0, 1, 50),
                        np.linspace(0, 1, len(vals)),
                        np.nan_to_num(vals, nan=np.nanmean(vals)),
                    ))
            if profile_data:
                arr = np.array(profile_data)
                x = np.linspace(0, 1, 50)
                mean, std = arr.mean(0), arr.std(0)
                ax.plot(x, mean, linewidth=2, label=lbl)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)

        ax.set_title(f"{form_class.replace('_', ' ').title()}", fontweight="bold", color=color)
        ax.set_xlabel("Gait cycle (normalized)")
        ax.set_ylabel("Angle (°)")
        ax.legend(fontsize=8)

    plt.suptitle("Temporal Feature Profiles by Form Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "05_temporal_profiles.png")
    plt.close()
    logger.info("Saved: 05_temporal_profiles.png")


def plot_arm_crossing(df: pd.DataFrame, output_dir: Path) -> None:
    """Arm crossing analysis: wrist x relative to body midline."""
    if "left_arm_cross" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for cls in CLASS_ORDER:
        sub = df[df["form_class"] == cls]["left_arm_cross"].dropna()
        if len(sub) > 10:
            sns.kdeplot(sub, ax=ax, color=CLASS_PALETTE[cls], label=cls,
                        fill=True, alpha=0.2, linewidth=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5,
               label="Body midline")
    ax.set_xlabel("Left wrist x relative to midline (torso units)\nPositive = crossing")
    ax.set_ylabel("Density")
    ax.set_title("Arm Crossing Analysis: Wrist Position vs Body Midline",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "06_arm_crossing_analysis.png")
    plt.close()
    logger.info("Saved: 06_arm_crossing_analysis.png")


def save_summary(df: pd.DataFrame, stride_df: pd.DataFrame, output_dir: Path) -> None:
    summary_parts = []

    summary_parts.append("=== DATASET SUMMARY ===")
    summary_parts.append(f"Total frames    : {len(df):,}")
    summary_parts.append(f"Total clips     : {df['video_stem'].nunique()}")

    if "form_class" in df.columns:
        dist = df.groupby("form_class")["video_stem"].nunique()
        summary_parts.append("\nClips per class:")
        for cls in CLASS_ORDER:
            summary_parts.append(f"  {cls:20s}: {dist.get(cls, 0)}")

    if not stride_df.empty:
        summary_parts.append("\n=== STRIDE METRICS BY CLASS ===")
        cols_to_show = ["trunk_lean_mean", "max_overstride_mean",
                        "arm_swing_symmetry_mean", "vertical_oscillation"]
        available = [c for c in cols_to_show if c in stride_df.columns]
        if available:
            tbl = stride_df.groupby("form_class")[available].mean().round(2)
            summary_parts.append(tbl.to_string())

    summary_txt = "\n".join(summary_parts)
    (output_dir / "summary.txt").write_text(summary_txt)
    logger.info("Saved: summary.txt")
    print("\n" + summary_txt)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_path   = Path(args.features)
    stride_path = Path(args.stride)

    if not feat_path.exists():
        logger.error(f"Features not found: {feat_path}")
        logger.info("Run biomech_features.py first.")
        sys.exit(1)

    df         = pd.read_csv(feat_path)
    stride_df  = pd.read_csv(stride_path) if stride_path.exists() else pd.DataFrame()

    logger.info(f"Loaded {len(df):,} rows | {df['video_stem'].nunique()} clips")
    logger.info("Generating EDA plots...")

    plot_class_distribution(df, output_dir)
    plot_angle_boxplots(df, output_dir)
    plot_trunk_vs_overstride(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_temporal_profiles(df, output_dir)
    plot_arm_crossing(df, output_dir)
    save_summary(df, stride_df, output_dir)

    logger.info(f"\n✅ EDA complete → {output_dir}")
    logger.info("Next: python src/modeling/build_sequences.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for running form dataset")
    parser.add_argument("--features", default="data/processed/features/biomech_features.csv")
    parser.add_argument("--stride",   default="data/processed/features/stride_metrics.csv")
    parser.add_argument("--output",   default="docs/eda_report")
    args = parser.parse_args()
    main(args)

"""
form_scorer.py
──────────────
Modeling Step 3: XGBoost form quality scorer (0–100) + corrective feedback engine.

Two components:
  1. FormScorer      — XGBoost regressor predicts a 0-100 quality score
  2. FeedbackEngine  — Rule-based system generates specific coaching cues

Usage:
    python src/modeling/form_scorer.py \\
        --features data/processed/features/biomech_features.csv \\
        --output   models/form_scorer.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FORM_CLASSES = ["good_form", "overstriding", "forward_lean", "arm_crossing"]

SCORER_FEATURES = [
    "trunk_lean_angle",
    "max_overstride",
    "arm_swing_symmetry",
    "hip_drop_angle",
    "knee_drive_angle",
    "stride_angle",
    "vertical_oscillation",
    "left_elbow_angle",
    "right_elbow_angle",
    "head_alignment",
    "left_arm_cross",
    "right_arm_cross",
]

# ─── RULE-BASED FEEDBACK ─────────────────────────────────────────────────────

FEEDBACK_RULES = [
    {
        "feature": "trunk_lean_angle",
        "ideal_low": 3.0, "ideal_high": 12.0,
        "low_msg":  None,
        "high_msg": "You're leaning forward too much ({val:.1f}°). Stand tall — target 3–12°.",
        "weight": 3,
    },
    {
        "feature": "max_overstride",
        "ideal_low": -0.15, "ideal_high": 0.05,
        "low_msg":  None,
        "high_msg": "Overstriding detected — foot landing {val:.2f} units ahead of hip. "
                    "Shorten stride and increase cadence.",
        "weight": 3,
    },
    {
        "feature": "arm_swing_symmetry",
        "ideal_low": 0.0, "ideal_high": 15.0,
        "low_msg":  None,
        "high_msg": "Arm swing asymmetry {val:.1f}°. Keep elbows at 90°, swing fore-aft not across body.",
        "weight": 2,
    },
    {
        "feature": "hip_drop_angle",
        "ideal_low": 0.0, "ideal_high": 5.0,
        "low_msg":  None,
        "high_msg": "Hip drop detected ({val:.1f}°). Strengthen glutes — single-leg exercises recommended.",
        "weight": 2,
    },
    {
        "feature": "knee_drive_angle",
        "ideal_low": 60.0, "ideal_high": 999,
        "low_msg":  "Insufficient knee drive ({val:.1f}°). Lift knees higher for better propulsion.",
        "high_msg": None,
        "weight": 2,
    },
    {
        "feature": "left_arm_cross",
        "ideal_low": -0.05, "ideal_high": 0.05,
        "low_msg":  None,
        "high_msg": "Left arm crossing body midline ({val:.2f} units). Keep arms swinging straight fore-aft.",
        "weight": 2,
    },
    {
        "feature": "right_arm_cross",
        "ideal_low": -0.05, "ideal_high": 0.05,
        "low_msg":  None,
        "high_msg": "Right arm crossing body midline ({val:.2f} units). Keep arms swinging straight fore-aft.",
        "weight": 2,
    },
    {
        "feature": "head_alignment",
        "ideal_low": 0.0, "ideal_high": 10.0,
        "low_msg":  None,
        "high_msg": "Head misalignment detected ({val:.1f}°). Look ahead 10–20m, keep chin level.",
        "weight": 1,
    },
    {
        "feature": "vertical_oscillation",
        "ideal_low": 0.0, "ideal_high": 0.08,
        "low_msg":  None,
        "high_msg": "High vertical oscillation ({val:.3f}). Run more horizontally — push back, not up.",
        "weight": 2,
    },
]


class FeedbackEngine:
    """Rule-based biomechanical feedback for running form."""

    def __init__(self, rules: list[dict] | None = None):
        self.rules = rules or FEEDBACK_RULES

    def generate(self, features: dict) -> list[dict]:
        """
        Generate coaching cues from a dict of feature values.

        Returns list of:
          {feature, value, message, severity, weight}
        """
        feedback = []
        for rule in self.rules:
            feat = rule["feature"]
            val  = features.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            msg = None
            if val < rule["ideal_low"] and rule.get("low_msg"):
                msg      = rule["low_msg"].format(val=val)
                severity = "high" if rule["weight"] >= 3 else "medium"
            elif val > rule["ideal_high"] and rule.get("high_msg"):
                msg      = rule["high_msg"].format(val=val)
                severity = "high" if rule["weight"] >= 3 else "medium"

            if msg:
                feedback.append({
                    "feature":  feat,
                    "value":    round(float(val), 3),
                    "message":  msg,
                    "severity": severity,
                    "weight":   rule["weight"],
                })

        severity_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(feedback, key=lambda x: (severity_order[x["severity"]], -x["weight"]))

    def rule_based_score(self, features: dict) -> float:
        """Simple rule-based form score (0–100) based on deviations."""
        total_w = sum(r["weight"] for r in self.rules)
        penalty = 0.0
        for rule in self.rules:
            val = features.get(rule["feature"])
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            low, high, w = rule["ideal_low"], rule["ideal_high"], rule["weight"]
            if val < low:
                penalty += w * min((low - val) / (abs(low) + 1e-9), 1.0)
            elif val > high:
                penalty += w * min((val - high) / (abs(high) + 1e-9), 1.0)
        score = 100.0 * (1.0 - penalty / max(total_w, 1))
        return max(0.0, min(100.0, score))


# ─── ML FORM SCORER ───────────────────────────────────────────────────────────

def build_score_labels(df: pd.DataFrame) -> pd.Series:
    """Use FeedbackEngine to build regression targets."""
    engine = FeedbackEngine()
    scores = []
    for _, row in df.iterrows():
        feat_dict = {f: row.get(f, np.nan) for f in SCORER_FEATURES}
        scores.append(engine.rule_based_score(feat_dict))
    return pd.Series(scores, index=df.index)


def aggregate_clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame-level features to clip-level stats."""
    avail = [f for f in SCORER_FEATURES if f in df.columns]
    agg   = df.groupby("video_stem")[avail + ["form_class"]].agg(
        lambda x: x.mean() if x.dtype != object else x.iloc[0]
    ).reset_index()
    return agg


def train_form_scorer(
    df: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """Train XGBoost form scorer with 5-fold CV."""
    clip_df  = aggregate_clip_features(df)
    avail    = [f for f in SCORER_FEATURES if f in clip_df.columns]
    X        = clip_df[avail].fillna(0)
    y        = build_score_labels(clip_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )),
    ])

    cv    = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(pipeline, X_train, y_train, cv=cv,
                               scoring="neg_mean_absolute_error", n_jobs=-1)
    logger.info(f"CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "test_mae": round(mean_absolute_error(y_test, y_pred), 3),
        "test_r2":  round(r2_score(y_test, y_pred), 3),
        "cv_mae":   round(cv_mae.mean(), 3),
    }
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v}")
    return pipeline, metrics


def main(args: argparse.Namespace) -> None:
    feat_path   = Path(args.features)
    output_path = Path(args.output)

    if not feat_path.exists():
        logger.error(f"Features not found: {feat_path}")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    mlflow.set_experiment("running-form-scoring")

    with mlflow.start_run(run_name="xgb_form_scorer"):
        pipeline, metrics = train_form_scorer(df)
        mlflow.log_metrics(metrics)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline":        pipeline,
            "feature_cols":    [f for f in SCORER_FEATURES if f in df.columns],
            "feedback_engine": FeedbackEngine(),
        }
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        mlflow.log_artifact(str(output_path), "model")

    logger.info(f"\n✅ Form scorer saved: {output_path}")
    logger.info("Next: python src/modeling/evaluate.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features/biomech_features.csv")
    p.add_argument("--output",   default="models/form_scorer.pkl")
    main(p.parse_args())

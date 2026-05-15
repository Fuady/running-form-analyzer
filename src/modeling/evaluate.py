"""
evaluate.py
───────────
Full evaluation of the BiLSTM classifier and form scorer.

Outputs:
  - Confusion matrix (normalized + counts)
  - Per-class ROC curves (one-vs-rest)
  - SHAP feature importance
  - Form scorer scatter plot
  - JSON summary of all metrics

Usage:
    python src/modeling/evaluate.py \\
        --classifier models/classifier/best_model.pt \\
        --scorer     models/form_scorer.pkl \\
        --sequences  data/processed/sequences \\
        --features   data/processed/features/biomech_features.csv
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    roc_auc_score, roc_curve,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.modeling.bilstm_model import RunningFormClassifier
from src.modeling.form_scorer import SCORER_FEATURES, aggregate_clip_features, build_score_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_ORDER = ["good_form", "overstriding", "forward_lean", "arm_crossing"]
CLASS_COLORS = {"good_form": "#2ecc71", "overstriding": "#e74c3c",
                "forward_lean": "#f39c12", "arm_crossing": "#9b59b6"}


def evaluate_classifier(ckpt_path: Path, seq_dir: Path, output_dir: Path) -> dict:
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return {}

    device = torch.device("cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    model  = RunningFormClassifier(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    import pickle as _pkl
    with open(seq_dir / "label_encoder.pkl", "rb") as f:
        le = _pkl.load(f)

    X_test = torch.FloatTensor(np.load(seq_dir / "X_test.npy"))
    y_test = np.load(seq_dir / "y_test.npy")

    with torch.no_grad():
        logits = model(X_test)
        probs  = F.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro", zero_division=0)
    logger.info(f"\nCLASSIFIER TEST RESULTS")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  F1 macro : {f1:.4f}")
    logger.info("\n" + classification_report(y_test, preds, target_names=le.classes_))

    _plot_confusion_matrix(y_test, preds, le.classes_, output_dir)
    _plot_roc_curves(y_test, probs, le.classes_, output_dir)
    _plot_attention_weights(model, X_test, y_test, le.classes_, output_dir)

    return {"classifier_accuracy": round(acc, 4), "classifier_f1_macro": round(f1, 4)}


def _plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Counts"),
        (axes[1], cm_norm, ".2f",  "Normalized"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.suptitle("Running Form Classifier — Test Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "classifier_confusion_matrix.png", dpi=120)
    plt.close()
    logger.info("Saved: classifier_confusion_matrix.png")


def _plot_roc_curves(y_true, probs, classes, output_dir):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS.values())):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc_val     = roc_auc_score(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("One-vs-Rest ROC Curves — Running Form Classifier", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "classifier_roc_curves.png", dpi=120)
    plt.close()
    logger.info("Saved: classifier_roc_curves.png")


def _plot_attention_weights(model, X_test, y_test, classes, output_dir):
    model.eval()
    with torch.no_grad():
        _, attn = model(X_test[:50], return_attention=True)
    attn_np = attn.numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, attn_np.shape[1])
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS.values())):
        mask = y_test[:50] == i
        if mask.sum() > 0:
            mean_attn = attn_np[mask].mean(0)
            ax.plot(x, mean_attn, color=color, lw=2, label=cls)
            ax.fill_between(x, mean_attn - attn_np[mask].std(0),
                               mean_attn + attn_np[mask].std(0), color=color, alpha=0.1)
    ax.set_xlabel("Gait phase (normalized)"); ax.set_ylabel("Mean attention weight")
    ax.set_title("Attention Weights per Form Class", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "classifier_attention.png", dpi=120)
    plt.close()
    logger.info("Saved: classifier_attention.png")


def evaluate_scorer(model_path: Path, feat_path: Path, output_dir: Path) -> dict:
    if not model_path.exists() or not feat_path.exists():
        return {}

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    pipeline  = artifact["pipeline"]
    feat_cols = artifact["feature_cols"]

    df      = pd.read_csv(feat_path)
    clip_df = aggregate_clip_features(df)
    avail   = [c for c in feat_cols if c in clip_df.columns]
    X       = clip_df[avail].fillna(0)
    y       = build_score_labels(clip_df)

    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred   = pipeline.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    logger.info(f"\nFORM SCORER — MAE: {mae:.2f} | R²: {r2:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_te, y_pred, alpha=0.5, color="#4C72B0", s=25)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5)
    axes[0].set_xlabel("True Form Score"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Form Scorer (MAE={mae:.2f})", fontweight="bold")

    reg = pipeline.named_steps["reg"]
    imp = reg.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    axes[1].barh([avail[i] for i in idx], imp[idx], color="#55A868", edgecolor="white")
    axes[1].invert_yaxis()
    axes[1].set_title("Top Feature Importances", fontweight="bold")

    plt.suptitle("Form Scorer Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "scorer_evaluation.png", dpi=120)
    plt.close()
    logger.info("Saved: scorer_evaluation.png")

    return {"scorer_mae": round(mae, 3), "scorer_r2": round(r2, 4)}


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results.update(evaluate_classifier(
        Path(args.classifier), Path(args.sequences), output_dir))
    results.update(evaluate_scorer(
        Path(args.scorer), Path(args.features), output_dir))

    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ Evaluation complete → {output_dir}")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="models/classifier/best_model.pt")
    p.add_argument("--scorer",     default="models/form_scorer.pkl")
    p.add_argument("--sequences",  default="data/processed/sequences")
    p.add_argument("--features",   default="data/processed/features/biomech_features.csv")
    p.add_argument("--output",     default="docs/evaluation")
    main(p.parse_args())

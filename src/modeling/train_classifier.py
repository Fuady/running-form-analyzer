"""
train_classifier.py
───────────────────
Modeling Step 2: Train the BiLSTM running form classifier.

Features:
  - Cross-entropy loss with class weighting (handles imbalance)
  - AdamW optimizer + ReduceLROnPlateau scheduler
  - Early stopping (patience=15)
  - Full MLflow experiment tracking
  - Checkpoint saving (best val accuracy)
  - Training curve plots

Usage:
    python src/modeling/train_classifier.py \\
        --sequences data/processed/sequences \\
        --output    models/classifier \\
        --epochs 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.modeling.bilstm_model import RunningFormClassifier, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── DATA ─────────────────────────────────────────────────────────────────────

def load_data(seq_dir: Path) -> dict:
    arrays = {}
    for split in ["train", "val", "test"]:
        X = np.load(seq_dir / f"X_{split}.npy")
        y = np.load(seq_dir / f"y_{split}.npy")
        arrays[split] = (torch.FloatTensor(X), torch.LongTensor(y))
    return arrays


def make_loader(X: torch.Tensor, y: torch.Tensor,
                batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def compute_class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = counts.sum() / (n_classes * counts.clamp(min=1))
    return weights


# ─── TRAINING HELPERS ─────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        total      += len(X_b)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits    = model(X_b)
        loss      = criterion(logits, y_b)
        total_loss += loss.item() * len(X_b)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y_b.cpu().tolist())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    seq_dir    = Path(args.sequences)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Device: {device}")

    data   = load_data(seq_dir)
    X_tr, y_tr = data["train"]
    X_vl, y_vl = data["val"]
    X_te, y_te = data["test"]

    n_features = X_tr.shape[2]
    n_classes  = 4
    logger.info(f"Train {X_tr.shape} | Val {X_vl.shape} | Test {X_te.shape}")

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    vl_loader = make_loader(X_vl, y_vl, args.batch_size, shuffle=False)
    te_loader = make_loader(X_te, y_te, args.batch_size, shuffle=False)

    model = RunningFormClassifier(
        input_size=n_features,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=n_classes,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    class_weights = compute_class_weights(y_tr, n_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5, verbose=True)
    early_stop    = EarlyStopping(patience=15)

    mlflow.set_experiment("running-form-classification")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": "BiLSTM",
            "input_size": n_features,
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_train": len(X_tr),
        })

        best_val_acc = 0.0
        best_ckpt    = output_dir / "best_model.pt"
        history      = {"tr_loss": [], "vl_loss": [], "vl_acc": [], "vl_f1": []}

        logger.info(f"\nTraining for up to {args.epochs} epochs...")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
            vl_loss, vl_acc, vl_f1 = eval_epoch(model, vl_loader, criterion, device)
            scheduler.step(vl_acc)

            for k, v in [("tr_loss", tr_loss), ("vl_loss", vl_loss),
                         ("vl_acc",  vl_acc),  ("vl_f1",  vl_f1)]:
                history[k].append(v)

            mlflow.log_metrics({
                "train_loss": round(tr_loss, 4),
                "train_acc":  round(tr_acc,  4),
                "val_loss":   round(vl_loss, 4),
                "val_acc":    round(vl_acc,  4),
                "val_f1":     round(vl_f1,   4),
            }, step=epoch)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "val_acc":      vl_acc,
                    "val_f1":       vl_f1,
                    "input_size":   n_features,
                    "hidden_size":  args.hidden,
                    "num_layers":   args.layers,
                    "dropout":      args.dropout,
                    "num_classes":  n_classes,
                    "seq_len":      X_tr.shape[1],
                }, best_ckpt)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Ep {epoch:4d}/{args.epochs} | "
                    f"tr={tr_loss:.4f}/{tr_acc:.3f} | "
                    f"vl={vl_loss:.4f}/{vl_acc:.3f}/f1={vl_f1:.3f} | "
                    f"best={best_val_acc:.3f} | {time.time()-t0:.1f}s"
                )

            if early_stop(vl_acc):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Test evaluation
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        te_loss, te_acc, te_f1 = eval_epoch(model, te_loader, criterion, device)

        logger.info(f"\nTEST: loss={te_loss:.4f} | acc={te_acc:.4f} | f1={te_f1:.4f}")
        mlflow.log_metrics({"test_acc": round(te_acc, 4), "test_f1": round(te_f1, 4)})
        mlflow.log_artifact(str(best_ckpt), "model")

        _save_curves(history, output_dir)
        mlflow.log_artifact(str(output_dir / "training_curves.png"), "plots")

    logger.info(f"\n✅ Best model: {best_ckpt}")
    logger.info("Next: python src/modeling/form_scorer.py")


def _save_curves(history: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["tr_loss"], label="Train", color="#4C72B0")
    axes[0].plot(history["vl_loss"], label="Val",   color="#C44E52")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["vl_acc"], label="Val Acc", color="#55A868")
    axes[1].plot(history["vl_f1"],  label="Val F1",  color="#8172B2")
    axes[1].set_title("Val Metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.suptitle("Training History — Running Form Classifier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train BiLSTM running form classifier")
    p.add_argument("--sequences", default="data/processed/sequences")
    p.add_argument("--output",    default="models/classifier")
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--hidden",    type=int,   default=128)
    p.add_argument("--layers",    type=int,   default=2)
    p.add_argument("--dropout",   type=float, default=0.3)
    p.add_argument("--batch-size",type=int,   default=32)
    p.add_argument("--lr",        type=float, default=0.001)
    p.add_argument("--cpu",       action="store_true")
    p.add_argument("--run-name",  default="bilstm_v1")
    train(p.parse_args())

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Full end-to-end pipeline for Running Form Analyzer
#
# Usage:
#   bash scripts/run_pipeline.sh               # full pipeline
#   bash scripts/run_pipeline.sh --skip-data   # skip download (use existing videos)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SKIP_DATA=false
for arg in "$@"; do [[ "$arg" == "--skip-data" ]] && SKIP_DATA=true; done

G='\033[0;32m' B='\033[0;34m' Y='\033[1;33m' R='\033[0;31m' NC='\033[0m'
step() { echo -e "\n${B}═══ STEP $1: $2 ${NC}"; }
ok()   { echo -e "${G}✅ $1${NC}"; }
warn() { echo -e "${Y}⚠️  $1${NC}"; }
fail() { echo -e "${R}❌ $1${NC}"; exit 1; }

echo -e "${B}"
echo "╔══════════════════════════════════════════════════╗"
echo "║   Running Form Analyzer — Full Pipeline          ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 0. Environment check ──────────────────────────────────────────────────────
step 0 "Environment Check"
python -c "import cv2, mediapipe, torch, xgboost, fastapi" 2>/dev/null || \
  fail "Missing dependencies. Run: pip install -r requirements.txt"
ok "Dependencies OK"

# ── 1. Download videos ────────────────────────────────────────────────────────
if [[ "$SKIP_DATA" == "false" ]]; then
  step 1 "Download Running Videos (sample mode)"
  python src/data_engineering/download_videos.py \
    --mode sample --max-videos 8 --output data/raw/videos
  ok "Videos downloaded"
else
  warn "Skipping download (--skip-data)"
fi

# ── 2. Extract poses ──────────────────────────────────────────────────────────
step 2 "Extract MediaPipe Poses"
python src/data_engineering/extract_poses.py \
  --input data/raw/videos --output data/raw/poses
ok "Poses extracted → data/raw/poses/"

# ── 3. Normalize poses ────────────────────────────────────────────────────────
step 3 "Normalize Keypoints (hip-center + torso-scale)"
python src/data_engineering/normalize_poses.py \
  --input data/raw/poses --output data/processed/keypoints
ok "Keypoints normalized → data/processed/keypoints/"

# ── 4. Label clips ────────────────────────────────────────────────────────────
step 4 "Auto-label Clips"
python src/data_engineering/label_clips.py \
  --auto --input data/processed/keypoints \
  --output data/annotations/form_labels.csv
ok "Labels saved → data/annotations/form_labels.csv"

# ── 5. Engineer features ──────────────────────────────────────────────────────
step 5 "Engineer Biomechanical Features (18 features per frame)"
python src/analytics/biomech_features.py \
  --keypoints data/processed/keypoints \
  --labels    data/annotations/form_labels.csv \
  --output    data/processed/features
ok "Features saved → data/processed/features/biomech_features.csv"

# ── 6. Stride analysis ────────────────────────────────────────────────────────
step 6 "Compute Stride Metrics"
python src/analytics/stride_analyzer.py \
  --features data/processed/features/biomech_features.csv \
  --output   data/processed/features/stride_metrics.csv
ok "Stride metrics saved"

# ── 7. EDA ────────────────────────────────────────────────────────────────────
step 7 "Exploratory Data Analysis"
python src/analytics/eda.py \
  --features data/processed/features/biomech_features.csv \
  --stride   data/processed/features/stride_metrics.csv \
  --output   docs/eda_report
ok "EDA plots saved → docs/eda_report/"

# ── 8. Build sequences ────────────────────────────────────────────────────────
step 8 "Build LSTM Sequences (30-frame sliding windows)"
python src/modeling/build_sequences.py \
  --features data/processed/features/biomech_features.csv \
  --output   data/processed/sequences \
  --seq-len  30 --step 10
ok "Sequences built → data/processed/sequences/"

# ── 9. Train BiLSTM ───────────────────────────────────────────────────────────
step 9 "Train BiLSTM Classifier"
python src/modeling/train_classifier.py \
  --sequences data/processed/sequences \
  --output    models/classifier \
  --epochs    50 \
  --run-name  bilstm_v1
ok "Classifier trained → models/classifier/"

# ── 10. Train form scorer ─────────────────────────────────────────────────────
step 10 "Train XGBoost Form Scorer"
python src/modeling/form_scorer.py \
  --features data/processed/features/biomech_features.csv \
  --output   models/form_scorer.pkl
ok "Form scorer saved → models/form_scorer.pkl"

# ── 11. Evaluate ─────────────────────────────────────────────────────────────
step 11 "Full Model Evaluation"
python src/modeling/evaluate.py \
  --classifier models/classifier/best_model.pt \
  --scorer     models/form_scorer.pkl \
  --sequences  data/processed/sequences \
  --features   data/processed/features/biomech_features.csv \
  --output     docs/evaluation
ok "Evaluation report → docs/evaluation/"

# ── 12. Tests ─────────────────────────────────────────────────────────────────
step 12 "Run Test Suite"
pytest tests/ -v --tb=short
ok "All tests passed"

echo -e "\n${G}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Pipeline complete! 🏃                               ║"
echo "║                                                      ║"
echo "║  Start all services:                                 ║"
echo "║    docker-compose up --build                         ║"
echo "║                                                      ║"
echo "║  URLs:                                               ║"
echo "║    API docs:   http://localhost:8000/docs            ║"
echo "║    Dashboard:  http://localhost:8501                 ║"
echo "║    MLflow:     http://localhost:5000                 ║"
echo "║    Grafana:    http://localhost:3000                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

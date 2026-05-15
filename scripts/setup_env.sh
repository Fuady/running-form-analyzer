#!/usr/bin/env bash
# setup_env.sh — One-time environment setup
set -euo pipefail

echo "🏃 Setting up Running Form Analyzer environment..."

# Python version check
python_ver=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_ver"

# Virtual environment
if [[ ! -d "venv" ]]; then
  python3 -m venv venv
  echo "✅ Virtual environment created"
fi
source venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt

# Create directory structure
mkdir -p data/raw/{videos/{good_form,overstriding,forward_lean,arm_crossing},poses}
mkdir -p data/processed/{keypoints,features,sequences}
mkdir -p data/annotations models/{classifier} docs/{eda_report,evaluation} outputs

# .gitkeep files for empty dirs
for d in data/raw/videos/good_form data/raw/videos/overstriding \
          data/raw/videos/forward_lean data/raw/videos/arm_crossing \
          data/raw/poses data/processed/keypoints data/processed/features \
          data/processed/sequences data/annotations models/classifier \
          docs/eda_report docs/evaluation outputs; do
  touch "$d/.gitkeep" 2>/dev/null || true
done

[[ ! -f ".env" ]] && cp .env.example .env && echo "✅ .env created"

# Check FFmpeg
if command -v ffmpeg &>/dev/null; then
  echo "✅ FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
  echo "⚠️  FFmpeg not found — install: sudo apt install ffmpeg"
fi

echo ""
echo "✅ Setup complete!"
echo "Next: source venv/bin/activate && bash scripts/run_pipeline.sh"

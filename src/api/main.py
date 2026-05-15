"""
main.py — FastAPI app for Running Form Analyzer.

Endpoints:
  POST /analyze   Upload video → form classification + score + feedback
  GET  /health    Health check
  GET  /docs      Swagger UI (auto-generated)

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.inference import RunningFormAnalyzer
from api.schemas import AnalysisResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "models/classifier/best_model.pt")
SCORER_PATH     = os.getenv("SCORER_PATH",     "models/form_scorer.pkl")
SCALER_PATH     = os.getenv("SCALER_PATH",     "data/processed/sequences/scaler.pkl")
LE_PATH         = os.getenv("LE_PATH",         "data/processed/sequences/label_encoder.pkl")
MAX_SIZE_MB     = int(os.getenv("MAX_VIDEO_SIZE_MB", "200"))
ALLOWED_EXT     = {".mp4", ".avi", ".mov", ".mkv"}

analyzer: RunningFormAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Loading models...")
    try:
        analyzer = RunningFormAnalyzer(
            classifier_path=CLASSIFIER_PATH,
            scorer_path=SCORER_PATH,
            scaler_path=SCALER_PATH,
            le_path=LE_PATH,
        )
        logger.info("✅ Models loaded")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        analyzer = None
    yield


app = FastAPI(
    title="🏃 Running Form Analyzer API",
    description="""
Upload a running video clip and receive:
- **Form classification** (good / overstriding / forward_lean / arm_crossing)
- **Form quality score** (0–100)
- **Specific coaching feedback** per detected biomechanical fault

### How it works
1. MediaPipe extracts 33 body keypoints per frame
2. 18 biomechanical features computed (trunk lean, overstride, arm symmetry, etc.)
3. BiLSTM classifies running form from 30-frame windows
4. XGBoost scores overall form quality
5. Rule engine generates actionable coaching cues

### Example
```bash
curl -X POST http://localhost:8000/analyze -F "video=@my_run.mp4"
```
""",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok" if analyzer else "degraded",
        classifier_loaded=analyzer is not None and analyzer.model    is not None,
        scorer_loaded    =analyzer is not None and analyzer.scorer   is not None,
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Inference"])
async def analyze_video(
    video: UploadFile = File(..., description="Running video (mp4/avi/mov/mkv)")
):
    """Analyze running form: classify fault + score + coaching feedback."""
    if analyzer is None:
        raise HTTPException(503, "Models not loaded. Check server logs.")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported type: {suffix}. Allowed: {ALLOWED_EXT}")

    content = await video.read()
    if len(content) / 1e6 > MAX_SIZE_MB:
        raise HTTPException(413, f"File too large (max {MAX_SIZE_MB}MB)")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0     = time.perf_counter()
        result = analyzer.analyze_video(tmp_path)
        elapsed = time.perf_counter() - t0

        if "error" in result:
            raise HTTPException(422, result["error"])

        logger.info(
            f"Analyzed: {result['video']} | "
            f"class={result['form_classification']['form_class']} | "
            f"{elapsed:.2f}s"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return AnalysisResponse(**result)


@app.get("/", tags=["System"])
async def root():
    return {"message": "Running Form Analyzer API", "docs": "/docs", "health": "/health"}

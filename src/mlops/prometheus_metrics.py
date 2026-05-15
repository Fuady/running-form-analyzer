"""mlops/prometheus_metrics.py — Prometheus metrics for the inference API."""
from prometheus_client import Counter, Histogram, make_asgi_app

REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["form_class", "status"],
)
LATENCY = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
)
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FORM_SCORE = Histogram(
    "form_score_distribution",
    "Distribution of form quality scores",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
DETECTION_RATE = Histogram(
    "pose_detection_rate",
    "Fraction of frames with pose detected",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

metrics_app = make_asgi_app()


def record(
    form_class: str,
    confidence: float,
    form_score: Optional[float],
    detection_rate: float,
    latency: float,
    success: bool,
) -> None:
    from typing import Optional
    status = "success" if success else "error"
    REQUESTS.labels(form_class=form_class, status=status).inc()
    LATENCY.observe(latency)
    CONFIDENCE.observe(confidence)
    DETECTION_RATE.observe(detection_rate)
    if form_score is not None:
        FORM_SCORE.observe(form_score)

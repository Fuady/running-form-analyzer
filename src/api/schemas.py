"""schemas.py — Pydantic request/response models for the API."""
from typing import Optional
from pydantic import BaseModel, Field

FORM_CLASSES = ["good_form", "overstriding", "forward_lean", "arm_crossing"]


class FormClassification(BaseModel):
    form_class:        str = Field(..., description="Detected running form class")
    probabilities:     dict[str, float] = Field(..., description="Per-class softmax probabilities")
    confidence:        Optional[float]  = Field(None, ge=0, le=1)
    attention_weights: Optional[list[float]] = None


class FeedbackItem(BaseModel):
    feature:  str
    value:    float
    message:  str
    severity: str   = Field(..., description="high / medium / low")
    weight:   int


class FormAnalysis(BaseModel):
    form_score:       Optional[float] = Field(None, ge=0, le=100)
    release_features: dict
    feedback:         list[FeedbackItem]
    feedback_count:   int


class AnalysisResponse(BaseModel):
    video:               str
    total_frames:        int
    detected_frames:     int
    detection_rate:      float
    form_classification: FormClassification
    form_analysis:       FormAnalysis


class HealthResponse(BaseModel):
    status:             str
    classifier_loaded:  bool
    scorer_loaded:      bool
    version:            str = "1.0.0"

# API Reference

Base URL: `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

---

## GET /health

**Response 200**
```json
{
  "status": "ok",
  "classifier_loaded": true,
  "scorer_loaded": true,
  "version": "1.0.0"
}
```

---

## POST /analyze

Analyze a running video: classify form fault + score + feedback.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | ✅ | Running clip (mp4/avi/mov/mkv, max 200MB) |

**Response 200**
```json
{
  "video": "my_run.mp4",
  "total_frames": 150,
  "detected_frames": 143,
  "detection_rate": 0.953,
  "form_classification": {
    "form_class": "overstriding",
    "probabilities": {
      "good_form": 0.08,
      "overstriding": 0.74,
      "forward_lean": 0.12,
      "arm_crossing": 0.06
    },
    "confidence": 0.74,
    "attention_weights": [0.03, 0.04, ...]
  },
  "form_analysis": {
    "form_score": 61.5,
    "release_features": {
      "trunk_lean_angle": 14.2,
      "max_overstride": 0.18
    },
    "feedback": [
      {
        "feature": "max_overstride",
        "value": 0.18,
        "message": "Overstriding detected — foot landing 0.18 units ahead of hip. Shorten stride.",
        "severity": "high",
        "weight": 3
      }
    ],
    "feedback_count": 1
  }
}
```

**Errors**

| Code | Reason |
|---|---|
| 400 | Unsupported file extension |
| 413 | File too large |
| 422 | Too few frames with pose detected |
| 500 | Inference error |
| 503 | Models not loaded |

---

## Examples

**cURL**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@my_run.mp4" | python -m json.tool
```

**Python**
```python
import requests

with open("my_run.mp4", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/analyze",
        files={"video": ("my_run.mp4", f, "video/mp4")},
    )

result = resp.json()
print(f"Form class : {result['form_classification']['form_class']}")
print(f"Confidence : {result['form_classification']['confidence']:.0%}")
print(f"Form score : {result['form_analysis']['form_score']:.0f}/100")
for fb in result['form_analysis']['feedback']:
    print(f"  [{fb['severity'].upper()}] {fb['message']}")
```

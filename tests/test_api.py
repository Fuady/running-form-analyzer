"""tests/test_api.py — FastAPI endpoint tests with mocked RunningFormAnalyzer."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

MOCK_RESULT = {
    "video":            "test_run.mp4",
    "total_frames":     150,
    "detected_frames":  143,
    "detection_rate":   0.953,
    "form_classification": {
        "form_class":        "overstriding",
        "probabilities": {
            "good_form":    0.08,
            "overstriding": 0.74,
            "forward_lean": 0.12,
            "arm_crossing": 0.06,
        },
        "confidence":        0.74,
        "attention_weights": [0.033] * 30,
    },
    "form_analysis": {
        "form_score":  61.5,
        "release_features": {
            "trunk_lean_angle":    14.2,
            "max_overstride":       0.18,
            "arm_swing_symmetry":   9.3,
            "hip_drop_angle":       3.1,
        },
        "feedback": [
            {
                "feature":  "max_overstride",
                "value":    0.18,
                "message":  "Overstriding detected — foot landing 0.18 units ahead of hip.",
                "severity": "high",
                "weight":   3,
            },
            {
                "feature":  "trunk_lean_angle",
                "value":    14.2,
                "message":  "You're leaning forward too much (14.2°). Stand tall.",
                "severity": "medium",
                "weight":   3,
            },
        ],
        "feedback_count": 2,
    },
}


@pytest.fixture
def client():
    """Create TestClient with RunningFormAnalyzer class mocked."""
    mock_instance = MagicMock()
    mock_instance.model              = MagicMock()
    mock_instance.scorer             = MagicMock()
    mock_instance.analyze_video.return_value = MOCK_RESULT

    mock_class = MagicMock(return_value=mock_instance)

    with patch("api.main.RunningFormAnalyzer", mock_class):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ─── Health endpoint ──────────────────────────────────────────────────────────

class TestHealth:
    def test_status_200(self, client):
        assert client.get("/health").status_code == 200

    def test_required_fields(self, client):
        data = client.get("/health").json()
        for field in ["status", "classifier_loaded", "scorer_loaded", "version"]:
            assert field in data, f"Missing: {field}"

    def test_version_string(self, client):
        assert isinstance(client.get("/health").json()["version"], str)


# ─── Root endpoint ────────────────────────────────────────────────────────────

class TestRoot:
    def test_status_200(self, client):
        assert client.get("/").status_code == 200

    def test_has_docs_key(self, client):
        assert "docs" in client.get("/").json()


# ─── Analyze endpoint ─────────────────────────────────────────────────────────

class TestAnalyze:
    def _post(self, client, filename="t.mp4", content=b"\x00" * 512):
        return client.post(
            "/analyze",
            files={"video": (filename, content, "video/mp4")},
        )

    def test_valid_mp4_returns_200(self, client):
        r = self._post(client)
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"

    def test_invalid_ext_returns_400(self, client):
        r = client.post("/analyze", files={"video": ("x.exe", b"bad", "application/octet-stream")})
        assert r.status_code == 400

    def test_response_top_level_fields(self, client):
        data = self._post(client).json()
        for field in ["video", "total_frames", "detected_frames",
                      "detection_rate", "form_classification", "form_analysis"]:
            assert field in data, f"Missing: {field}"

    def test_form_classification_fields(self, client):
        pred = self._post(client).json()["form_classification"]
        for field in ["form_class", "probabilities", "confidence"]:
            assert field in pred, f"Missing: {field}"

    def test_form_class_is_valid(self, client):
        valid = {"good_form", "overstriding", "forward_lean", "arm_crossing"}
        cls   = self._post(client).json()["form_classification"]["form_class"]
        assert cls in valid, f"Unexpected class: {cls}"

    def test_probabilities_four_classes(self, client):
        probs = self._post(client).json()["form_classification"]["probabilities"]
        assert len(probs) == 4

    def test_confidence_in_range(self, client):
        conf = self._post(client).json()["form_classification"]["confidence"]
        assert conf is None or 0.0 <= conf <= 1.0

    def test_form_analysis_fields(self, client):
        fa = self._post(client).json()["form_analysis"]
        for field in ["form_score", "feedback", "feedback_count"]:
            assert field in fa, f"Missing: {field}"

    def test_form_score_in_range(self, client):
        score = self._post(client).json()["form_analysis"]["form_score"]
        assert score is None or 0.0 <= score <= 100.0

    def test_feedback_is_list(self, client):
        fb = self._post(client).json()["form_analysis"]["feedback"]
        assert isinstance(fb, list)

    def test_feedback_item_structure(self, client):
        feedback = self._post(client).json()["form_analysis"]["feedback"]
        for item in feedback:
            for field in ["feature", "value", "message", "severity", "weight"]:
                assert field in item, f"Missing feedback field: {field}"

    def test_detection_rate_in_range(self, client):
        rate = self._post(client).json()["detection_rate"]
        assert 0.0 <= rate <= 1.0

    def test_avi_extension_accepted(self, client):
        r = client.post("/analyze", files={"video": ("t.avi", b"\x00" * 512, "video/avi")})
        assert r.status_code == 200

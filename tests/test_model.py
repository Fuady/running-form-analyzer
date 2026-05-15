"""tests/test_model.py — Unit tests for BiLSTM model and FeedbackEngine."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from modeling.bilstm_model import (
    RunningFormClassifier, TemporalAttention, build_model, count_parameters
)
from modeling.form_scorer import FeedbackEngine, FEEDBACK_RULES


# ─── BiLSTM Tests ─────────────────────────────────────────────────────────────

class TestTemporalAttention:
    def test_output_shapes(self):
        attn   = TemporalAttention(hidden_dim=64)
        lstm_out = torch.randn(4, 30, 64)
        ctx, weights = attn(lstm_out)
        assert ctx.shape     == (4, 64)
        assert weights.shape == (4, 30)

    def test_weights_sum_to_one(self):
        attn     = TemporalAttention(hidden_dim=32)
        lstm_out = torch.randn(5, 20, 32)
        _, weights = attn(lstm_out)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(5), atol=1e-5)

    def test_context_correct_dim(self):
        H = 128
        attn = TemporalAttention(hidden_dim=H)
        out  = torch.randn(3, 15, H)
        ctx, _ = attn(out)
        assert ctx.shape[-1] == H


class TestRunningFormClassifier:
    @pytest.fixture
    def model(self):
        return RunningFormClassifier(
            input_size=18, hidden_size=64, num_layers=2,
            num_classes=4, dropout=0.1,
        )

    def test_output_shape(self, model):
        x = torch.randn(4, 30, 18)
        logits = model(x)
        assert logits.shape == (4, 4)

    def test_output_with_attention(self, model):
        x = torch.randn(4, 30, 18)
        logits, attn = model(x, return_attention=True)
        assert logits.shape == (4, 4)
        assert attn.shape   == (4, 30)

    def test_probabilities_sum_to_one(self, model):
        x     = torch.randn(8, 30, 18)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        sums   = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_predict_returns_class_indices(self, model):
        x     = torch.randn(6, 30, 18)
        preds = model.predict(x)
        assert preds.shape == (6,)
        assert all(0 <= p.item() < 4 for p in preds)

    def test_single_sample(self, model):
        x      = torch.randn(1, 30, 18)
        logits = model(x)
        assert logits.shape == (1, 4)

    def test_variable_seq_len(self, model):
        for T in [10, 30, 60]:
            x = torch.randn(2, T, 18)
            out = model(x)
            assert out.shape == (2, 4)

    def test_build_model_factory(self):
        m = build_model(input_size=12, hidden_size=64, num_classes=4)
        assert isinstance(m, RunningFormClassifier)

    def test_parameter_count_positive(self):
        m = RunningFormClassifier(18, 64, 1, 4, 0.0)
        assert count_parameters(m) > 0

    def test_no_nan_output(self, model):
        x      = torch.randn(4, 30, 18)
        logits = model(x)
        assert not torch.isnan(logits).any()


# ─── FeedbackEngine Tests ─────────────────────────────────────────────────────

class TestFeedbackEngine:
    @pytest.fixture
    def engine(self):
        return FeedbackEngine()

    def _good_feats(self):
        return {
            "trunk_lean_angle":    8.0,
            "max_overstride":      0.02,
            "arm_swing_symmetry": 10.0,
            "hip_drop_angle":      3.0,
            "knee_drive_angle":   80.0,
            "left_arm_cross":      0.01,
            "right_arm_cross":     0.01,
            "head_alignment":      5.0,
            "vertical_oscillation": 0.05,
        }

    def test_good_form_no_feedback(self, engine):
        fb = engine.generate(self._good_feats())
        assert len(fb) == 0

    def test_overstriding_triggers_feedback(self, engine):
        feats = self._good_feats()
        feats["max_overstride"] = 0.25   # well above 0.05 threshold
        fb = engine.generate(feats)
        overstride_fb = [f for f in fb if "overstride" in f["feature"].lower() or "overstride" in f["message"].lower()]
        assert len(overstride_fb) > 0

    def test_trunk_lean_triggers_feedback(self, engine):
        feats = self._good_feats()
        feats["trunk_lean_angle"] = 25.0  # above 12° threshold
        fb = engine.generate(feats)
        trunk_fb = [f for f in fb if "trunk" in f["feature"]]
        assert len(trunk_fb) > 0

    def test_arm_crossing_triggers_feedback(self, engine):
        feats = self._good_feats()
        feats["left_arm_cross"] = 0.20   # crossing midline
        fb = engine.generate(feats)
        arm_fb = [f for f in fb if "arm" in f["feature"]]
        assert len(arm_fb) > 0

    def test_feedback_has_required_fields(self, engine):
        feats = self._good_feats()
        feats["max_overstride"] = 0.30
        for item in engine.generate(feats):
            for field in ["feature", "value", "message", "severity", "weight"]:
                assert field in item, f"Missing: {field}"

    def test_severity_is_valid(self, engine):
        feats = self._good_feats()
        feats["max_overstride"] = 0.30
        for item in engine.generate(feats):
            assert item["severity"] in ("high", "medium", "low")

    def test_high_severity_comes_first(self, engine):
        feats = self._good_feats()
        feats["max_overstride"]    = 0.30   # weight=3 → high
        feats["arm_swing_symmetry"] = 25.0  # weight=2 → medium
        fb = engine.generate(feats)
        if len(fb) >= 2:
            sev_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(fb) - 1):
                assert sev_order[fb[i]["severity"]] <= sev_order[fb[i+1]["severity"]]

    def test_score_perfect_form_near_100(self, engine):
        score = engine.rule_based_score(self._good_feats())
        assert score > 85.0

    def test_score_bad_form_lower(self, engine):
        feats = self._good_feats()
        feats["max_overstride"]    = 0.40
        feats["trunk_lean_angle"]  = 30.0
        feats["arm_swing_symmetry"] = 40.0
        score = engine.rule_based_score(feats)
        assert score < 70.0

    def test_score_in_0_to_100(self, engine):
        import random
        for _ in range(20):
            feats = {r["feature"]: random.uniform(-0.5, 180) for r in FEEDBACK_RULES}
            score = engine.rule_based_score(feats)
            assert 0.0 <= score <= 100.0

    def test_missing_features_handled(self, engine):
        fb = engine.generate({})
        assert isinstance(fb, list)
        assert len(fb) == 0

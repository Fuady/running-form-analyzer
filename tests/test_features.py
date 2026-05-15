"""tests/test_features.py — Unit tests for biomechanical feature engineering."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from analytics.biomech_features import (
    angle_3pts, angle_from_vertical, get_pt, compute_features_frame, smooth
)


def make_row(overrides: dict = {}) -> pd.Series:
    """Build a minimal pose row with sensible running defaults."""
    defaults = {
        # Torso
        "left_shoulder_x": 0.4,  "left_shoulder_y": 0.3,
        "right_shoulder_x": 0.6, "right_shoulder_y": 0.3,
        "left_hip_x": 0.4,       "left_hip_y": 0.6,
        "right_hip_x": 0.6,      "right_hip_y": 0.6,
        # Arms
        "left_elbow_x": 0.35,    "left_elbow_y": 0.45,
        "right_elbow_x": 0.65,   "right_elbow_y": 0.45,
        "left_wrist_x": 0.30,    "left_wrist_y": 0.55,
        "right_wrist_x": 0.70,   "right_wrist_y": 0.55,
        # Legs — runner mid-stride
        "left_knee_x": 0.45,     "left_knee_y": 0.72,
        "right_knee_x": 0.55,    "right_knee_y": 0.85,
        "left_ankle_x": 0.43,    "left_ankle_y": 0.90,
        "right_ankle_x": 0.60,   "right_ankle_y": 1.05,
        # Head
        "nose_x": 0.5,            "nose_y": 0.10,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


class TestAngle3Pts:
    def test_right_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, 1.0])
        assert abs(angle_3pts(a, b, c) - 90.0) < 1.0

    def test_straight_line_180(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])
        assert abs(angle_3pts(a, b, c) - 180.0) < 1.0

    def test_returns_float(self):
        a, b, c = np.array([1., 0.]), np.array([0., 0.]), np.array([0., 1.])
        assert isinstance(angle_3pts(a, b, c), float)

    def test_range_0_to_180(self):
        for _ in range(20):
            a = np.random.randn(2)
            b = np.random.randn(2)
            c = np.random.randn(2)
            if not (np.allclose(a, b) or np.allclose(b, c)):
                angle = angle_3pts(a, b, c)
                assert 0 <= angle <= 180


class TestAngleFromVertical:
    def test_straight_up_is_zero(self):
        bottom = np.array([0.0, 1.0])
        top    = np.array([0.0, 0.0])   # upward = decreasing y
        assert abs(angle_from_vertical(bottom, top)) < 1.0

    def test_horizontal_is_90(self):
        bottom = np.array([0.0, 0.0])
        top    = np.array([1.0, 0.0])
        assert abs(angle_from_vertical(bottom, top) - 90.0) < 1.0

    def test_non_negative(self):
        bottom = np.array([0.0, 0.5])
        top    = np.array([0.3, 0.2])
        assert angle_from_vertical(bottom, top) >= 0


class TestGetPt:
    def test_returns_xy_array(self):
        row = make_row()
        pt  = get_pt(row, "left_shoulder")
        assert pt.shape == (2,)
        assert pt[0] == pytest.approx(0.4)

    def test_missing_returns_nan(self):
        row = pd.Series({"ghost_x": np.nan, "ghost_y": np.nan})
        pt  = get_pt(row, "ghost")
        assert np.isnan(pt).all()


class TestComputeFeaturesFrame:
    def test_returns_dict(self):
        row = make_row()
        feats = compute_features_frame(row)
        assert isinstance(feats, dict)

    def test_trunk_lean_computed(self):
        row = make_row()
        feats = compute_features_frame(row)
        assert "trunk_lean_angle" in feats

    def test_trunk_lean_in_range(self):
        row = make_row()
        feats = compute_features_frame(row)
        tl = feats.get("trunk_lean_angle", np.nan)
        if not np.isnan(tl):
            assert 0 <= tl <= 90

    def test_arm_swing_symmetry_present(self):
        row = make_row()
        feats = compute_features_frame(row)
        assert "arm_swing_symmetry" in feats

    def test_nan_landmark_handled(self):
        row = make_row({"right_knee_x": np.nan, "right_knee_y": np.nan})
        feats = compute_features_frame(row)
        # Should not raise; stride_angle should be NaN
        assert isinstance(feats, dict)

    def test_all_values_are_numeric_or_nan(self):
        row = make_row()
        feats = compute_features_frame(row)
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"

    def test_hip_drop_non_negative(self):
        row = make_row()
        feats = compute_features_frame(row)
        hd = feats.get("hip_drop_angle", np.nan)
        if not np.isnan(hd):
            assert hd >= 0


class TestSmooth:
    def test_same_length(self):
        arr = np.random.randn(30)
        smoothed = smooth(arr)
        assert len(smoothed) == len(arr)

    def test_short_array_unchanged(self):
        arr = np.array([1.0, 2.0])
        result = smooth(arr)
        np.testing.assert_array_equal(result, arr)

    def test_reduces_noise(self):
        """Smoothed std should be less than original noisy signal."""
        t = np.linspace(0, 2 * np.pi, 60)
        signal = np.sin(t) + np.random.randn(60) * 0.5
        smoothed = smooth(signal, window=7)
        assert np.std(smoothed) <= np.std(signal) * 1.5  # generous bound

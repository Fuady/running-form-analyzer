"""
inference.py
────────────
End-to-end inference pipeline: video → pose → features → form class + score + feedback.
Used by FastAPI app and can be run standalone.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER = "models/classifier/best_model.pt"
DEFAULT_SCORER     = "models/form_scorer.pkl"
DEFAULT_SCALER     = "data/processed/sequences/scaler.pkl"
DEFAULT_LE         = "data/processed/sequences/label_encoder.pkl"


class RunningFormAnalyzer:
    """Full pipeline: video clip → form class + quality score + coaching feedback."""

    def __init__(
        self,
        classifier_path: str = DEFAULT_CLASSIFIER,
        scorer_path:     str = DEFAULT_SCORER,
        scaler_path:     str = DEFAULT_SCALER,
        le_path:         str = DEFAULT_LE,
        seq_len:         int = 30,
        device: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.device  = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load_classifier(classifier_path)
        self._load_scorer(scorer_path)
        self._load_scaler(scaler_path, le_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_classifier(self, path: str) -> None:
        try:
            sys.path.insert(0, str(Path(__file__).parents[1]))
            from modeling.bilstm_model import RunningFormClassifier
            ckpt = torch.load(path, map_location=self.device)
            self.model = RunningFormClassifier(
                input_size=ckpt["input_size"],
                hidden_size=ckpt["hidden_size"],
                num_layers=ckpt["num_layers"],
                num_classes=ckpt["num_classes"],
                dropout=ckpt["dropout"],
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.seq_len = ckpt.get("seq_len", self.seq_len)
            logger.info(f"Classifier loaded: {path}")
        except Exception as e:
            logger.warning(f"Classifier not loaded: {e}")
            self.model = None

    def _load_scorer(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
            self.scorer          = artifact["pipeline"]
            self.scorer_features = artifact["feature_cols"]
            self.feedback_engine = artifact["feedback_engine"]
            logger.info(f"Scorer loaded: {path}")
        except Exception as e:
            logger.warning(f"Scorer not loaded: {e}")
            self.scorer = self.feedback_engine = None
            self.scorer_features = []

    def _load_scaler(self, scaler_path: str, le_path: str) -> None:
        try:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(le_path, "rb") as f:
                self.le = pickle.load(f)
            logger.info("Scaler + LabelEncoder loaded")
        except Exception as e:
            logger.warning(f"Scaler/LE not loaded: {e}")
            self.scaler = self.le = None

    # ── Inference pipeline ────────────────────────────────────────────────────

    def extract_poses(self, video_path: str) -> pd.DataFrame:
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        records = []
        frame_idx = 0

        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                row = {"frame": frame_idx, "timestamp_ms": frame_idx * 1000 / fps,
                       "pose_detected": int(results.pose_landmarks is not None),
                       "form_class": "unknown"}
                if results.pose_landmarks:
                    for lm_enum in mp_pose.PoseLandmark:
                        lm   = results.pose_landmarks.landmark[lm_enum.value]
                        name = lm_enum.name.lower()
                        row[f"{name}_x"]   = lm.x
                        row[f"{name}_y"]   = lm.y
                        row[f"{name}_z"]   = lm.z
                        row[f"{name}_vis"] = lm.visibility
                records.append(row)
                frame_idx += 1
        cap.release()
        return pd.DataFrame(records)

    def compute_features(self, pose_df: pd.DataFrame) -> pd.DataFrame:
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from analytics.biomech_features import (
            compute_features_frame, smooth, angular_velocity
        )
        rows = []
        for _, row in pose_df.iterrows():
            f = compute_features_frame(row)
            f["frame"]        = int(row["frame"])
            f["timestamp_ms"] = float(row.get("timestamp_ms", 0))
            rows.append(f)
        feat_df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
        angle_cols = [c for c in feat_df.columns if "angle" in c]
        for col in angle_cols:
            feat_df[col] = smooth(feat_df[col].values)
            feat_df[f"{col}_vel"] = angular_velocity(feat_df[col].values)
        feat_df["vertical_oscillation"] = (
            feat_df["hip_height"].rolling(15, min_periods=5).std().fillna(0)
            if "hip_height" in feat_df.columns else 0.0
        )
        feat_df["video_stem"] = "inference"
        feat_df["form_class"] = "unknown"
        return feat_df

    def build_sequence(self, feat_df: pd.DataFrame) -> np.ndarray:
        from modeling.build_sequences import SEQUENCE_FEATURES, extract_windows
        feat_df["video_stem"] = "inference"
        windows = extract_windows(feat_df, self.seq_len, self.seq_len, SEQUENCE_FEATURES)
        if not windows:
            # Single window from available frames
            avail = [f for f in SEQUENCE_FEATURES if f in feat_df.columns]
            data  = feat_df[avail].fillna(0).values.astype(np.float32)
            if len(data) < self.seq_len:
                pad  = np.zeros((self.seq_len - len(data), len(avail)), dtype=np.float32)
                data = np.vstack([pad, data])
            windows = [data[-self.seq_len:]]
        # Average all windows from the clip
        return np.mean(windows, axis=0, keepdims=True)  # (1, seq_len, n_feat)

    def predict_form(self, sequence: np.ndarray) -> dict:
        if self.model is None:
            return {"form_class": "unknown", "probabilities": {}, "confidence": None}

        # Normalize
        if self.scaler:
            flat = sequence.reshape(-1, sequence.shape[-1])
            sequence = self.scaler.transform(flat).reshape(sequence.shape)

        x = torch.FloatTensor(sequence).to(self.device)
        with torch.no_grad():
            logits, attn = self.model(x, return_attention=True)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

        classes = self.le.classes_.tolist() if self.le else [
            "good_form", "overstriding", "forward_lean", "arm_crossing"
        ]
        return {
            "form_class":       classes[pred_idx],
            "probabilities":    {cls: round(float(p), 3) for cls, p in zip(classes, probs)},
            "confidence":       round(float(probs.max()), 3),
            "attention_weights": attn.cpu().numpy()[0].tolist(),
        }

    def score_and_feedback(self, feat_df: pd.DataFrame) -> dict:
        avail_feats = {f: float(feat_df[f].mean())
                       for f in self.scorer_features
                       if f in feat_df.columns and not feat_df[f].isna().all()}

        form_score = None
        if self.scorer:
            X = pd.DataFrame([{f: avail_feats.get(f, 0) for f in self.scorer_features}])
            form_score = round(float(self.scorer.predict(X)[0]), 1)

        feedback = []
        if self.feedback_engine:
            feedback = self.feedback_engine.generate(avail_feats)

        return {
            "form_score":        form_score,
            "release_features":  {k: round(v, 3) for k, v in avail_feats.items()},
            "feedback":          feedback,
            "feedback_count":    len(feedback),
        }

    def analyze_video(self, video_path: str) -> dict:
        """Full pipeline: returns analysis dict."""
        pose_df  = self.extract_poses(video_path)
        n_frames = len(pose_df)
        detected = int(pose_df.get("pose_detected", pd.Series([0])).sum())

        if detected < 5:
            return {"error": f"Only {detected} frames with pose detected. Need at least 5."}

        feat_df  = self.compute_features(pose_df)
        sequence = self.build_sequence(feat_df)
        pred     = self.predict_form(sequence)
        form     = self.score_and_feedback(feat_df)

        return {
            "video":              Path(video_path).name,
            "total_frames":       n_frames,
            "detected_frames":    detected,
            "detection_rate":     round(detected / max(n_frames, 1), 3),
            "form_classification": pred,
            "form_analysis":       form,
        }

"""
extract_poses.py
────────────────
Data Engineering Step 2: Extract 33 body keypoints per frame using MediaPipe Pose.

For each video:
  - Processes every frame with MediaPipe Pose
  - Saves (x, y, z, visibility) × 33 landmarks per frame to CSV
  - Optionally saves annotated skeleton overlay video
  - Generates a JSON manifest with detection rates

Output CSV columns:
  frame, timestamp_ms, form_class,
  [nose_x, nose_y, nose_z, nose_vis, left_shoulder_x, ... ] × 33

Usage:
    python src/data_engineering/extract_poses.py \\
        --input  data/raw/videos \\
        --output data/raw/poses \\
        --save-video
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LANDMARK_NAMES = [lm.name.lower() for lm in mp_pose.PoseLandmark]

# Running-relevant landmarks (subset for quick validation)
KEY_LANDMARKS = {
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
}

# Build CSV header
CSV_HEADER = ["frame", "timestamp_ms", "pose_detected", "form_class"]
for name in LANDMARK_NAMES:
    for coord in ["x", "y", "z", "vis"]:
        CSV_HEADER.append(f"{name}_{coord}")


def extract_poses_from_video(
    video_path: Path,
    output_dir: Path,
    form_class: str,
    save_annotated: bool = False,
    min_detection_conf: float = 0.5,
    min_tracking_conf: float = 0.5,
) -> dict:
    """
    Run MediaPipe Pose on all frames of a video.

    Returns metadata dict with detection stats.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path.name}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (video_path.stem + "_poses.csv")

    # Optional annotated video writer
    writer = None
    if save_annotated:
        ann_path = output_dir / (video_path.stem + "_skeleton.mp4")
        writer = cv2.VideoWriter(
            str(ann_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )

    records: list[dict] = []
    detected = 0
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf,
    ) as pose:
        with tqdm(total=total, desc=f"  {video_path.name[:45]}", unit="fr", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = round(cap.get(cv2.CAP_PROP_POS_MSEC), 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True

                row: dict = {
                    "frame": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "pose_detected": int(results.pose_landmarks is not None),
                    "form_class": form_class,
                }

                if results.pose_landmarks:
                    detected += 1
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        n = LANDMARK_NAMES[i]
                        row[f"{n}_x"]   = round(lm.x, 6)
                        row[f"{n}_y"]   = round(lm.y, 6)
                        row[f"{n}_z"]   = round(lm.z, 6)
                        row[f"{n}_vis"] = round(lm.visibility, 4)
                else:
                    for n in LANDMARK_NAMES:
                        for c in ["x", "y", "z", "vis"]:
                            row[f"{n}_{c}"] = float("nan")

                records.append(row)

                if writer is not None:
                    annotated = frame.copy()
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated, results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                        )
                    cv2.putText(
                        annotated, f"frame={frame_idx} | {form_class}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    writer.write(annotated)

                frame_idx += 1
                pbar.update(1)

    cap.release()
    if writer:
        writer.release()

    # Write CSV
    if records:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(records)

    detection_rate = detected / max(frame_idx, 1)
    logger.info(
        f"  {video_path.name[:50]}: {detected}/{frame_idx} frames detected "
        f"({detection_rate:.1%}) → {csv_path.name}"
    )

    return {
        "video": video_path.name,
        "form_class": form_class,
        "total_frames": frame_idx,
        "detected_frames": detected,
        "detection_rate": round(detection_rate, 3),
        "fps": fps,
        "csv_path": str(csv_path),
    }


def process_directory(
    input_dir: Path,
    output_dir: Path,
    save_annotated: bool,
) -> list[dict]:
    """Walk input_dir, extract poses from all videos."""
    video_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

    if not video_paths:
        logger.error(f"No videos found in {input_dir}")
        logger.info(f"Supported: {SUPPORTED_EXTS}")
        return []

    logger.info(f"Found {len(video_paths)} videos")
    meta_list = []

    for vp in sorted(video_paths):
        # Infer form_class from parent folder name
        form_class = vp.parent.name
        out_dir = output_dir / form_class
        csv_existing = out_dir / (vp.stem + "_poses.csv")

        if csv_existing.exists():
            logger.info(f"  Skipping (exists): {vp.name}")
            continue

        meta = extract_poses_from_video(
            video_path=vp,
            output_dir=out_dir,
            form_class=form_class,
            save_annotated=save_annotated,
        )
        if meta:
            meta_list.append(meta)

    return meta_list


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input not found: {input_dir}")
        sys.exit(1)

    logger.info(f"Pose extraction: {input_dir} → {output_dir}")
    meta_list = process_directory(input_dir, output_dir, args.save_video)

    if not meta_list:
        logger.warning("No new videos processed.")
        return

    manifest_path = output_dir / "poses_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta_list, f, indent=2)

    avg_det = sum(m["detection_rate"] for m in meta_list) / len(meta_list)
    logger.info(f"\n✅ Processed {len(meta_list)} videos | Avg detection: {avg_det:.1%}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info("Next: python src/data_engineering/normalize_poses.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe poses from running videos")
    parser.add_argument("--input",  default="data/raw/videos",  help="Input video directory")
    parser.add_argument("--output", default="data/raw/poses",   help="Output CSV directory")
    parser.add_argument("--save-video", action="store_true",    help="Save annotated skeleton video")
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--min-tracking",  type=float, default=0.5)
    args = parser.parse_args()
    main(args)

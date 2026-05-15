"""
label_clips.py
──────────────
Data Engineering Step 4: Assign form class labels to keypoint CSVs.

Three labeling modes:
  --auto       : Uses parent folder name as label (recommended if downloaded
                 with download_videos.py — folders are already named by class)
  --manual     : Interactive OpenCV viewer (press G/O/F/A/S/Q)
  --synthetic  : Assign labels from folder names for pipeline testing

Output: data/annotations/form_labels.csv
  Columns: video_stem, form_class (good_form / overstriding / forward_lean / arm_crossing)

Usage:
    python src/data_engineering/label_clips.py --auto  --input data/processed/keypoints
    python src/data_engineering/label_clips.py --manual --input data/raw/videos
    python src/data_engineering/label_clips.py --synthetic --input data/processed/keypoints
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_CLASSES = {"good_form", "overstriding", "forward_lean", "arm_crossing"}
KEY_MAP = {
    ord("g"): "good_form",
    ord("o"): "overstriding",
    ord("f"): "forward_lean",
    ord("a"): "arm_crossing",
}
LABEL_COLORS = {
    "good_form":    (50, 205, 50),
    "overstriding": (0, 165, 255),
    "forward_lean": (0, 100, 255),
    "arm_crossing": (180, 0, 255),
}


def auto_label(input_dir: Path) -> list[dict]:
    """
    Use parent folder name as form_class label.
    Works when keypoints/ or videos/ are organized with class subfolders.
    """
    records = []
    for csv_path in sorted(input_dir.rglob("*_norm.csv")):
        form_class = csv_path.parent.name
        if form_class not in VALID_CLASSES:
            logger.debug(f"  Skipping unknown class '{form_class}': {csv_path.name}")
            continue
        records.append({
            "video_stem": csv_path.stem.replace("_norm", ""),
            "csv_path":   str(csv_path),
            "form_class": form_class,
            "label_source": "auto_folder",
        })
        logger.debug(f"  {form_class:20s} ← {csv_path.name}")

    logger.info(f"Auto-labeled {len(records)} clips from folder names")
    _print_class_dist(records)
    return records


def manual_label(input_dir: Path) -> list[dict]:
    """
    Interactive labeler: shows a frame from each video, user presses a key.
    Keys: G=good_form, O=overstriding, F=forward_lean, A=arm_crossing,
          S=skip, Q=quit
    """
    import cv2

    video_paths = [
        p for p in sorted(input_dir.rglob("*"))
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ]

    if not video_paths:
        logger.error(f"No videos found in {input_dir}")
        return []

    logger.info(f"Manual labeling {len(video_paths)} videos")
    logger.info("Keys: [G]=good_form  [O]=overstriding  [F]=forward_lean  [A]=arm_crossing  [S]=skip  [Q]=quit")

    records = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(total * 0.4)))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        disp = cv2.resize(frame, (900, 500))
        cv2.putText(disp, f"{vp.name[:60]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(disp, "[G]=good  [O]=overstride  [F]=lean  [A]=arm_cross  [S]=skip  [Q]=quit",
                    (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Running Form Labeler", disp)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            logger.info("Labeling stopped.")
            break
        if key == ord("s"):
            logger.info(f"  ⏭ Skipped: {vp.name}")
            continue
        form_class = KEY_MAP.get(key)
        if form_class:
            logger.info(f"  [{form_class}] {vp.name}")
            records.append({
                "video_stem": vp.stem,
                "csv_path":   "",
                "form_class": form_class,
                "label_source": "manual",
            })

    cv2.destroyAllWindows()
    _print_class_dist(records)
    return records


def synthetic_label(input_dir: Path) -> list[dict]:
    """Label from folder name — identical to auto but called 'synthetic' in label_source."""
    records = []
    for csv_path in sorted(input_dir.rglob("*_norm.csv")):
        form_class = csv_path.parent.name
        if form_class not in VALID_CLASSES:
            continue
        records.append({
            "video_stem": csv_path.stem.replace("_norm", ""),
            "csv_path":   str(csv_path),
            "form_class": form_class,
            "label_source": "synthetic",
        })
    logger.info(f"Synthetic labels: {len(records)}")
    _print_class_dist(records)
    return records


def _print_class_dist(records: list[dict]) -> None:
    from collections import Counter
    dist = Counter(r["form_class"] for r in records)
    logger.info("Class distribution:")
    for cls in sorted(VALID_CLASSES):
        n = dist.get(cls, 0)
        bar = "█" * n
        logger.info(f"  {cls:20s}: {n:3d}  {bar}")


def save_labels(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        logger.warning("No records to save.")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Labels saved: {output_path} ({len(records)} clips)")


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_path = Path(args.output)

    if args.auto:
        records = auto_label(input_dir)
    elif args.manual:
        records = manual_label(input_dir)
    elif args.synthetic:
        records = synthetic_label(input_dir)
    else:
        logger.error("Provide --auto, --manual, or --synthetic")
        sys.exit(1)

    save_labels(records, output_path)
    logger.info("Next: python src/analytics/biomech_features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label running form clips")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto",      action="store_true", help="Auto-label from folder names")
    group.add_argument("--manual",    action="store_true", help="Interactive labeler (OpenCV)")
    group.add_argument("--synthetic", action="store_true", help="Synthetic labels (same as auto)")
    parser.add_argument("--input",  default="data/processed/keypoints", help="Keypoints directory")
    parser.add_argument("--output", default="data/annotations/form_labels.csv")
    args = parser.parse_args()
    main(args)

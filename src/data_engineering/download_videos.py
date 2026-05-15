"""
download_videos.py
──────────────────
Data Engineering Step 1: Download running form videos from YouTube.

Searches by running form category so videos are pre-organized by label.
Each subfolder (good_form / overstriding / forward_lean / arm_crossing)
becomes the default label for all videos inside it.

Usage:
    python src/data_engineering/download_videos.py --mode sample
    python src/data_engineering/download_videos.py --mode full --max-videos 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Form-specific search queries ──────────────────────────────────────────────
QUERIES: dict[str, list[str]] = {
    "good_form": [
        "perfect running form slow motion biomechanics",
        "elite runner technique side view slow motion",
        "proper running form analysis midfoot strike",
        "efficient running cadence 180spm slow motion",
    ],
    "overstriding": [
        "running overstriding heel strike slow motion",
        "overstriding running form fault analysis",
        "heel striking running injury form correction",
        "running heel strike braking force slow motion",
    ],
    "forward_lean": [
        "running forward lean too much trunk flexion",
        "runner excessive forward lean form fault",
        "running posture forward lean correction slow motion",
        "bad running posture upper body lean forward",
    ],
    "arm_crossing": [
        "running arm crossing midline form fault",
        "runner arms crossing body slow motion analysis",
        "running bad arm swing crossing technique",
        "runner arm swing fault crossing midline correction",
    ],
}

SAMPLE_QUERIES: dict[str, list[str]] = {
    "good_form":    ["perfect running form biomechanics slow motion"],
    "overstriding": ["running overstriding heel strike analysis"],
    "forward_lean": ["running forward lean trunk flexion fault"],
    "arm_crossing": ["running arm crossing midline fault"],
}

FORM_DESCRIPTIONS = {
    "good_form":    "✅ Efficient mechanics — upright posture, symmetric arm swing",
    "overstriding": "⚠️  Foot lands ahead of CoM — braking force, injury risk",
    "forward_lean": "⚠️  Excessive trunk flexion — weak core or fatigue pattern",
    "arm_crossing": "⚠️  Arms cross body midline — energy waste, rotational inefficiency",
}


def build_ydl_opts(output_dir: Path, max_duration: int = 180) -> dict:
    """Build yt-dlp download options."""
    return {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s_%(title).60s.%(ext)s"),
        "noplaylist": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration < {max_duration}"),
        "quiet": False,
        "ignoreerrors": True,
        "writeinfojson": True,
        "retries": 3,
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
    }


def search_and_download(
    query: str,
    output_dir: Path,
    max_videos: int = 5,
    max_duration: int = 180,
) -> list[dict]:
    """Search YouTube and download up to max_videos results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = build_ydl_opts(output_dir, max_duration)
    downloaded = []

    logger.info(f"  Searching: '{query}' (max {max_videos})")
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch{max_videos}:{query}", download=True)
            if info and "entries" in info:
                for entry in (info["entries"] or []):
                    if entry:
                        downloaded.append({
                            "id": entry.get("id"),
                            "title": entry.get("title"),
                            "duration": entry.get("duration"),
                            "url": entry.get("webpage_url"),
                            "query": query,
                        })
        except Exception as e:
            logger.warning(f"  Query failed: {e}")

    return downloaded


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    query_map = SAMPLE_QUERIES if args.mode == "sample" else QUERIES

    if args.form_class:
        if args.form_class not in QUERIES:
            logger.error(f"Unknown form class: {args.form_class}")
            logger.info(f"Available: {list(QUERIES.keys())}")
            sys.exit(1)
        query_map = {args.form_class: QUERIES[args.form_class]}

    all_records: list[dict] = []

    logger.info(f"\nRunning Form Video Downloader")
    logger.info(f"Mode: {args.mode} | Output: {output_dir}")
    logger.info("=" * 60)

    for form_class, queries in query_map.items():
        logger.info(f"\n[{form_class}] {FORM_DESCRIPTIONS.get(form_class, '')}")
        class_dir = output_dir / form_class
        per_query = max(1, args.max_videos // len(queries))

        for query in queries:
            records = search_and_download(
                query=query,
                output_dir=class_dir,
                max_videos=per_query,
                max_duration=args.max_duration,
            )
            for r in records:
                r["form_class"] = form_class
            all_records.extend(records)
            time.sleep(2)  # rate-limit

    # Save manifest
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Downloaded {len(all_records)} videos → {output_dir}")

    # Print breakdown
    from collections import Counter
    counts = Counter(r["form_class"] for r in all_records)
    for cls, cnt in sorted(counts.items()):
        logger.info(f"  {cls:20s}: {cnt} videos")

    logger.info(f"\nManifest saved: {manifest_path}")
    logger.info("Next: python src/data_engineering/extract_poses.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download running form videos from YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick sample (1 video per class for testing)
  python src/data_engineering/download_videos.py --mode sample

  # Full dataset (50 videos per class)
  python src/data_engineering/download_videos.py --mode full --max-videos 50

  # Single class only
  python src/data_engineering/download_videos.py --form-class overstriding --max-videos 20
        """,
    )
    parser.add_argument("--mode", choices=["sample", "full"], default="sample",
                        help="sample = 1 video/query, full = max_videos spread across queries")
    parser.add_argument("--max-videos", type=int, default=8,
                        help="Maximum videos per form class (default: 8)")
    parser.add_argument("--max-duration", type=int, default=180,
                        help="Max video duration in seconds (default: 180)")
    parser.add_argument("--output", type=str, default="data/raw/videos",
                        help="Output directory (default: data/raw/videos)")
    parser.add_argument("--form-class", type=str, default=None,
                        help="Download only this form class")
    args = parser.parse_args()
    main(args)

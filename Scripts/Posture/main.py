"""
main.py — Posture Feature Extraction from Videos

Processes all videos in Video/Normal/ and Video/Anomalous/ folders,
extracts posture features using sliding windows, and saves CSVs.

Usage:
    python main.py              # Process all videos in both folders
    python main.py --normal     # Process only Normal videos
    python main.py --anomalous  # Process only Anomalous videos

Output:
    data/raw/posture_normal.csv
    data/raw/posture_anomalous.csv
"""

import argparse
import csv
import os
import glob

import config
from pose_extractor import PoseExtractor
from posture_analyzer import PostureAnalyzer
from feature_extractor import FeatureExtractor


def find_videos(directory):
    """
    Find all video files in a directory.

    Args:
        directory (str): Path to the video folder.

    Returns:
        list: Sorted list of video file paths.
    """
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(directory, ext)))
    videos.sort()
    return videos


def process_video(video_path, analyzer, feature_extractor):
    """
    Process a single video: extract landmarks, compute values,
    accumulate in the feature extractor.

    Args:
        video_path (str): Path to the video file.
        analyzer (PostureAnalyzer): Posture metric computer.
        feature_extractor (FeatureExtractor): Sliding window feature computer.

    Returns:
        int: Number of valid frames processed.
    """
    extractor = PoseExtractor(video_path)

    if not extractor.start():
        return 0

    valid_frames = 0
    total_frames = 0

    for frame_data in extractor:
        total_frames += 1
        landmarks = frame_data["landmarks"]
        timestamp = frame_data["timestamp"]

        values = analyzer.extract_values(landmarks)

        if values is not None:
            shoulder_asym, head_tilt, fhp = values
            feature_extractor.add_sample(timestamp, shoulder_asym, head_tilt, fhp)
            valid_frames += 1

        # Progress every 500 frames
        if total_frames % 500 == 0:
            pct = (total_frames / extractor.total_frames * 100
                   if extractor.total_frames > 0 else 0)
            print(f"  Frame {total_frames}/{extractor.total_frames} "
                  f"({pct:.0f}%) — {valid_frames} valid")

    extractor.stop()

    print(f"  Done: {valid_frames}/{total_frames} valid frames "
          f"({valid_frames/total_frames*100:.1f}%)")

    return valid_frames


def save_csv(rows, output_path):
    """
    Save feature rows to a CSV file.

    Args:
        rows (list of dict): Feature rows from FeatureExtractor.
        output_path (str): Path to the output CSV.
    """
    if not rows:
        print(f"[WARNING] No rows to save for {output_path}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config.FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} rows to {output_path}")


def process_folder(video_dir, csv_path, label):
    """
    Process all videos in a folder and save a single CSV.

    Args:
        video_dir (str): Path to the video folder.
        csv_path (str): Path to the output CSV.
        label (str): Label for logging ("Normal" or "Anomalous").
    """
    print(f"\n{'='*60}")
    print(f"  Processing {label} videos")
    print(f"  Folder: {video_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(video_dir):
        print(f"[WARNING] Folder not found: {video_dir}")
        print(f"          Create it and add videos, then re-run.")
        return

    videos = find_videos(video_dir)

    if not videos:
        print(f"[WARNING] No video files found in {video_dir}")
        return

    print(f"  Found {len(videos)} video(s)")

    analyzer = PostureAnalyzer()
    all_rows = []

    for video_path in videos:
        video_name = os.path.basename(video_path)
        print(f"\n  --- {video_name} ---")

        feature_ext = FeatureExtractor()
        valid = process_video(video_path, analyzer, feature_ext)

        if valid == 0:
            print(f"  [WARNING] No valid frames in {video_name}, skipping.")
            continue

        rows = feature_ext.compute_windows()
        print(f"  Windows generated: {len(rows)}")

        # Add video source to each row
        for row in rows:
            row["video_source"] = video_name

        all_rows.extend(rows)

    # Save
    save_csv(all_rows, csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract posture features from videos"
    )
    parser.add_argument(
        "--normal", action="store_true",
        help="Process only Normal videos"
    )
    parser.add_argument(
        "--anomalous", action="store_true",
        help="Process only Anomalous videos"
    )
    args = parser.parse_args()

    # If neither flag is set, process both
    do_both = not args.normal and not args.anomalous

    print("=" * 60)
    print("  POSTURE FEATURE EXTRACTION")
    print(f"  Window: {config.WINDOW_SIZE}s | Step: {config.WINDOW_STEP}s")
    print(f"  Features: 9 (mean, std, bad_pct × 3 parameters)")
    print("=" * 60)

    if do_both or args.normal:
        process_folder(
            config.VIDEO_NORMAL_DIR,
            config.CSV_NORMAL_PATH,
            "Normal"
        )

    if do_both or args.anomalous:
        process_folder(
            config.VIDEO_ANOMALOUS_DIR,
            config.CSV_ANOMALOUS_PATH,
            "Anomalous"
        )

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

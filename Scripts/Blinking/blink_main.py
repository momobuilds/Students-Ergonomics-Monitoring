"""
main.py — Blinking Feature Extraction from Videos

Processes all videos in Video/Normal/ and Video/Anomalous/ folders,
extracts blinking features using sliding windows, and saves CSVs.

Usage:
    python main.py              # Process all videos in both folders
    python main.py --normal     # Process only Normal videos
    python main.py --anomalous  # Process only Anomalous videos

Output:
    data/raw/blinking_normal.csv
    data/raw/blinking_anomalous.csv
"""

import argparse
import csv
import os
import glob

import cv2

import blink_config as config
from blink_analyzer import BlinkAnalyzer
from blink_feature_extractor import BlinkFeatureExtractor


def find_videos(directory):
    """Find all video files in a directory."""
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(directory, ext)))
    videos.sort()
    return videos


def process_video(video_path, analyzer, feature_extractor):
    """
    Process a single video: extract EAR per frame, accumulate in
    the feature extractor.

    Args:
        video_path (str): Path to the video file.
        analyzer (BlinkAnalyzer): EAR computer.
        feature_extractor (BlinkFeatureExtractor): Sliding window feature computer.

    Returns:
        int: Number of valid frames (face detected) processed.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"[INFO] Video opened: {video_path}")
    print(f"       FPS: {fps:.1f} | Frames: {total_frames} | "
          f"Duration: {duration:.1f}s")

    valid_frames = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps if fps > 0 else 0

        result = analyzer.analyze(frame)

        if result["face_detected"] and result["ear"] is not None:
            feature_extractor.add_sample(
                timestamp, result["ear"], result["is_closed"]
            )
            valid_frames += 1

        frame_index += 1

        # Progress every 500 frames
        if frame_index % 500 == 0:
            pct = (frame_index / total_frames * 100
                   if total_frames > 0 else 0)
            print(f"  Frame {frame_index}/{total_frames} "
                  f"({pct:.0f}%) — {valid_frames} valid")

    cap.release()

    print(f"  Done: {valid_frames}/{frame_index} valid frames "
          f"({valid_frames/max(frame_index,1)*100:.1f}%)")

    return valid_frames


def save_csv(rows, output_path):
    """Save feature rows to a CSV file."""
    if not rows:
        print(f"[WARNING] No rows to save for {output_path}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config.FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} rows to {output_path}")


def process_folder(video_dir, csv_path, label):
    """Process all videos in a folder and save a single CSV."""
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

    all_rows = []

    for video_path in videos:
        video_name = os.path.basename(video_path)
        print(f"\n  --- {video_name} ---")

        analyzer = BlinkAnalyzer()
        feature_ext = BlinkFeatureExtractor()

        valid = process_video(video_path, analyzer, feature_ext)
        analyzer.close()

        if valid == 0:
            print(f"  [WARNING] No valid frames in {video_name}, skipping.")
            continue

        rows = feature_ext.compute_windows()
        print(f"  Windows generated: {len(rows)}")

        # Add video source
        for row in rows:
            row["video_source"] = video_name

        all_rows.extend(rows)

    save_csv(all_rows, csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract blinking features from videos"
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

    do_both = not args.normal and not args.anomalous

    print("=" * 60)
    print("  BLINKING FEATURE EXTRACTION")
    print(f"  Window: {config.WINDOW_SIZE}s | Step: {config.WINDOW_STEP}s")
    print(f"  Features: 7 (EAR mean/std, PERCLOS, blink count/freq, IBI mean/std)")
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

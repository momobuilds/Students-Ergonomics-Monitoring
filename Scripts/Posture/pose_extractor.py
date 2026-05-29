"""
pose_extractor.py — Video Reading and Pose Landmark Extraction

Reads frames from a video file and extracts 33 body landmarks
using MediaPipe Pose.

Returns raw landmark data — does NOT perform any posture analysis.

MediaPipe Landmark Indices (relevant for posture):
    0  = nose
    7  = left ear
    8  = right ear
    11 = left shoulder
    12 = right shoulder
    23 = left hip
    24 = right hip
"""

import cv2
import mediapipe as mp
import config


class PoseExtractor:
    """
    Extracts pose landmarks from video frames using MediaPipe.

    Usage:
        extractor = PoseExtractor("path/to/video.mp4")
        extractor.start()

        for frame_data in extractor:
            landmarks = frame_data["landmarks"]
            timestamp = frame_data["timestamp"]
            ...

        extractor.stop()
    """

    def __init__(self, video_path):
        """
        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=config.MP_STATIC_IMAGE_MODE,
            model_complexity=config.MP_MODEL_COMPLEXITY,
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
        )

        self.cap = None
        self.fps = None
        self.total_frames = None
        self.duration = None

    def start(self):
        """
        Open the video file.

        Returns:
            bool: True if the video opened successfully.
        """
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"[INFO] Video opened: {self.video_path}")
        print(f"       FPS: {self.fps:.1f} | Frames: {self.total_frames} | "
              f"Duration: {self.duration:.1f}s")

        return True

    def stop(self):
        """Release video and MediaPipe resources."""
        if self.cap is not None:
            self.cap.release()
        self.pose.close()

    def __iter__(self):
        """
        Iterate over all frames in the video.

        Yields:
            dict: {
                "landmarks": list of 33 dicts or None,
                "timestamp": float (seconds from start of video),
                "frame_index": int,
            }
        """
        if self.cap is None or not self.cap.isOpened():
            return

        frame_index = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_index / self.fps if self.fps > 0 else 0

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            landmarks = self._extract_landmarks(results)

            yield {
                "landmarks": landmarks,
                "timestamp": timestamp,
                "frame_index": frame_index,
            }

            frame_index += 1

    def _extract_landmarks(self, results):
        """
        Convert MediaPipe results into a list of landmark dicts.

        Returns:
            list or None: 33 landmarks with (x, y, z, visibility), or None.
        """
        if results.pose_landmarks is None:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            })

        return landmarks

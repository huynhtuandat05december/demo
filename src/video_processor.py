"""Video processing utilities for extracting frames."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image

from src import config


class VideoProcessor:
    """Handles video frame extraction and preprocessing."""

    def __init__(
        self,
        frame_sample_rate: float = config.FRAME_SAMPLE_RATE,
        max_frames: int = config.MAX_FRAMES_PER_VIDEO,
        use_mid_frame_only: bool = config.USE_MID_FRAME_ONLY,
    ):
        """
        Initialize video processor.

        Args:
            frame_sample_rate: Frames to extract per second
            max_frames: Maximum number of frames to extract
            use_mid_frame_only: If True, only extract the middle frame
        """
        self.frame_sample_rate = frame_sample_rate
        self.max_frames = max_frames
        self.use_mid_frame_only = use_mid_frame_only
        self._frame_cache = {}

    def extract_frames(self, video_path: Path) -> List[Image.Image]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            List of PIL Images extracted from the video
        """
        # Check cache if enabled
        if config.CACHE_FRAMES and str(video_path) in self._frame_cache:
            return self._frame_cache[str(video_path)]

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frames = []

            if self.use_mid_frame_only:
                # Extract only the middle frame
                mid_frame_idx = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(self._convert_frame(frame))
            else:
                # Calculate frame sampling interval
                frame_interval = int(fps / self.frame_sample_rate) if self.frame_sample_rate > 0 else 1
                frame_interval = max(1, frame_interval)

                frame_idx = 0
                while len(frames) < self.max_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frames.append(self._convert_frame(frame))
                    frame_idx += frame_interval

                    if frame_idx >= total_frames:
                        break

            # Cache frames if enabled
            if config.CACHE_FRAMES:
                self._frame_cache[str(video_path)] = frames

            return frames

        finally:
            cap.release()

    def _convert_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Convert OpenCV frame (BGR) to PIL Image (RGB).

        Args:
            frame: OpenCV frame in BGR format

        Returns:
            PIL Image in RGB format
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def extract_frames_at_timestamps(
        self, video_path: Path, timestamps: List[float]
    ) -> List[Image.Image]:
        """
        Extract frames at specific timestamps from a video.

        Args:
            video_path: Path to the video file
            timestamps: List of timestamps in seconds

        Returns:
            List of PIL Images at the specified timestamps
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []

            for timestamp in timestamps:
                # Convert timestamp to frame number
                frame_num = int(timestamp * fps)

                # Seek to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    frames.append(self._convert_frame(frame))
                else:
                    print(f"Warning: Could not extract frame at {timestamp}s")

            return frames

        finally:
            cap.release()

    def clear_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached videos."""
        return len(self._frame_cache)

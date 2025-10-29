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

    def get_video_duration(self, video_path: Path) -> float:
        """
        Get the duration of a video in seconds.

        Args:
            video_path: Path to the video file

        Returns:
            Duration in seconds
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            return duration
        finally:
            cap.release()

    def extract_frames_adaptive(
        self,
        video_path: Path,
        min_frames: int = 6,
        max_frames: int = 12
    ) -> List[Image.Image]:
        """
        Extract frames adaptively based on video duration.

        For traffic dashcam videos (5-15 seconds):
        - 5-7 seconds → 6-8 frames
        - 8-12 seconds → 8-10 frames
        - 13-15 seconds → 10-12 frames

        Frames are uniformly distributed across the video duration.

        Args:
            video_path: Path to the video file
            min_frames: Minimum number of frames to extract
            max_frames: Maximum number of frames to extract

        Returns:
            List of PIL Images uniformly sampled from the video
        """
        # Check cache if enabled
        cache_key = f"{video_path}_adaptive_{min_frames}_{max_frames}"
        if config.CACHE_FRAMES and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Determine optimal frame count based on duration
            if duration <= 7:
                num_frames = min(max(min_frames, 6), 8)
            elif duration <= 12:
                num_frames = min(max(min_frames, 8), 10)
            else:
                num_frames = min(max(min_frames, 10), max_frames)

            # Calculate frame indices uniformly distributed
            if num_frames >= total_frames:
                # If requested frames >= total frames, use all frames
                frame_indices = list(range(total_frames))
            else:
                # Uniform sampling
                frame_indices = [
                    int(i * total_frames / num_frames)
                    for i in range(num_frames)
                ]

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(self._convert_frame(frame))
                else:
                    print(f"Warning: Could not extract frame at index {frame_idx}")

            # Cache frames if enabled
            if config.CACHE_FRAMES:
                self._frame_cache[cache_key] = frames

            return frames

        finally:
            cap.release()

    def extract_frames_with_support(
        self,
        video_path: Path,
        support_frames: Optional[List[float]] = None,
        context_window: float = 0.5,
        min_frames: int = 6,
        max_frames: int = 12
    ) -> List[Image.Image]:
        """
        Extract frames using support_frames timestamps with context.

        Strategy:
        1. Extract frames at support_frames timestamps
        2. Extract context frames around each support frame (±context_window seconds)
        3. Fill remaining slots with uniform sampling
        4. Ensure total frames is between min_frames and max_frames

        Args:
            video_path: Path to the video file
            support_frames: List of important timestamps in seconds (from dataset)
            context_window: Seconds before/after support frames to extract (default: 0.5s)
            min_frames: Minimum total frames to extract
            max_frames: Maximum total frames to extract

        Returns:
            List of PIL Images with support frames + context + uniform sampling
        """
        # If no support frames provided, fall back to adaptive extraction
        if not support_frames or len(support_frames) == 0:
            return self.extract_frames_adaptive(video_path, min_frames, max_frames)

        # Check cache if enabled
        cache_key = f"{video_path}_support_{support_frames}_{context_window}_{min_frames}_{max_frames}"
        if config.CACHE_FRAMES and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Collect all timestamps to extract
            timestamps_to_extract = set()

            # Add support frames and context
            for support_time in support_frames:
                # Clip to video bounds
                support_time = max(0, min(support_time, duration))

                # Add the support frame itself
                timestamps_to_extract.add(support_time)

                # Add context frames
                before = max(0, support_time - context_window)
                after = min(duration, support_time + context_window)

                timestamps_to_extract.add(before)
                timestamps_to_extract.add(after)

            # Determine how many more frames we need
            current_count = len(timestamps_to_extract)

            if current_count < min_frames:
                # Need more frames - add uniform sampling
                additional_needed = min_frames - current_count

                # Get uniform frame indices
                uniform_timestamps = [
                    i * duration / additional_needed
                    for i in range(additional_needed)
                ]
                timestamps_to_extract.update(uniform_timestamps)

            # Limit to max_frames
            timestamps_list = sorted(list(timestamps_to_extract))[:max_frames]

            # Extract frames at calculated timestamps
            frames = []
            for timestamp in timestamps_list:
                frame_num = int(timestamp * fps)
                frame_num = min(frame_num, total_frames - 1)  # Ensure within bounds

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    frames.append(self._convert_frame(frame))
                else:
                    print(f"Warning: Could not extract frame at {timestamp}s")

            # If we got fewer frames than min_frames, fill with uniform sampling
            if len(frames) < min_frames:
                print(f"Warning: Only extracted {len(frames)} frames, filling to {min_frames}")
                additional_frames = self.extract_frames_adaptive(video_path, min_frames, min_frames)
                frames = additional_frames[:min_frames]

            # Cache frames if enabled
            if config.CACHE_FRAMES:
                self._frame_cache[cache_key] = frames

            return frames

        finally:
            cap.release()

    def clear_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached videos."""
        return len(self._frame_cache)

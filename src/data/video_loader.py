"""Video data loader for Sea Surface Temperature prediction.

This module provides utilities to load, process, and convert SST video data
into training-ready sequences for ConvLSTM models.

Author: Raaj Eshwar S
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SSTVideoLoader:
    """Load and preprocess SST video data for time series prediction.
    
    This class handles loading SST videos, extracting frames, converting
    color-coded frames to temperature values, and creating sequences for
    training ConvLSTM models.
    
    Args:
        video_path: Path to SST video file
        reference_image_path: Path to color reference/legend image (optional)
        resize_shape: Target frame size (height, width). Default: (64, 64)
        color_mode: Color space for processing ("rgb", "hsv", "grayscale")
        temp_range: Temperature range for mapping (min_temp, max_temp). Default: (0, 100)
    
    Example:
        >>> loader = SSTVideoLoader(
        ...     video_path="data/sst_video.mp4",
        ...     resize_shape=(64, 64)
        ... )
        >>> frames = loader.load_frames()
        >>> temp_frames = loader.frames_to_temperature(frames)
        >>> X, y = loader.create_sequences(temp_frames, sequence_length=10)
    """
    
    def __init__(
        self,
        video_path: str,
        reference_image_path: Optional[str] = None,
        resize_shape: Tuple[int, int] = (64, 64),
        color_mode: str = "hsv",
        temp_range: Tuple[float, float] = (0.0, 100.0)
    ):
        self.video_path = video_path
        self.reference_image_path = reference_image_path
        self.resize_shape = resize_shape
        self.color_mode = color_mode.lower()
        self.temp_range = temp_range
        
        # Validate inputs
        self._validate_inputs()
        
        # Store reference image if provided
        self.reference_image = None
        if reference_image_path:
            self.reference_image = self._load_reference_image()
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        # Check video exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        # Check color mode
        valid_modes = ["rgb", "hsv", "grayscale"]
        if self.color_mode not in valid_modes:
            raise ValueError(f"color_mode must be one of {valid_modes}")
        
        # Check resize shape
        if len(self.resize_shape) != 2:
            raise ValueError("resize_shape must be (height, width)")
        if any(s <= 0 for s in self.resize_shape):
            raise ValueError("resize_shape dimensions must be positive")
    
    def _load_reference_image(self) -> np.ndarray:
        """Load and preprocess reference color legend image.
        
        Returns:
            Preprocessed reference image
        """
        if not os.path.exists(self.reference_image_path):
            raise FileNotFoundError(
                f"Reference image not found: {self.reference_image_path}"
            )
        
        ref_img = cv2.imread(self.reference_image_path)
        if ref_img is None:
            raise ValueError(
                f"Could not load reference image: {self.reference_image_path}"
            )
        
        # Convert to HSV for color matching
        ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
        logger.info(f"Loaded reference image: {self.reference_image_path}")
        
        return ref_hsv
    
    def load_frames(
        self,
        max_frames: Optional[int] = None,
        skip_frames: int = 1,
        progress_bar: bool = True
    ) -> np.ndarray:
        """Extract frames from video.
        
        Args:
            max_frames: Maximum number of frames to extract (None for all)
            skip_frames: Skip every N frames (1 = no skipping, 2 = every other frame)
            progress_bar: Show progress bar during extraction
        
        Returns:
            Array of frames with shape (num_frames, height, width, channels)
            
        Raises:
            ValueError: If video cannot be opened
        """
        logger.info(f"Loading frames from {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(
            f"Video info: {total_frames} frames, {fps} FPS, "
            f"extracting every {skip_frames} frame(s)"
        )
        
        frames = []
        frame_idx = 0
        frames_extracted = 0
        
        # Setup progress bar
        pbar = None
        if progress_bar:
            max_to_extract = max_frames if max_frames else total_frames // skip_frames
            pbar = tqdm(total=max_to_extract, desc="Extracting frames")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue
                
                # Process frame
                processed_frame = self._process_frame(frame)
                frames.append(processed_frame)
                frames_extracted += 1
                
                if pbar:
                    pbar.update(1)
                
                # Check max frames limit
                if max_frames and frames_extracted >= max_frames:
                    break
                
                frame_idx += 1
        
        finally:
            cap.release()
            if pbar:
                pbar.close()
        
        frames_array = np.array(frames)
        logger.info(
            f"Extracted {len(frames)} frames with shape {frames_array.shape}"
        )
        
        return frames_array
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame.
        
        Args:
            frame: Raw frame from video (BGR format)
        
        Returns:
            Processed frame
        """
        # Resize
        frame = cv2.resize(frame, self.resize_shape)
        
        # Convert color space
        if self.color_mode == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.color_mode == "grayscale":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Add channel dimension for consistency
            frame = frame[..., np.newaxis]
        # rgb/bgr stays as-is
        
        return frame
    
    def frames_to_temperature(
        self,
        frames: np.ndarray,
        method: str = "hue_mapping"
    ) -> np.ndarray:
        """Convert color-coded frames to temperature values.
        
        Different methods for mapping frame colors to temperature:
        - "hue_mapping": Use HSV hue channel (0-179) mapped to temp range
        - "linear": Simple linear mapping of pixel intensity
        - "reference": Use reference image for color matching (if provided)
        
        Args:
            frames: Array of frames (num_frames, H, W, C)
            method: Conversion method to use
        
        Returns:
            Temperature array (num_frames, H, W)
        """
        logger.info(f"Converting frames to temperature using method: {method}")
        
        if method == "hue_mapping":
            return self._hue_to_temperature(frames)
        elif method == "linear":
            return self._linear_to_temperature(frames)
        elif method == "reference":
            if self.reference_image is None:
                raise ValueError("reference method requires reference_image_path")
            return self._reference_to_temperature(frames)
        else:
            raise ValueError(f"Unknown conversion method: {method}")
    
    def _hue_to_temperature(self, frames: np.ndarray) -> np.ndarray:
        """Convert HSV hue values to temperature.
        
        Assumes frames are in HSV format where hue channel (0-179)
        represents temperature range.
        """
        if self.color_mode != "hsv":
            raise ValueError("hue_mapping requires color_mode='hsv'")
        
        # Extract hue channel (first channel in HSV)
        hue = frames[:, :, :, 0]
        
        # Map hue (0-179) to temperature range
        min_temp, max_temp = self.temp_range
        temp_frames = (hue / 179.0) * (max_temp - min_temp) + min_temp
        
        return temp_frames
    
    def _linear_to_temperature(self, frames: np.ndarray) -> np.ndarray:
        """Convert frames using simple linear mapping."""
        # Use first channel or grayscale
        if frames.shape[-1] == 1:
            intensity = frames[:, :, :, 0]
        else:
            # Average across channels
            intensity = frames.mean(axis=-1)
        
        # Normalize to 0-1
        intensity_norm = intensity / 255.0
        
        # Map to temperature range
        min_temp, max_temp = self.temp_range
        temp_frames = intensity_norm * (max_temp - min_temp) + min_temp
        
        return temp_frames
    
    def _reference_to_temperature(self, frames: np.ndarray) -> np.ndarray:
        """Convert frames using reference color map."""
        # TODO: Implement color matching against reference image
        # This is more complex - requires building a color-to-temp lookup table
        raise NotImplementedError(
            "Reference-based conversion not yet implemented. "
            "Use 'hue_mapping' or 'linear' instead."
        )
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int = 10,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences for training.
        
        Creates overlapping sequences where each X contains `sequence_length`
        frames, and y is the next frame.
        
        Args:
            data: Temperature data (num_frames, H, W)
            sequence_length: Number of frames in each sequence
            stride: Step size between sequences (1 = maximum overlap)
        
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences (num_sequences, seq_length, H, W)
            - y: Target frames (num_sequences, H, W)
        
        Example:
            >>> data.shape  # (100, 64, 64)
            >>> X, y = loader.create_sequences(data, sequence_length=10, stride=1)
            >>> X.shape  # (90, 10, 64, 64)
            >>> y.shape  # (90, 64, 64)
        """
        logger.info(
            f"Creating sequences: length={sequence_length}, stride={stride}"
        )
        
        num_frames = len(data)
        sequences = []
        targets = []
        
        for i in range(0, num_frames - sequence_length, stride):
            # Input sequence
            seq = data[i:i + sequence_length]
            # Target (next frame)
            target = data[i + sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        logger.info(f"Created {len(X)} sequences: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def normalize_data(
        self,
        data: np.ndarray,
        method: str = "minmax"
    ) -> Tuple[np.ndarray, dict]:
        """Normalize temperature data.
        
        Args:
            data: Temperature data to normalize
            method: Normalization method ("minmax" or "zscore")
        
        Returns:
            Tuple of (normalized_data, normalization_params)
            normalization_params can be used to denormalize later
        """
        if method == "minmax":
            min_val = data.min()
            max_val = data.max()
            normalized = (data - min_val) / (max_val - min_val)
            params = {"method": "minmax", "min": min_val, "max": max_val}
        
        elif method == "zscore":
            mean_val = data.mean()
            std_val = data.std()
            normalized = (data - mean_val) / std_val
            params = {"method": "zscore", "mean": mean_val, "std": std_val}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalized data using {method}: {params}")
        
        return normalized, params
    
    def denormalize_data(
        self,
        normalized_data: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """Reverse normalization to get original scale.
        
        Args:
            normalized_data: Normalized data
            params: Normalization parameters from normalize_data()
        
        Returns:
            Denormalized data
        """
        method = params["method"]
        
        if method == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            return normalized_data * (max_val - min_val) + min_val
        
        elif method == "zscore":
            mean_val = params["mean"]
            std_val = params["std"]
            return normalized_data * std_val + mean_val
        
        else:
            raise ValueError(f"Unknown method in params: {method}")
    
    def get_video_info(self) -> dict:
        """Get information about the video file.
        
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        info = {
            "path": self.video_path,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / 
                               int(cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        
        return info


def load_multiple_videos(
    video_paths: List[str],
    loader_config: dict,
    combine: bool = True
) -> Union[List[np.ndarray], np.ndarray]:
    """Load and process multiple SST videos.
    
    Useful for loading data from multiple time periods or regions.
    
    Args:
        video_paths: List of paths to video files
        loader_config: Configuration dict for SSTVideoLoader
        combine: If True, concatenate all frames into single array
    
    Returns:
        Combined array or list of arrays (one per video)
    
    Example:
        >>> video_paths = ["2020_sst.mp4", "2021_sst.mp4", "2022_sst.mp4"]
        >>> config = {"resize_shape": (64, 64), "color_mode": "hsv"}
        >>> all_frames = load_multiple_videos(video_paths, config)
    """
    all_frames = []
    
    for video_path in tqdm(video_paths, desc="Loading videos"):
        loader = SSTVideoLoader(video_path=video_path, **loader_config)
        frames = loader.load_frames(progress_bar=False)
        temp_frames = loader.frames_to_temperature(frames)
        all_frames.append(temp_frames)
    
    if combine:
        return np.concatenate(all_frames, axis=0)
    else:
        return all_frames


if __name__ == "__main__":
    # Example usage
    print("Testing SSTVideoLoader...")
    
    # NOTE: Update these paths to your actual files
    video_path = "data/sst_video.mp4"
    
    if os.path.exists(video_path):
        # Create loader
        loader = SSTVideoLoader(
            video_path=video_path,
            resize_shape=(64, 64),
            color_mode="hsv",
            temp_range=(0, 100)
        )
        
        # Get video info
        info = loader.get_video_info()
        print(f"\nVideo Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Load frames
        frames = loader.load_frames(max_frames=100)
        print(f"\nExtracted frames: {frames.shape}")
        
        # Convert to temperature
        temp_frames = loader.frames_to_temperature(frames, method="hue_mapping")
        print(f"Temperature frames: {temp_frames.shape}")
        print(f"Temp range: {temp_frames.min():.2f}°C to {temp_frames.max():.2f}°C")
        
        # Create sequences
        X, y = loader.create_sequences(temp_frames, sequence_length=10)
        print(f"\nSequences created:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        
        # Normalize
        X_norm, params = loader.normalize_data(X)
        print(f"\nNormalized X: range [{X_norm.min():.3f}, {X_norm.max():.3f}]")
        
        print("\n✓ Video loader tested successfully!")
    else:
        print(f"\nTest skipped - video not found at: {video_path}")
        print("Update the video_path variable to test with your data.")

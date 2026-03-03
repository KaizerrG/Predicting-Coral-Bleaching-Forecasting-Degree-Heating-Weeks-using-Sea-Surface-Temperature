"""Tests for SSTVideoLoader.

Run with: pytest tests/test_video_loader.py -v
"""

import numpy as np
import pytest
import tempfile
import cv2
import os

from src.data.video_loader import SSTVideoLoader, load_multiple_videos


# Fixtures

@pytest.fixture
def dummy_video():
    """Create a temporary dummy video for testing."""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    video_path = temp_file.name
    temp_file.close()
    
    # Create a simple video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (128, 128))
    
    # Write 30 frames
    for i in range(30):
        # Create frame with gradient (simulating temperature variation)
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 6) % 180  # Varying hue
        frame[:, :, 1] = 255  # Full saturation
        frame[:, :, 2] = 255  # Full value
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture
def video_loader(dummy_video):
    """Create a SSTVideoLoader instance."""
    return SSTVideoLoader(
        video_path=dummy_video,
        resize_shape=(64, 64),
        color_mode="hsv",
        temp_range=(0, 100)
    )


# Tests

class TestSSTVideoLoaderInit:
    """Test initialization and validation."""
    
    def test_initialization_success(self, dummy_video):
        """Test successful initialization."""
        loader = SSTVideoLoader(
            video_path=dummy_video,
            resize_shape=(64, 64)
        )
        assert loader.video_path == dummy_video
        assert loader.resize_shape == (64, 64)
        assert loader.color_mode == "hsv"
    
    def test_invalid_video_path(self):
        """Test error when video doesn't exist."""
        with pytest.raises(FileNotFoundError):
            SSTVideoLoader(video_path="nonexistent_video.mp4")
    
    def test_invalid_color_mode(self, dummy_video):
        """Test error with invalid color mode."""
        with pytest.raises(ValueError, match="color_mode"):
            SSTVideoLoader(
                video_path=dummy_video,
                color_mode="invalid_mode"
            )
    
    def test_invalid_resize_shape(self, dummy_video):
        """Test error with invalid resize shape."""
        with pytest.raises(ValueError):
            SSTVideoLoader(
                video_path=dummy_video,
                resize_shape=(64, -1)  # Negative dimension
            )


class TestLoadFrames:
    """Test frame loading functionality."""
    
    def test_load_all_frames(self, video_loader):
        """Test loading all frames from video."""
        frames = video_loader.load_frames(progress_bar=False)
        
        assert isinstance(frames, np.ndarray)
        assert len(frames) == 30  # We created 30 frames
        assert frames.shape[1:] == (64, 64, 3)  # Resized to 64x64, HSV (3 channels)
    
    def test_load_max_frames(self, video_loader):
        """Test loading limited number of frames."""
        frames = video_loader.load_frames(max_frames=10, progress_bar=False)
        
        assert len(frames) == 10
    
    def test_skip_frames(self, video_loader):
        """Test frame skipping."""
        frames = video_loader.load_frames(skip_frames=2, progress_bar=False)
        
        # Should get roughly half the frames
        assert len(frames) == 15
    
    def test_frame_shape(self, video_loader):
        """Test that frames have correct shape."""
        frames = video_loader.load_frames(max_frames=5, progress_bar=False)
        
        assert frames.shape == (5, 64, 64, 3)


class TestFramesToTemperature:
    """Test temperature conversion."""
    
    def test_hue_mapping(self, video_loader):
        """Test HSV hue to temperature conversion."""
        frames = video_loader.load_frames(max_frames=10, progress_bar=False)
        temp_frames = video_loader.frames_to_temperature(frames, method="hue_mapping")
        
        assert isinstance(temp_frames, np.ndarray)
        assert temp_frames.shape == (10, 64, 64)  # Lost color channel
        assert temp_frames.min() >= 0
        assert temp_frames.max() <= 100
    
    def test_linear_mapping(self, video_loader):
        """Test linear intensity to temperature conversion."""
        frames = video_loader.load_frames(max_frames=10, progress_bar=False)
        temp_frames = video_loader.frames_to_temperature(frames, method="linear")
        
        assert temp_frames.shape == (10, 64, 64)
        assert temp_frames.min() >= 0
        assert temp_frames.max() <= 100
    
    def test_invalid_method(self, video_loader):
        """Test error with invalid conversion method."""
        frames = video_loader.load_frames(max_frames=5, progress_bar=False)
        
        with pytest.raises(ValueError, match="Unknown conversion method"):
            video_loader.frames_to_temperature(frames, method="invalid")
    
    def test_temperature_range(self, dummy_video):
        """Test custom temperature range."""
        loader = SSTVideoLoader(
            video_path=dummy_video,
            temp_range=(20, 30)  # Custom range
        )
        frames = loader.load_frames(max_frames=5, progress_bar=False)
        temp_frames = loader.frames_to_temperature(frames, method="hue_mapping")
        
        assert temp_frames.min() >= 20
        assert temp_frames.max() <= 30


class TestCreateSequences:
    """Test sequence creation."""
    
    def test_basic_sequence_creation(self, video_loader):
        """Test creating sequences with default parameters."""
        frames = video_loader.load_frames(max_frames=20, progress_bar=False)
        temp_frames = video_loader.frames_to_temperature(frames)
        
        X, y = video_loader.create_sequences(temp_frames, sequence_length=5)
        
        # Should have 15 sequences (20 - 5)
        assert X.shape == (15, 5, 64, 64)
        assert y.shape == (15, 64, 64)
    
    def test_sequence_stride(self, video_loader):
        """Test sequence creation with stride."""
        frames = video_loader.load_frames(max_frames=20, progress_bar=False)
        temp_frames = video_loader.frames_to_temperature(frames)
        
        X, y = video_loader.create_sequences(temp_frames, sequence_length=5, stride=2)
        
        # With stride=2, should have fewer sequences
        assert X.shape[0] == 8  # (20 - 5) // 2 + 1
        assert X.shape[1:] == (5, 64, 64)
    
    def test_sequence_content(self, video_loader):
        """Test that sequences contain correct frames."""
        # Create simple test data
        test_data = np.arange(10 * 4 * 4).reshape(10, 4, 4)
        
        X, y = video_loader.create_sequences(test_data, sequence_length=3)
        
        # Check first sequence
        assert np.array_equal(X[0], test_data[0:3])
        assert np.array_equal(y[0], test_data[3])
        
        # Check second sequence
        assert np.array_equal(X[1], test_data[1:4])
        assert np.array_equal(y[1], test_data[4])


class TestNormalization:
    """Test data normalization."""
    
    def test_minmax_normalization(self, video_loader):
        """Test min-max normalization."""
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        normalized, params = video_loader.normalize_data(data, method="minmax")
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert params["method"] == "minmax"
        assert params["min"] == 1
        assert params["max"] == 8
    
    def test_zscore_normalization(self, video_loader):
        """Test z-score normalization."""
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        normalized, params = video_loader.normalize_data(data, method="zscore")
        
        assert np.abs(normalized.mean()) < 1e-10  # Mean should be ~0
        assert np.abs(normalized.std() - 1.0) < 1e-10  # Std should be ~1
        assert params["method"] == "zscore"
    
    def test_denormalization(self, video_loader):
        """Test denormalization restores original data."""
        original_data = np.random.randn(5, 10, 10) * 20 + 50
        
        # Normalize
        normalized, params = video_loader.normalize_data(original_data, method="minmax")
        
        # Denormalize
        denormalized = video_loader.denormalize_data(normalized, params)
        
        # Should match original
        np.testing.assert_array_almost_equal(original_data, denormalized)


class TestVideoInfo:
    """Test video information retrieval."""
    
    def test_get_video_info(self, video_loader):
        """Test retrieving video metadata."""
        info = video_loader.get_video_info()
        
        assert "total_frames" in info
        assert "fps" in info
        assert "width" in info
        assert "height" in info
        assert "duration_seconds" in info
        
        assert info["total_frames"] == 30
        assert info["width"] == 128
        assert info["height"] == 128


class TestMultipleVideos:
    """Test loading multiple videos."""
    
    def test_load_multiple_videos_combined(self, dummy_video):
        """Test loading and combining multiple videos."""
        video_paths = [dummy_video, dummy_video]  # Use same video twice
        loader_config = {"resize_shape": (32, 32), "color_mode": "hsv"}
        
        combined = load_multiple_videos(video_paths, loader_config, combine=True)
        
        # Should have 60 frames total (30 from each video)
        assert combined.shape == (60, 32, 32)
    
    def test_load_multiple_videos_separate(self, dummy_video):
        """Test loading multiple videos separately."""
        video_paths = [dummy_video, dummy_video]
        loader_config = {"resize_shape": (32, 32), "color_mode": "hsv"}
        
        separate = load_multiple_videos(video_paths, loader_config, combine=False)
        
        assert isinstance(separate, list)
        assert len(separate) == 2
        assert all(frames.shape == (30, 32, 32) for frames in separate)


# Integration Tests

def test_full_pipeline(video_loader):
    """Test complete data loading pipeline."""
    # Load frames
    frames = video_loader.load_frames(max_frames=20, progress_bar=False)
    assert frames.shape == (20, 64, 64, 3)
    
    # Convert to temperature
    temp_frames = video_loader.frames_to_temperature(frames)
    assert temp_frames.shape == (20, 64, 64)
    
    # Create sequences
    X, y = video_loader.create_sequences(temp_frames, sequence_length=5)
    assert X.shape == (15, 5, 64, 64)
    assert y.shape == (15, 64, 64)
    
    # Normalize
    X_norm, params = video_loader.normalize_data(X)
    assert X_norm.min() >= 0.0
    assert X_norm.max() <= 1.0
    
    # Denormalize
    X_denorm = video_loader.denormalize_data(X_norm, params)
    np.testing.assert_array_almost_equal(X, X_denorm)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

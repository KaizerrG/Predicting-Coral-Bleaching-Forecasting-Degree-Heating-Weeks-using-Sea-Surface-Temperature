"""Example usage of SSTVideoLoader.

This script demonstrates how to use the video loader to prepare
SST data for ConvLSTM training.

Usage:
    python examples/load_sst_data.py --video data/sst_video.mp4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.video_loader import SSTVideoLoader


def visualize_frames(frames, temp_frames, num_samples=4):
    """Visualize original frames and temperature conversions."""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        idx = i * (len(frames) // num_samples)
        
        # Original frame
        axes[0, i].imshow(frames[idx])
        axes[0, i].set_title(f"Frame {idx}")
        axes[0, i].axis('off')
        
        # Temperature frame
        im = axes[1, i].imshow(temp_frames[idx], cmap='hot')
        axes[1, i].set_title(f"Temperature {idx}")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig('outputs/frame_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/frame_visualization.png")


def visualize_sequences(X, y, num_samples=2):
    """Visualize training sequences."""
    fig, axes = plt.subplots(num_samples, 12, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        # Show the sequence (first 10 frames)
        for t in range(10):
            axes[sample_idx, t].imshow(X[sample_idx, t], cmap='hot', vmin=0, vmax=100)
            axes[sample_idx, t].set_title(f"t={t}")
            axes[sample_idx, t].axis('off')
        
        # Show the target (frame at t+1)
        axes[sample_idx, 10].imshow(y[sample_idx], cmap='hot', vmin=0, vmax=100)
        axes[sample_idx, 10].set_title("Target")
        axes[sample_idx, 10].axis('off')
        
        # Show difference
        if sample_idx < len(X) - 1:
            diff = y[sample_idx] - X[sample_idx, -1]
            im = axes[sample_idx, 11].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10)
            axes[sample_idx, 11].set_title("Change")
            axes[sample_idx, 11].axis('off')
            plt.colorbar(im, ax=axes[sample_idx, 11])
    
    plt.tight_layout()
    plt.savefig('outputs/sequence_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved sequences to outputs/sequence_visualization.png")


def print_data_statistics(temp_frames):
    """Print statistics about the temperature data."""
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Shape: {temp_frames.shape}")
    print(f"Temperature range: {temp_frames.min():.2f}°C to {temp_frames.max():.2f}°C")
    print(f"Mean temperature: {temp_frames.mean():.2f}°C")
    print(f"Std deviation: {temp_frames.std():.2f}°C")
    print(f"Data type: {temp_frames.dtype}")
    print(f"Memory usage: {temp_frames.nbytes / 1024 / 1024:.2f} MB")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Load and process SST video data")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to SST video file"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to load (default: 100)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Length of training sequences (default: 10)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Resize frames to (height, width) (default: 64 64)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("SST VIDEO DATA LOADER - EXAMPLE")
    print("="*60)
    
    # Step 1: Initialize loader
    print(f"\n1. Initializing video loader...")
    print(f"   Video: {args.video}")
    print(f"   Resize: {tuple(args.resize)}")
    
    loader = SSTVideoLoader(
        video_path=args.video,
        resize_shape=tuple(args.resize),
        color_mode="hsv",
        temp_range=(0, 100)
    )
    
    # Show video info
    info = loader.get_video_info()
    print(f"\n   Video Info:")
    print(f"     Total frames: {info['total_frames']}")
    print(f"     FPS: {info['fps']}")
    print(f"     Duration: {info['duration_seconds']:.1f} seconds")
    print(f"     Resolution: {info['width']}x{info['height']}")
    
    # Step 2: Load frames
    print(f"\n2. Loading frames (max {args.max_frames})...")
    frames = loader.load_frames(max_frames=args.max_frames)
    print(f"   Loaded {len(frames)} frames")
    print(f"   Frame shape: {frames.shape}")
    
    # Step 3: Convert to temperature
    print(f"\n3. Converting frames to temperature...")
    temp_frames = loader.frames_to_temperature(frames, method="hue_mapping")
    print(f"   Temperature data shape: {temp_frames.shape}")
    print_data_statistics(temp_frames)
    
    # Step 4: Create sequences
    print(f"\n4. Creating training sequences...")
    print(f"   Sequence length: {args.sequence_length}")
    X, y = loader.create_sequences(temp_frames, sequence_length=args.sequence_length)
    print(f"   Input sequences (X): {X.shape}")
    print(f"   Target frames (y): {y.shape}")
    
    # Step 5: Normalize
    print(f"\n5. Normalizing data...")
    X_norm, params = loader.normalize_data(X, method="minmax")
    y_norm, _ = loader.normalize_data(y, method="minmax")
    print(f"   Normalized X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"   Normalized y range: [{y_norm.min():.3f}, {y_norm.max():.3f}]")
    print(f"   Normalization params: {params}")
    
    # Step 6: Visualize
    print(f"\n6. Creating visualizations...")
    visualize_frames(frames, temp_frames, num_samples=4)
    visualize_sequences(X, y, num_samples=2)
    
    # Step 7: Save processed data
    print(f"\n7. Saving processed data...")
    np.save("outputs/X_train.npy", X_norm)
    np.save("outputs/y_train.npy", y_norm)
    np.save("outputs/normalization_params.npy", params)
    print(f"   ✓ Saved to outputs/")
    
    print("\n" + "="*60)
    print("COMPLETE! Data ready for training.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Use X_train.npy and y_train.npy to train ConvLSTM model")
    print("  2. Check visualizations in outputs/ folder")
    print("  3. Adjust parameters and rerun if needed")
    print()


if __name__ == "__main__":
    main()

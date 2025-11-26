#!/usr/bin/env python3
"""
Convert batch session structure to augmentation-compatible format

Batch session structure:
  video_XXX_YYY/frame_ZZZZ/original.png, mask.png

Augmentation-compatible structure:
  rgb/frame_XXXXYYYY_ZZZZ.png
  masks/frame_XXXXYYYY_ZZZZ.png
"""

import sys
from pathlib import Path
import shutil
import json

def convert_batch_session(session_dir: Path, output_dir: Path = None):
    """
    Convert batch session to augmentation-compatible format

    Args:
        session_dir: Path to batch session directory
        output_dir: Output directory (default: session_dir with _augmentable suffix)
    """
    session_dir = Path(session_dir)

    # Check if it's a batch session
    metadata_file = session_dir / "session_metadata.json"
    if not metadata_file.exists():
        print(f"❌ No session_metadata.json found in {session_dir}")
        return False

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    if metadata.get('session_type') != 'batch':
        print(f"❌ Session is not a batch type: {metadata.get('session_type')}")
        return False

    # Set output directory
    if output_dir is None:
        output_dir = session_dir.parent / f"{session_dir.name}_augmentable"
    output_dir = Path(output_dir)

    # Create output directories
    rgb_out = output_dir / "rgb"
    masks_out = output_dir / "masks"
    rgb_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    print(f"Converting batch session: {session_dir.name}")
    print(f"Output directory: {output_dir}")
    print(f"Total videos: {metadata['total_videos']}")
    print(f"Total frames: {metadata['total_frames']}")

    # Find all video directories
    video_dirs = sorted([d for d in session_dir.iterdir() if d.is_dir() and d.name.startswith('video_')])

    total_copied = 0

    for video_dir in video_dirs:
        video_idx = video_dir.name  # e.g., "video_000_0"

        # Find all frame directories
        frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

        for frame_dir in frame_dirs:
            frame_num = frame_dir.name.split('_')[1]  # e.g., "0000"

            original_file = frame_dir / "original.png"
            mask_file = frame_dir / "mask.png"

            if not original_file.exists() or not mask_file.exists():
                print(f"⚠️  Skipping {frame_dir} (missing files)")
                continue

            # Create consistent naming: videoXXX_frameYYYY.png
            output_name = f"{video_idx}_frame{frame_num}.png"

            # Copy files
            shutil.copy2(original_file, rgb_out / output_name)
            shutil.copy2(mask_file, masks_out / output_name)

            total_copied += 1

            if total_copied % 500 == 0:
                print(f"  Copied {total_copied} frames...")

    # Create session metadata for augmentation system
    aug_metadata = {
        "session_id": f"{session_dir.name}_augmentable",
        "session_type": "interactive",  # Compatible with augmentation system
        "timestamp": metadata['timestamp'],
        "source_session": str(session_dir),
        "total_frames": total_copied,
        "converted_from_batch": True,
        "original_metadata": metadata
    }

    with open(output_dir / "session_metadata.json", 'w') as f:
        json.dump(aug_metadata, f, indent=2)

    print(f"\n✅ Conversion complete!")
    print(f"  Total frames copied: {total_copied}")
    print(f"  RGB directory: {rgb_out}")
    print(f"  Masks directory: {masks_out}")
    print(f"  Output session: {output_dir}")
    print(f"\nYou can now load this session in Data Augmentation tab:")
    print(f"  {output_dir}")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_batch_session_for_augmentation.py <session_dir> [output_dir]")
        print("\nExample:")
        print("  python convert_batch_session_for_augmentation.py outputs/sessions/mouse_batch_20251125_185700")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    convert_batch_session(session_dir, output_dir)

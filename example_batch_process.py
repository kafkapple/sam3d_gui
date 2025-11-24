#!/usr/bin/env python3
"""
Example: Batch Process Videos to Find Motion Segments
This script demonstrates how to use the SAM3DProcessor API for batch processing
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sam3d_processor import SAM3DProcessor


def find_motion_in_video(
    video_path: str,
    segment_duration: float = 3.0,
    motion_threshold: float = 50.0,
    output_base_dir: str = "outputs/batch"
):
    """
    Scan a video for motion segments

    Args:
        video_path: Path to video file
        segment_duration: Length of each segment to test (seconds)
        motion_threshold: Minimum displacement for motion (pixels)
        output_base_dir: Base directory for outputs

    Returns:
        List of (start_time, end_time) tuples with motion
    """
    processor = SAM3DProcessor()

    # Get video info
    print(f"\nProcessing: {video_path}")
    video_info = processor.get_video_info(video_path)
    print(f"Duration: {video_info['duration']:.1f}s, FPS: {video_info['fps']:.1f}")

    # Calculate number of segments
    num_segments = int(video_info['duration'] / segment_duration)
    print(f"Testing {num_segments} segments of {segment_duration}s each...")

    motion_segments = []

    # Process each segment
    for i in range(num_segments):
        start_time = i * segment_duration

        print(f"\n  Segment {i+1}/{num_segments} (t={start_time:.1f}s)...", end=" ")

        try:
            # Process segment (without 3D reconstruction for speed)
            result, _ = processor.process_video_segment(
                video_path=video_path,
                start_time=start_time,
                duration=segment_duration,
                output_dir=None,  # Skip 3D reconstruction
                motion_threshold=motion_threshold
            )

            if result.motion_detected:
                motion_segments.append((start_time, start_time + segment_duration))
                print("✅ MOTION")
            else:
                print("❌ No motion")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n✅ Found {len(motion_segments)} segments with motion:")
    for start, end in motion_segments:
        print(f"   - {start:.1f}s to {end:.1f}s")

    return motion_segments


def process_motion_segments_3d(
    video_path: str,
    motion_segments: list,
    output_dir: str = "outputs/batch"
):
    """
    Perform 3D reconstruction on motion segments

    Args:
        video_path: Path to video file
        motion_segments: List of (start_time, end_time) tuples
        output_dir: Output directory for 3D meshes
    """
    processor = SAM3DProcessor()

    print(f"\nPerforming 3D reconstruction on {len(motion_segments)} segments...")

    for idx, (start, end) in enumerate(motion_segments):
        segment_dir = os.path.join(output_dir, f"segment_{idx:03d}")
        os.makedirs(segment_dir, exist_ok=True)

        print(f"\nSegment {idx+1}/{len(motion_segments)} (t={start:.1f}s-{end:.1f}s)")

        try:
            result, reconstruction = processor.process_video_segment(
                video_path=video_path,
                start_time=start,
                duration=end - start,
                output_dir=segment_dir,
                motion_threshold=0.0  # Already know there's motion
            )

            if reconstruction:
                print(f"   ✅ 3D mesh saved to {segment_dir}/")
            else:
                print(f"   ❌ 3D reconstruction skipped (checkpoints not found)")

        except Exception as e:
            print(f"   ERROR: {e}")


def batch_process_directory(
    data_dir: str,
    pattern: str = "**/*.mp4",
    segment_duration: float = 3.0,
    motion_threshold: float = 50.0,
    perform_3d: bool = False
):
    """
    Batch process all videos in a directory

    Args:
        data_dir: Root directory to search
        pattern: Glob pattern for videos
        segment_duration: Segment length (seconds)
        motion_threshold: Motion threshold (pixels)
        perform_3d: Whether to perform 3D reconstruction
    """
    data_path = Path(data_dir)
    video_files = list(data_path.glob(pattern))

    print("=" * 70)
    print(f"Batch Processing: {len(video_files)} videos")
    print("=" * 70)

    all_results = {}

    for video_file in video_files:
        video_path = str(video_file)
        rel_path = video_file.relative_to(data_path)

        print(f"\n{'=' * 70}")
        print(f"Video: {rel_path}")
        print('=' * 70)

        # Find motion segments
        motion_segments = find_motion_in_video(
            video_path=video_path,
            segment_duration=segment_duration,
            motion_threshold=motion_threshold
        )

        all_results[str(rel_path)] = motion_segments

        # Optionally perform 3D reconstruction
        if perform_3d and motion_segments:
            output_dir = f"outputs/batch/{rel_path.parent}/{rel_path.stem}"
            process_motion_segments_3d(
                video_path=video_path,
                motion_segments=motion_segments,
                output_dir=output_dir
            )

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)

    total_motion_segments = sum(len(segments) for segments in all_results.values())

    print(f"Videos processed: {len(video_files)}")
    print(f"Total motion segments found: {total_motion_segments}")
    print()

    for video_path, segments in all_results.items():
        if segments:
            print(f"✅ {video_path}: {len(segments)} motion segments")
            for start, end in segments:
                print(f"     {start:.1f}s - {end:.1f}s")
        else:
            print(f"❌ {video_path}: No motion detected")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   SAM 3D Batch Processing Examples                   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Find motion in a single video
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Find Motion Segments in Single Video")
    print("=" * 70)

    single_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if os.path.exists(single_video):
        motion_segments = find_motion_in_video(
            video_path=single_video,
            segment_duration=3.0,
            motion_threshold=50.0
        )

        # Optionally reconstruct 3D for first motion segment
        # if motion_segments:
        #     process_motion_segments_3d(
        #         video_path=single_video,
        #         motion_segments=[motion_segments[0]],  # Just first segment
        #         output_dir="outputs/example1"
        #     )
    else:
        print(f"❌ Video not found: {single_video}")

    # Example 2: Batch process all videos in mouse_1
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Batch Process All Videos in mouse_1")
    print("=" * 70)

    data_dir = "/home/joon/dev/data/markerless_mouse/mouse_1"

    if os.path.exists(data_dir):
        batch_process_directory(
            data_dir=data_dir,
            pattern="**/*.mp4",
            segment_duration=3.0,
            motion_threshold=50.0,
            perform_3d=False  # Set to True to enable 3D reconstruction
        )
    else:
        print(f"❌ Directory not found: {data_dir}")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("""
Next steps:
1. Review motion segments found above
2. Set perform_3d=True to enable 3D reconstruction
3. Adjust motion_threshold if needed (lower = more sensitive)
4. Modify segment_duration for different time windows
    """)

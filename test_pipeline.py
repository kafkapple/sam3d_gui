#!/usr/bin/env python3
"""
Test script for SAM 3D GUI pipeline
Tests video processing and segmentation without 3D reconstruction
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sam3d_processor import SAM3DProcessor
from pathlib import Path


def test_video_info():
    """Test video information extraction"""
    print("=" * 60)
    print("TEST 1: Video Information Extraction")
    print("=" * 60)

    processor = SAM3DProcessor()
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False

    try:
        info = processor.get_video_info(test_video)
        print(f"✅ Video info extracted successfully:")
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   FPS: {info['fps']:.2f}")
        print(f"   Frame count: {info['frame_count']}")
        print(f"   Duration: {info['duration']:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_frame_extraction():
    """Test frame extraction"""
    print("\n" + "=" * 60)
    print("TEST 2: Frame Extraction")
    print("=" * 60)

    processor = SAM3DProcessor()
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False

    try:
        # Extract 10 frames
        frames = processor.extract_frames(test_video, start_frame=0, num_frames=10, stride=1)
        print(f"✅ Extracted {len(frames)} frames")
        print(f"   Frame shape: {frames[0].shape}")
        print(f"   Frame dtype: {frames[0].dtype}")
        return len(frames) == 10
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_segmentation():
    """Test object segmentation"""
    print("\n" + "=" * 60)
    print("TEST 3: Object Segmentation")
    print("=" * 60)

    processor = SAM3DProcessor()
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False

    try:
        # Extract single frame
        frames = processor.extract_frames(test_video, start_frame=0, num_frames=1)
        frame = frames[0]

        # Test different segmentation methods
        methods = ['simple_threshold', 'contour']
        for method in methods:
            mask = processor.segment_object_interactive(frame, method=method)
            object_pixels = mask.sum()
            print(f"✅ {method}: segmented {object_pixels} pixels ({object_pixels/mask.size*100:.1f}%)")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_tracking():
    """Test object tracking"""
    print("\n" + "=" * 60)
    print("TEST 4: Object Tracking")
    print("=" * 60)

    processor = SAM3DProcessor()
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False

    try:
        # Extract frames for 1 second
        info = processor.get_video_info(test_video)
        num_frames = int(info['fps'])  # 1 second
        frames = processor.extract_frames(test_video, start_frame=0, num_frames=num_frames)

        print(f"Tracking across {len(frames)} frames...")

        # Track object
        result = processor.track_object_across_frames(
            frames,
            motion_threshold=50.0,
            fps=info['fps']
        )

        print(f"✅ Tracking complete:")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Frames analyzed: {len(result.segments)}")
        print(f"   Motion detected: {result.motion_detected}")

        if result.segments:
            # Calculate motion statistics
            displacements = []
            for i in range(1, len(result.segments)):
                dx = result.segments[i].center[0] - result.segments[i-1].center[0]
                dy = result.segments[i].center[1] - result.segments[i-1].center[1]
                disp = (dx**2 + dy**2) ** 0.5
                displacements.append(disp)

            if displacements:
                print(f"   Max displacement: {max(displacements):.1f} pixels")
                print(f"   Avg displacement: {sum(displacements)/len(displacements):.1f} pixels")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_segment_processing():
    """Test complete video segment processing"""
    print("\n" + "=" * 60)
    print("TEST 5: Video Segment Processing (without 3D reconstruction)")
    print("=" * 60)

    processor = SAM3DProcessor()
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"
    output_dir = "/home/joon/dev/sam3d_gui/outputs/test"

    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False

    try:
        print(f"Processing 2-second segment from {test_video}...")

        # Process without 3D reconstruction (output_dir=None)
        result, reconstruction = processor.process_video_segment(
            video_path=test_video,
            start_time=0.0,
            duration=2.0,
            output_dir=None,  # Skip 3D reconstruction for quick test
            motion_threshold=30.0
        )

        print(f"✅ Processing complete:")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Frames: {len(result.segments)}")
        print(f"   Motion detected: {result.motion_detected}")
        print(f"   3D reconstruction: {'Yes' if reconstruction else 'No (skipped for test)'}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_motion_segments():
    """Test finding video segments with motion"""
    print("\n" + "=" * 60)
    print("TEST 6: Finding Motion Segments (Advanced)")
    print("=" * 60)

    processor = SAM3DProcessor()
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False

    try:
        info = processor.get_video_info(test_video)
        print(f"Analyzing video: {info['duration']:.1f}s total")

        # Sample multiple segments
        segment_duration = 3.0
        num_segments = min(3, int(info['duration'] / segment_duration))

        motion_segments = []

        for i in range(num_segments):
            start_time = i * segment_duration
            print(f"\n  Testing segment {i+1}/{num_segments} (start: {start_time:.1f}s)...")

            result, _ = processor.process_video_segment(
                video_path=test_video,
                start_time=start_time,
                duration=segment_duration,
                output_dir=None,
                motion_threshold=30.0
            )

            if result.motion_detected:
                motion_segments.append((start_time, start_time + segment_duration))
                print(f"    ✅ Motion detected!")
            else:
                print(f"    ❌ No significant motion")

        print(f"\n✅ Analysis complete:")
        print(f"   Total segments analyzed: {num_segments}")
        print(f"   Segments with motion: {len(motion_segments)}")

        if motion_segments:
            print(f"   Motion segments:")
            for start, end in motion_segments:
                print(f"     - {start:.1f}s to {end:.1f}s")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SAM 3D GUI Pipeline Test Suite")
    print("=" * 60)

    tests = [
        ("Video Info", test_video_info),
        ("Frame Extraction", test_frame_extraction),
        ("Segmentation", test_segmentation),
        ("Object Tracking", test_tracking),
        ("Video Segment Processing", test_video_segment_processing),
        ("Motion Detection", test_find_motion_segments),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

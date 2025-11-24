#!/usr/bin/env python3
"""
SAM 3D GUI - CLI Processing Script (No GUI Required)
헤드리스 환경에서 비디오 처리
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sam3d_processor import SAM3DProcessor

def main():
    """
    GUI 없이 비디오 처리 예제
    """
    print("=" * 60)
    print("SAM 3D Video Processor (Headless Mode)")
    print("=" * 60)
    print()

    # 프로세서 초기화
    processor = SAM3DProcessor()
    print("✓ Processor initialized")

    # 비디오 경로 설정
    video_path = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    # 처리 파라미터
    start_time = 0.0        # 시작 시간 (초)
    duration = 3.0          # 처리 구간 길이 (초)
    motion_threshold = 50.0 # 모션 감지 임계값 (픽셀)
    output_dir = "outputs/"

    print(f"Video: {video_path}")
    print(f"Processing: {start_time}s - {start_time + duration}s")
    print(f"Motion threshold: {motion_threshold} pixels")
    print()

    # 비디오 정보 확인
    try:
        info = processor.get_video_info(video_path)
        print(f"Video Info:")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}")
        print(f"  Total frames: {info['frame_count']}")
        print(f"  Duration: {info['duration']:.2f}s")
        print()
    except Exception as e:
        print(f"Error reading video info: {e}")
        return 1

    # 비디오 처리
    print("Processing video segment...")
    try:
        result, reconstruction = processor.process_video_segment(
            video_path=video_path,
            start_time=start_time,
            duration=duration,
            output_dir=output_dir,
            motion_threshold=motion_threshold,
            segmentation_method='contour',
            do_3d_reconstruction=False  # 3D 재구성 생략 (빠른 처리)
        )

        print("✓ Processing complete!")
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Frames analyzed: {len(result.segments)}")
        print(f"Motion detected: {'Yes ✓' if result.motion_detected else 'No ✗'}")

        if result.motion_detected:
            # 변위 통계 계산
            displacements = []
            for i in range(1, len(result.segments)):
                prev_center = result.segments[i-1].center
                curr_center = result.segments[i].center
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                displacement = (dx**2 + dy**2)**0.5
                displacements.append(displacement)

            if displacements:
                max_disp = max(displacements)
                avg_disp = sum(displacements) / len(displacements)
                print(f"Max displacement: {max_disp:.1f} pixels")
                print(f"Avg displacement: {avg_disp:.1f} pixels")

        print()
        print(f"Output directory: {output_dir}")
        print("Files:")
        print("  - mask_overlay.png (시각화)")

        if reconstruction:
            print("  - reconstruction data available")

        print()
        print("✓ Done!")
        return 0

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

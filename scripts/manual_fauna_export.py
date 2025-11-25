#!/usr/bin/env python3
"""
Manual Fauna Export Script
직접 /tmp 배치 결과를 fauna dataset으로 변환

Usage:
    python scripts/manual_fauna_export.py <temp_dir> <output_dir> [--file-structure {flat|video_folders}]

Example:
    python scripts/manual_fauna_export.py /tmp/sam3d_batch_5zjagqze outputs/fauna_datasets/mouse_batch_manual
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse


def export_fauna_dataset(temp_dir: Path, output_dir: Path, file_structure: str = "video_folders"):
    """
    /tmp 배치 결과를 fauna dataset 형식으로 변환

    Args:
        temp_dir: /tmp/sam3d_batch_XXX 경로
        output_dir: 출력 디렉토리 (예: outputs/fauna_datasets/mouse_batch_manual)
        file_structure: 'flat' 또는 'video_folders'
    """

    print(f"\n{'='*60}")
    print(f"Fauna Dataset Export")
    print(f"{'='*60}")
    print(f"Source: {temp_dir}")
    print(f"Output: {output_dir}")
    print(f"Structure: {file_structure}")
    print()

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # video_XXX 디렉토리 스캔
    video_dirs = sorted([d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith("video_")])

    if not video_dirs:
        print(f"❌ 비디오 디렉토리를 찾을 수 없습니다: {temp_dir}")
        return

    print(f"✅ {len(video_dirs)}개 비디오 발견")
    print()

    # 메타데이터 초기화
    metadata = {
        "dataset_name": output_dir.name,
        "created_at": datetime.now().isoformat(),
        "file_structure": file_structure,
        "videos": [],
        "total_frames": 0,
    }

    total_frames_copied = 0

    # 각 비디오 처리
    for video_idx, video_dir in enumerate(video_dirs):
        video_name = video_dir.name  # video_000, video_001, ...
        video_prefix = f"video{int(video_name.split('_')[1]):03d}"  # video001, video002, ...

        print(f"[{video_idx+1}/{len(video_dirs)}] {video_name} 처리 중...")

        # 프레임 디렉토리 스캔
        frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])

        if not frame_dirs:
            print(f"  ⚠️  경고: {video_name}에 프레임 없음, 건너뜀")
            continue

        # file_structure에 따라 출력 경로 결정
        if file_structure == "video_folders":
            video_output_dir = output_dir / video_prefix
            video_output_dir.mkdir(exist_ok=True)

        video_frames_copied = 0

        # 각 프레임 복사
        for frame_dir in frame_dirs:
            frame_idx = int(frame_dir.name.split('_')[1])

            # 원본 파일 경로
            src_rgb = frame_dir / "original.png"
            src_mask = frame_dir / "mask.png"

            if not src_rgb.exists() or not src_mask.exists():
                print(f"  ⚠️  경고: {frame_dir.name} 파일 누락")
                continue

            # 목적지 경로 결정
            if file_structure == "video_folders":
                # video001/frame_0000_rgb.png
                dst_rgb = video_output_dir / f"frame_{frame_idx:04d}_rgb.png"
                dst_mask = video_output_dir / f"frame_{frame_idx:04d}_mask.png"
            else:  # flat
                # video001_frame_0000_rgb.png
                dst_rgb = output_dir / f"{video_prefix}_frame_{frame_idx:04d}_rgb.png"
                dst_mask = output_dir / f"{video_prefix}_frame_{frame_idx:04d}_mask.png"

            # 파일 복사
            shutil.copy2(src_rgb, dst_rgb)
            shutil.copy2(src_mask, dst_mask)

            video_frames_copied += 1

        total_frames_copied += video_frames_copied

        # 메타데이터 추가
        metadata["videos"].append({
            "video_name": video_prefix,
            "source_dir": str(video_dir),
            "num_frames": video_frames_copied,
        })

        print(f"  ✅ {video_frames_copied} 프레임 복사 완료")

    # 메타데이터 저장
    metadata["total_frames"] = total_frames_copied
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(f"{'='*60}")
    print(f"✅ Fauna Export 완료!")
    print(f"{'='*60}")
    print(f"총 비디오: {len(metadata['videos'])}")
    print(f"총 프레임: {total_frames_copied}")
    print(f"출력 경로: {output_dir}")
    print(f"메타데이터: {metadata_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Manual Fauna Export - /tmp 배치 결과를 fauna dataset으로 변환"
    )
    parser.add_argument(
        "temp_dir",
        type=str,
        help="임시 배치 디렉토리 경로 (예: /tmp/sam3d_batch_XXX)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="출력 디렉토리 (예: outputs/fauna_datasets/mouse_batch_manual)"
    )
    parser.add_argument(
        "--file-structure",
        type=str,
        choices=["flat", "video_folders"],
        default="video_folders",
        help="파일 구조 (기본값: video_folders)"
    )

    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    output_dir = Path(args.output_dir)

    if not temp_dir.exists():
        print(f"❌ 오류: 임시 디렉토리를 찾을 수 없습니다: {temp_dir}")
        sys.exit(1)

    export_fauna_dataset(temp_dir, output_dir, args.file_structure)


if __name__ == "__main__":
    main()

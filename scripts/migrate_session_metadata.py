#!/usr/bin/env python3
"""
Session Metadata Migration Script

기존 session_metadata.json 파일의 절대 경로를 상대 경로로 변환합니다.

Usage:
    python scripts/migrate_session_metadata.py <session_path> [--data-root <path>]
    python scripts/migrate_session_metadata.py outputs/sessions/mouse_batch_20251128_163151/session_metadata.json
    python scripts/migrate_session_metadata.py outputs/sessions/mouse_batch_20251128_163151/session_metadata.json --data-root /media/joon/kafka/data/markerless_mouse

Features:
    - 절대 경로의 video_path를 상대 경로로 변환
    - data_root 필드 추가
    - 기존 파일 백업 (.bak)
    - 호환성: 이미 변환된 파일은 건너뜀
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Optional


def compute_common_data_root(video_paths: list) -> str:
    """
    여러 비디오 경로에서 공통 루트 디렉토리 계산
    """
    if not video_paths:
        return ""

    # 절대 경로만 필터링
    abs_paths = [Path(p) for p in video_paths if Path(p).is_absolute()]

    if not abs_paths:
        return ""

    # 모든 경로를 Path 객체로 변환
    paths = [p.resolve() for p in abs_paths]

    # 공통 상위 경로 찾기
    common_parts = list(paths[0].parts)

    for path in paths[1:]:
        path_parts = list(path.parts)
        # 공통 부분만 유지
        new_common = []
        for a, b in zip(common_parts, path_parts):
            if a == b:
                new_common.append(a)
            else:
                break
        common_parts = new_common

    if not common_parts:
        return ""

    common_root = Path(*common_parts)
    return str(common_root)


def migrate_metadata(
    metadata_path: str,
    data_root: Optional[str] = None,
    dry_run: bool = False,
    backup: bool = True
) -> dict:
    """
    메타데이터 파일 마이그레이션

    Args:
        metadata_path: session_metadata.json 파일 경로
        data_root: 데이터 루트 경로 (None이면 자동 계산)
        dry_run: True면 실제 변경 없이 결과만 출력
        backup: True면 기존 파일 백업

    Returns:
        마이그레이션된 메타데이터 딕셔너리
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 메타데이터 로드
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 이미 변환되었는지 확인
    if 'data_root' in metadata:
        print(f"⚠️  Already migrated (data_root exists): {metadata['data_root']}")

        # 기존 video_path가 상대 경로인지 확인
        if metadata.get('videos'):
            first_video_path = metadata['videos'][0].get('video_path', '')
            if not Path(first_video_path).is_absolute():
                print("   video_paths are already relative. Skipping.")
                return metadata
            else:
                print("   video_paths are still absolute. Continuing migration...")

    # video_path 목록 추출
    video_paths = [v.get('video_path', '') for v in metadata.get('videos', [])]

    if not video_paths:
        print("⚠️  No videos found in metadata")
        return metadata

    # data_root 계산 또는 사용자 지정 값 사용
    if data_root is None:
        data_root = compute_common_data_root(video_paths)
        print(f"📁 Auto-detected data_root: {data_root}")
    else:
        print(f"📁 Using specified data_root: {data_root}")

    # data_root를 최상단에 추가 (기존 키 순서 유지하면서)
    # OrderedDict를 사용하여 data_root를 맨 앞에 배치
    from collections import OrderedDict
    new_metadata = OrderedDict()
    new_metadata['data_root'] = data_root  # 최상단에 배치

    # 기존 키들 복사 (data_root 제외)
    for key, value in metadata.items():
        if key != 'data_root':
            new_metadata[key] = value

    metadata = dict(new_metadata)

    # video_path를 상대 경로로 변환
    converted_count = 0
    for video in metadata.get('videos', []):
        abs_video_path = video.get('video_path', '')

        if not abs_video_path:
            continue

        if data_root and Path(abs_video_path).is_absolute():
            try:
                rel_video_path = str(Path(abs_video_path).relative_to(data_root))
                video['video_path'] = rel_video_path
                converted_count += 1
            except ValueError:
                # data_root 하위가 아닌 경우 절대 경로 유지
                print(f"   ⚠️  Cannot convert (not under data_root): {abs_video_path}")

    print(f"✅ Converted {converted_count}/{len(video_paths)} video paths to relative")

    if dry_run:
        print("\n[DRY RUN] Changes not saved. Preview:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False)[:1000] + "...")
        return metadata

    # 백업 생성
    if backup:
        backup_path = metadata_path.with_suffix('.json.bak')
        shutil.copy2(metadata_path, backup_path)
        print(f"💾 Backup created: {backup_path}")

    # 메타데이터 저장
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Metadata saved: {metadata_path}")

    return metadata


def update_data_root(metadata_path: str, new_data_root: str, backup: bool = True) -> dict:
    """
    기존 메타데이터의 data_root만 업데이트

    다른 환경으로 이동했을 때 사용

    Args:
        metadata_path: session_metadata.json 파일 경로
        new_data_root: 새로운 데이터 루트 경로
        backup: True면 기존 파일 백업

    Returns:
        업데이트된 메타데이터 딕셔너리
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 메타데이터 로드
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    old_data_root = metadata.get('data_root', '(not set)')
    print(f"📁 Old data_root: {old_data_root}")
    print(f"📁 New data_root: {new_data_root}")

    # data_root 업데이트
    metadata['data_root'] = new_data_root

    # 백업 생성
    if backup:
        backup_path = metadata_path.with_suffix('.json.bak')
        shutil.copy2(metadata_path, backup_path)
        print(f"💾 Backup created: {backup_path}")

    # 메타데이터 저장
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ data_root updated: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Migrate session metadata to use relative video paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 자동으로 data_root 계산하여 변환
  python scripts/migrate_session_metadata.py outputs/sessions/mouse_batch_20251128_163151/session_metadata.json

  # data_root 지정하여 변환
  python scripts/migrate_session_metadata.py session_metadata.json --data-root /media/joon/kafka/data/markerless_mouse

  # data_root만 업데이트 (다른 환경으로 이동 후)
  python scripts/migrate_session_metadata.py session_metadata.json --update-data-root /new/path/to/data

  # 변경 내용 미리보기 (실제 변경 없음)
  python scripts/migrate_session_metadata.py session_metadata.json --dry-run
        """
    )

    parser.add_argument('metadata_path', help='Path to session_metadata.json')
    parser.add_argument('--data-root', help='Specify data root path (auto-detect if not provided)')
    parser.add_argument('--update-data-root', help='Only update data_root to new value')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without saving')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup file')

    args = parser.parse_args()

    try:
        if args.update_data_root:
            # data_root만 업데이트
            update_data_root(
                args.metadata_path,
                args.update_data_root,
                backup=not args.no_backup
            )
        else:
            # 전체 마이그레이션
            migrate_metadata(
                args.metadata_path,
                data_root=args.data_root,
                dry_run=args.dry_run,
                backup=not args.no_backup
            )

        print("\n🎉 Migration completed successfully!")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()

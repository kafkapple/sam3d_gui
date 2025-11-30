# Session 이동 가이드

다른 컴퓨터로 SAM3D GUI 세션을 복사하는 방법을 설명합니다.

---

## 세션 구조 이해

### 디렉토리 구조

```
sam3d_gui/
├── outputs/
│   ├── sessions/                          # 세션 데이터
│   │   └── {session_name}/
│   │       ├── session_metadata.json      # 세션 메타데이터 (필수)
│   │       ├── video_000_xxx/             # 비디오별 폴더
│   │       │   ├── frame_0000/
│   │       │   │   ├── original.png       # 원본 프레임
│   │       │   │   └── mask.png           # 세그멘테이션 마스크
│   │       │   ├── frame_0001/
│   │       │   └── ...
│   │       └── video_001_xxx/
│   │           └── ...
│   │
│   └── 3d_meshes/                         # 3D 메시 데이터 (선택)
│       └── {session_name}/
│           ├── video_000_frame0015_143022.ply
│           └── ...
```

### 필수 파일

| 파일/폴더 | 용도 | 크기 예상 |
|-----------|------|----------|
| `session_metadata.json` | 세션 정보, 비디오 목록, 설정 | ~20KB |
| `video_XXX/frame_XXXX/original.png` | 원본 RGB 프레임 | ~1-2MB/프레임 |
| `video_XXX/frame_XXXX/mask.png` | 세그멘테이션 마스크 | ~5-50KB/프레임 |

### 선택 파일

| 파일/폴더 | 용도 | 크기 예상 |
|-----------|------|----------|
| `3d_meshes/{session}/` | 생성된 3D 메시 | ~10-100MB/메시 |

---

## 세션 복사 방법

### 방법 1: 전체 세션 복사 (권장)

```bash
# 소스 서버에서
cd /path/to/sam3d_gui

# 세션만 압축
tar -czf session_backup.tar.gz outputs/sessions/{session_name}

# 3D 메시 포함 시
tar -czf session_with_mesh.tar.gz \
  outputs/sessions/{session_name} \
  outputs/3d_meshes/{session_name}
```

```bash
# 대상 서버로 복사
scp session_backup.tar.gz user@target:/path/to/sam3d_gui/

# 대상 서버에서 압축 해제
cd /path/to/sam3d_gui
tar -xzf session_backup.tar.gz
```

### 방법 2: rsync 사용 (대용량, 증분 복사)

```bash
# 세션 폴더만 동기화
rsync -avz --progress \
  outputs/sessions/{session_name}/ \
  user@target:/path/to/sam3d_gui/outputs/sessions/{session_name}/

# 3D 메시 포함
rsync -avz --progress \
  outputs/3d_meshes/{session_name}/ \
  user@target:/path/to/sam3d_gui/outputs/3d_meshes/{session_name}/
```

### 방법 3: 전체 outputs 폴더 복사

```bash
# 모든 세션 복사
rsync -avz --progress \
  outputs/ \
  user@target:/path/to/sam3d_gui/outputs/
```

---

## 세션 로드 확인

### 1. 복사 후 확인

```bash
# 대상 서버에서
cd /path/to/sam3d_gui

# 세션 폴더 확인
ls -la outputs/sessions/

# 메타데이터 확인
cat outputs/sessions/{session_name}/session_metadata.json | head -50
```

### 2. GUI에서 로드

1. SAM3D GUI 실행: `./run.sh`
2. Batch Processing 탭 이동
3. Session 섹션에서 "🔄 스캔" 클릭
4. 드롭다운에서 복사한 세션 선택
5. "📂 로드" 클릭

---

## session_metadata.json 구조

```json
{
  "session_name": "mouse_batch_20251128_163151",
  "created_at": "2025-11-28T16:31:51",
  "updated_at": "2025-11-28T16:32:05",
  "source_directory": "/path/to/original/data",
  "file_structure": "video_folders",
  "target_frames": 100,
  "videos": [
    {
      "video_name": "video_000_0",
      "source_path": "/original/path/to/video.mp4",
      "frame_count": 100,
      "has_masks": true
    }
  ],
  "annotation_points": {
    "foreground": [[x1, y1], [x2, y2]],
    "background": [[x3, y3]]
  },
  "per_video_annotations": {
    "video_000_0": {
      "foreground": [...],
      "background": [...]
    }
  }
}
```

---

## 주의사항

### 1. 경로 문제

`session_metadata.json`의 `source_path`는 원본 비디오 경로를 참조합니다. 다른 컴퓨터에서는 이 경로가 다를 수 있습니다.

**해결 방법**:
- 세션 로드 시 원본 비디오가 없어도 저장된 프레임/마스크를 사용하여 Preview 가능
- 추가 Propagation이 필요하면 동일한 비디오를 같은 경로에 배치하거나 메타데이터 수정

### 2. 디스크 공간

| 항목 | 크기 예상 |
|------|----------|
| 100프레임 × 1개 비디오 | ~150MB |
| 100프레임 × 72개 비디오 | ~10GB |
| 3D 메시 (비디오당 1개) | ~5-10MB |

### 3. 권한 설정

```bash
# 복사 후 권한 확인
chmod -R u+rw outputs/sessions/{session_name}
```

---

## 예시: gpu05 → gpu06 세션 복사

```bash
# gpu05에서 (소스)
cd ~/sam3d_gui
tar -czf mouse_session.tar.gz outputs/sessions/mouse_batch_20251128_163151

# 로컬로 복사 후 gpu06으로
scp joon@gpu05:~/sam3d_gui/mouse_session.tar.gz .
scp mouse_session.tar.gz joon@gpu06:~/sam3d_gui/

# 또는 직접 전송
ssh gpu05 "cd ~/sam3d_gui && tar -czf - outputs/sessions/mouse_batch_20251128_163151" | \
ssh gpu06 "cd ~/sam3d_gui && tar -xzf -"
```

```bash
# gpu06에서 (대상)
cd ~/sam3d_gui
tar -xzf mouse_session.tar.gz

# 확인
ls outputs/sessions/
# mouse_batch_20251128_163151

# GUI 실행 및 세션 로드
./run.sh
```

---

## 문제 해결

### "세션을 찾을 수 없습니다"

1. 경로 확인: `outputs/sessions/{session_name}/session_metadata.json` 존재 여부
2. 권한 확인: 읽기 권한 있는지
3. JSON 유효성: `python -m json.tool session_metadata.json`

### "프레임/마스크가 없습니다"

1. 폴더 구조 확인: `video_XXX/frame_XXXX/` 형식인지
2. 파일 존재 확인: `original.png`, `mask.png`

### "3D 메시가 없습니다"

1. `outputs/3d_meshes/{session_name}/` 폴더도 함께 복사했는지 확인
2. 메시는 별도로 생성해야 함 (복사하지 않았다면)

---

## 요약

| 작업 | 명령어 |
|------|--------|
| **세션만 백업** | `tar -czf backup.tar.gz outputs/sessions/{name}` |
| **메시 포함 백업** | `tar -czf backup.tar.gz outputs/sessions/{name} outputs/3d_meshes/{name}` |
| **원격 복사** | `scp backup.tar.gz user@target:/path/` |
| **압축 해제** | `tar -xzf backup.tar.gz` |
| **증분 동기화** | `rsync -avz outputs/ user@target:/path/outputs/` |

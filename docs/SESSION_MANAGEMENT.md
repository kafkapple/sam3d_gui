# Session Management - 세션 저장 및 로드 가이드

## 📋 개요

SAM 3D GUI는 Annotation 작업 전체를 저장하고 불러올 수 있는 세션 관리 기능을 제공합니다.

**주요 기능:**
- ✅ Annotation points 저장 (foreground/background)
- ✅ 모든 프레임 원본 이미지 저장
- ✅ 세그멘테이션 마스크 저장
- ✅ 시각화 이미지 저장
- ✅ 메타데이터 JSON 저장
- ✅ 세션 완전 복원

---

## 🗂️ 저장 위치 및 구조

### 디렉토리 구조

```
outputs/sessions/{YYYYMMDD_HHMMSS}/
├── session_metadata.json       # 세션 메타데이터
├── frame_0000/
│   ├── original.png           # 원본 프레임
│   ├── mask.png               # 세그멘테이션 마스크 (있는 경우)
│   └── visualization.png      # 마스크 오버레이 시각화
├── frame_0001/
│   └── ...
└── frame_XXXX/
    └── ...
```

### 메타데이터 JSON 구조

`session_metadata.json`:
```json
{
  "session_id": "20251124_131200",
  "video_path": "/path/to/video.mp4",
  "num_frames": 90,
  "current_frame_idx": 0,
  "annotations": {
    "foreground": [[x1, y1], [x2, y2], ...],
    "background": [[x3, y3], ...]
  },
  "frame_info": [
    {
      "frame_idx": 0,
      "has_mask": true,
      "mask_area": 12345,
      "mask_percentage": 5.2
    },
    ...
  ],
  "timestamp": "2025-11-24T13:12:00"
}
```

---

## 🚀 사용 방법

### 1. 세션 저장

**워크플로우:**
1. Interactive Mode에서 비디오 로드
2. Foreground/Background points로 annotation
3. Segment Current Frame 또는 Propagate to All Frames 실행
4. **"💾 Save Session"** 버튼 클릭

**결과:**
- 자동으로 타임스탬프 ID 생성 (예: `20251124_131200`)
- `outputs/sessions/{session_id}/` 폴더에 모든 데이터 저장
- 상태 메시지에 세션 ID 표시

**예시:**
```
✅ 세션 저장 완료: 20251124_131200
   - 90 frames
   - 15 masks
   - 위치: outputs/sessions/20251124_131200/
```

---

### 2. 저장된 세션 목록 조회

**방법:**
1. **"📋 목록 조회"** 버튼 클릭

**결과:**
- 저장된 모든 세션 목록 표시
- 각 세션의 메타정보 (비디오명, 프레임 수, 마스크 수)

**예시 출력:**
```
📋 저장된 세션 목록:

- 20251124_131200
  비디오: mouse_video.mp4
  프레임: 90 / 마스크: 15

- 20251124_100530
  비디오: test_video.mp4
  프레임: 60 / 마스크: 60
```

---

### 3. 세션 로드

**방법:**
1. **"📋 목록 조회"** 버튼으로 세션 ID 확인 (선택사항)
2. 세션 ID 입력란에 ID 입력 (예: `20251124_131200`)
3. **"📂 Load Session"** 버튼 클릭

**결과:**
- 모든 프레임 복원
- 모든 마스크 복원
- Annotation points 복원
- 현재 프레임 시각화 표시

**예시:**
```
✅ 세션 로드 완료: 20251124_131200
   - 90 frames 로드됨
   - 15 masks 로드됨
   - Annotation points 복원됨
```

---

## 📊 기능 비교

### Save Session vs Save Masks Only

| 기능 | Save Session | Save Masks Only |
|------|-------------|-----------------|
| **Annotation points** | ✅ 저장 | ❌ 저장 안됨 |
| **원본 프레임** | ✅ 저장 | ❌ 저장 안됨 |
| **마스크** | ✅ 저장 | ✅ 저장 |
| **시각화** | ✅ 저장 | ❌ 저장 안됨 |
| **메타데이터** | ✅ JSON | ❌ 없음 |
| **완전 복원** | ✅ 가능 | ❌ 불가능 |
| **용량** | 크다 | 작다 |

**권장:**
- **Save Session**: 작업 중단 후 이어서 하려면 필수
- **Save Masks Only**: 마스크만 빠르게 내보내기

---

## 💡 사용 사례

### 사례 1: 작업 중단 후 재개

**상황:**
- 90프레임 비디오 작업 중
- 30프레임까지만 segmentation 완료
- 작업 중단 필요

**해결:**
1. "💾 Save Session" 클릭 → `20251124_131200` 저장
2. 나중에 다시 GUI 접속
3. 세션 ID `20251124_131200` 입력 후 "📂 Load Session"
4. 31번째 프레임부터 이어서 작업

---

### 사례 2: 여러 비디오 동시 작업

**상황:**
- 비디오 A, B, C를 각각 annotation 중
- 비디오 간 전환이 필요

**해결:**
1. 비디오 A 작업 후 Save Session → `session_A`
2. 비디오 B 작업 후 Save Session → `session_B`
3. 비디오 C 작업 후 Save Session → `session_C`
4. 필요할 때마다 각 세션 Load

---

### 사례 3: 품질 검증 후 재작업

**상황:**
- Propagation 완료 후 마스크 품질 확인 필요
- 일부 프레임 재작업 필요

**해결:**
1. Propagation 완료 후 즉시 Save Session
2. 프레임 네비게이션으로 품질 확인
3. 문제 발견 시 이전 세션 Load
4. Annotation 수정 후 다시 Propagate

---

## 🔧 구현 세부사항

### 핵심 함수 (src/web_app.py)

#### 1. save_annotation_session()
- **위치**: lines 597-709
- **기능**: 전체 세션 저장
- **반환**: 저장 결과 메시지

```python
def save_annotation_session(self) -> str:
    """
    Annotation 세션 전체 저장 (annotation points + masks + metadata)
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/sessions/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 메타데이터 저장
    # 2. 각 프레임 저장
    # 3. JSON 저장

    return f"✅ 세션 저장 완료: {session_id}"
```

#### 2. load_annotation_session()
- **위치**: lines 711-800
- **기능**: 저장된 세션 로드
- **입력**: session_id (str)
- **반환**: (visualization, message)

```python
def load_annotation_session(self, session_id: str) -> Tuple[np.ndarray, str]:
    """
    저장된 annotation 세션 로드
    """
    session_dir = Path(f"outputs/sessions/{session_id}")

    # 1. 메타데이터 로드
    # 2. 프레임 및 마스크 로드
    # 3. Annotation points 복원

    return visualization, message
```

#### 3. list_saved_sessions()
- **위치**: lines 802-837
- **기능**: 저장된 세션 목록 조회
- **반환**: 세션 목록 텍스트

```python
def list_saved_sessions(self) -> str:
    """
    저장된 세션 목록 조회
    """
    sessions_dir = Path("outputs/sessions")

    # 각 세션의 메타데이터 읽기
    # 포맷팅하여 반환

    return session_list_text
```

---

## 🐛 문제 해결

### 오류 1: "세션을 찾을 수 없습니다"

**증상:**
```
❌ 세션을 찾을 수 없습니다: 20251124_131200
```

**원인:**
- 잘못된 세션 ID 입력
- 세션이 삭제됨

**해결:**
1. "📋 목록 조회" 클릭하여 실제 세션 ID 확인
2. 정확한 ID 복사하여 입력

---

### 오류 2: 세션 로드 후 프레임 순서 잘못됨

**증상:**
- 프레임이 뒤섞여 표시됨

**원인:**
- 파일 시스템의 정렬 순서 문제

**해결:**
- 코드에서 `sorted()`로 정렬 처리됨 (수정 불필요)

---

### 오류 3: 저장 공간 부족

**증상:**
```
❌ 세션 저장 실패: No space left on device
```

**원인:**
- 디스크 공간 부족
- 고해상도 비디오 + 많은 프레임

**해결:**
1. 오래된 세션 삭제:
   ```bash
   rm -rf outputs/sessions/old_session_id/
   ```
2. Save Masks Only 사용 (용량 적음)

---

## 📝 Best Practices

### ✅ 권장사항

1. **주기적 저장**
   - Propagation 전후 각각 저장
   - Annotation 수정할 때마다 저장

2. **명확한 세션 관리**
   - 세션 ID는 자동 생성되므로 메모 권장
   - 목록 조회로 주기적 확인

3. **디스크 공간 관리**
   - 완료된 작업은 별도 백업 후 삭제
   - 테스트 세션은 즉시 삭제

### ❌ 피해야 할 사항

1. **세션 ID 수동 수정**
   - 자동 생성된 ID를 변경하지 말 것
   - 파일 구조 손상 가능

2. **부분 삭제**
   - 세션 내 일부 파일만 삭제하지 말 것
   - 전체 세션 폴더를 삭제할 것

3. **동시 작업**
   - 같은 세션을 여러 창에서 동시 로드/저장 금지

---

## 📊 통계 및 메타정보

### 저장 용량 예상

| 비디오 | 프레임 수 | 해상도 | 마스크 | 예상 용량 |
|--------|----------|--------|--------|----------|
| 짧은 비디오 | 60 | 640x480 | 60 | ~100MB |
| 중간 비디오 | 90 | 800x600 | 90 | ~200MB |
| 긴 비디오 | 300 | 1920x1080 | 300 | ~1.5GB |

### 처리 시간

| 작업 | 프레임 수 | 예상 시간 |
|------|----------|----------|
| Save Session | 60 | ~5-10초 |
| Load Session | 60 | ~3-5초 |
| List Sessions | - | <1초 |

---

## 🔍 관련 문서

- [UPDATES_LOG.md](../UPDATES_LOG.md) - 전체 업데이트 로그
- [README_CHECKPOINTS.md](../README_CHECKPOINTS.md) - 체크포인트 관리
- [src/web_app.py](../src/web_app.py) - 구현 코드

---

**작성일**: 2025-11-24
**버전**: 1.0
**상태**: ✅ 구현 완료

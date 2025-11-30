# Interactive/Batch Mode 통합 계획

**날짜**: 2025-11-30
**목적**: Batch mode 중심으로 Interactive mode 기능 통합

---

## 현재 상태 분석

### 기능 비교 요약

| 기능 | Interactive | Batch | 통합 후 |
|------|------------|-------|---------|
| **비디오 로딩** | 단일 + 시간 범위 | 다중 + 패턴 | Batch 유지 |
| **어노테이션** | 수동 프레임별 | 참조 + 개별 비디오 | Batch 유지 |
| **프레임 네비게이션** | 슬라이더 + 버튼 | 비디오 선택 + 슬라이더 | Batch 유지 |
| **세그멘테이션** | 현재 프레임 | 참조 프레임 | 통합 |
| **프로파게이션** | 단일 | 공유 + 개별 | Batch 유지 |
| **3D Mesh 생성** | ✅ | ❌ | **추가 필요** |
| **세션 관리** | 단일 비디오 | 다중 비디오 | Batch 유지 |
| **Fauna Export** | ✅ | ✅ | 통합 |
| **QC/Preview** | 프레임별 | 비디오별 + 시각화 | Batch 유지 |

---

## 통합 원칙

> **"비디오 1개 선택 시 Interactive와 동일하게 동작"**

Batch mode에서:
- 1개 비디오 선택 → Interactive 워크플로우
- N개 비디오 선택 → Batch 워크플로우

---

## Interactive에만 있는 기능 분석

### 1. 3D Mesh 생성 - **필요 (추가)**

**현재 상태**: Interactive에서 `generate_3d_mesh()` 버튼으로 현재 프레임의 3D 메시 생성

**Batch에 추가할 내용**:
- 선택된 비디오의 특정 프레임에서 3D 메시 생성
- 일괄 처리: 각 비디오의 지정된 프레임에서 메시 생성

**파일명 형식**:
```
outputs/3d_meshes/{session_name}/
├── video001_frame0015_143022.ply
├── video002_frame0020_143045.ply
└── ...
```

### 2. 시간 범위 지정 (start_time, duration) - **불필요**

**근거**:
- Batch는 전체 비디오 또는 target_frames로 제어
- 시간 범위는 비디오 편집 도구에서 처리하는 것이 적절
- 복잡도 증가 대비 이점 적음

### 3. 수동 Stride 조정 (frame_step 슬라이더) - **불필요**

**근거**:
- `target_frames`로 자동 계산되는 것이 더 직관적
- Batch에서 개별 비디오마다 다른 stride 필요 없음
- 이미 Batch에 `batch_target_frames` 존재

### 4. 프레임 직접 이동 (goto_frame) - **선택적**

**근거**:
- Batch의 Preview 섹션에서 슬라이더로 충분
- 큰 비디오에서는 유용할 수 있음
- 우선순위 낮음

### 5. Frames + Masks Export - **불필요**

**근거**:
- Fauna Export가 동일한 기능 + 더 나은 구조 제공
- 중복 기능

---

## Batch에만 있는 기능 분석

### 1. 다중 비디오 처리 - **유지**

핵심 기능, Interactive 통합의 기반

### 2. 참조 + 개별 어노테이션 - **유지**

효율적인 워크플로우

### 3. QC/Preview 섹션 - **유지**

품질 관리에 필수

### 4. 어노테이션 파일 저장/로드 - **유지**

재사용성에 중요

### 5. 파일 구조 선택 (video_folders/flat) - **유지**

유연성 제공

---

## 통합 계획

### Phase 1: Batch에 3D Mesh 추가 (즉시 구현)

1. **단일 프레임 메시 생성**
   - Preview 섹션에서 현재 보고 있는 프레임의 메시 생성
   - 버튼: "Generate 3D Mesh (Current Frame)"

2. **일괄 메시 생성**
   - 각 비디오의 지정된 프레임(예: 중간 프레임)에서 메시 생성
   - 버튼: "Generate 3D Mesh (All Videos)"

### Phase 2: UI 정리 (추후)

1. Interactive tab 제거 또는 "Legacy" 표시
2. Batch tab 이름을 "Main" 또는 기본으로 변경
3. Quick Mode는 유지 (완전 자동화 용도)

### Phase 3: 세션 마이그레이션 (추후)

1. 기존 Interactive 세션을 Batch 형식으로 변환하는 유틸리티
2. 하위 호환성 유지

---

## 구현 우선순위

| 순위 | 작업 | 이유 |
|------|------|------|
| **1** | Batch에 3D Mesh 버튼 추가 | 핵심 누락 기능 |
| **2** | 메시 파일명/폴더 구조 통일 | 일관성 |
| **3** | Session 이동 문서화 | 사용자 가이드 |
| 4 | Interactive tab 정리 | UI 단순화 |
| 5 | 세션 마이그레이션 도구 | 하위 호환성 |

---

## 파일 구조 제안

### 3D Mesh 출력 구조

```
outputs/
├── 3d_meshes/
│   └── {session_name}/
│       ├── video001_frame0015_143022.ply
│       ├── video002_frame0020_143045.ply
│       └── mesh_metadata.json
│
└── sessions/
    └── {session_name}/
        ├── session_metadata.json
        ├── video_001/
        │   ├── frame_0000_rgb.png
        │   ├── frame_0000_mask.png
        │   └── ...
        └── video_002/
            └── ...
```

### mesh_metadata.json 예시

```json
{
  "session_name": "mouse_experiment_01",
  "created_at": "2025-11-30T14:30:22",
  "meshes": [
    {
      "video": "video001",
      "frame_idx": 15,
      "filename": "video001_frame0015_143022.ply",
      "timestamp": "2025-11-30T14:30:22"
    }
  ]
}
```

---

## 세션 이동 가이드 (문서화 대상)

### 다른 컴퓨터로 세션 복사

**필요한 파일**:
```
outputs/sessions/{session_name}/
├── session_metadata.json    # 세션 정보
├── video_001/               # 각 비디오 폴더
│   ├── frame_0000_rgb.png
│   ├── frame_0000_mask.png
│   └── ...
└── video_002/
    └── ...
```

**복사 방법**:
```bash
# 소스 서버에서
tar -czf session_backup.tar.gz outputs/sessions/{session_name}

# 대상 서버로 복사
scp session_backup.tar.gz user@target:/path/to/sam3d_gui/

# 대상 서버에서 압축 해제
cd /path/to/sam3d_gui
tar -xzf session_backup.tar.gz
```

**3D Mesh 포함 시**:
```bash
# 메시 폴더도 함께 복사
tar -czf session_with_mesh.tar.gz \
  outputs/sessions/{session_name} \
  outputs/3d_meshes/{session_name}
```

---

## 결론

**통합 방향**:
1. Batch mode를 기본으로 사용
2. 3D Mesh 생성 기능을 Batch에 추가
3. Interactive의 시간 범위, 수동 stride 등은 제외 (복잡도 대비 이점 부족)
4. 세션 구조는 Batch 형식으로 통일 (다중 비디오 지원)

**즉시 구현할 것**:
1. Batch Preview 섹션에 "Generate 3D Mesh" 버튼 추가
2. 메시 파일 세션별 폴더 구조
3. 일괄 메시 생성 옵션

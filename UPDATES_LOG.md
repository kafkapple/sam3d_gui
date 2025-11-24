# SAM 3D GUI - 업데이트 로그 (2025-11-24)

## 🎯 해결된 문제들

### 1. ✅ Propagate 시 잘못된 마스크 표시 문제
**문제**: Propagate 완료 후 마지막 프레임에 지저분한 마스크 표시

**원인**:
- `propagate_to_all_frames()`가 contour 기반 세그멘테이션 사용
- 각 프레임마다 독립적으로 처리하여 일관성 없음

**해결**:
- **SAM2를 사용하도록 변경** (`src/web_app.py:451-539`)
- 모든 프레임에 동일한 point annotation 적용
- 현재 프레임 표시 (마지막 프레임 아님)

**코드**:
```python
# SAM2를 사용하여 각 프레임 세그멘테이션
for i, frame in enumerate(self.frames):
    if self.sam2_predictor is not None:
        self.sam2_predictor.set_image(frame_rgb)
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        mask = masks[np.argmax(scores)]
```

---

### 2. ✅ 프레임 네비게이션 기능 추가
**문제**: 특정 프레임별로 확인하는 기능 없음

**해결**:
- 프레임 네비게이션 UI 추가 (`src/web_app.py:798-823`)
- `navigate_frame()` 함수 구현 (`src/web_app.py:628-687`)

**기능**:
- ⏮️ **처음**: 첫 번째 프레임으로 이동
- ◀️ **이전**: N 프레임 뒤로 이동
- ▶️ **다음**: N 프레임 앞으로 이동
- ⏭️ **마지막**: 마지막 프레임으로 이동
- **이동 간격**: 1-10 프레임 슬라이더
- **특정 프레임**: 프레임 번호 입력하여 이동

**시각화**:
- 현재 프레임에 마스크 표시 (있을 경우)
- Annotation points 표시 (foreground: 녹색, background: 빨간색)
- 프레임 번호 및 마스크 통계 표시

---

### 3. ✅ SAM 3D 체크포인트 다운로드 스크립트
**문제**: "SAM 3D config not found" 오류

**원인**: SAM 3D 체크포인트가 다운로드되지 않음

**해결**:
- `download_sam3d.sh` 스크립트 생성
- HuggingFace에서 자동 다운로드

**사용 방법**:
```bash
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh
```

**다운로드 위치** (자동 감지):
1. Primary: `~/dev/sam-3d-objects/checkpoints/hf/`
2. Alternative: `~/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf/`

---

### 4. ✅ Hydra Config 기반 체크포인트 관리
**문제**: 체크포인트 경로가 코드에 하드코딩됨

**해결**:
- **Config 파일**: `config/model_config.yaml`
- **Loader**: `src/config_loader.py`

---

### 5. ✅ 세션 저장 및 로드 기능
**문제**: Annotation 작업 결과를 저장/로드하는 방법 없음

**해결**:
- `save_annotation_session()` 함수 구현 (`src/web_app.py:597-709`)
- `load_annotation_session()` 함수 구현 (`src/web_app.py:711-800`)
- `list_saved_sessions()` 함수 구현 (`src/web_app.py:802-837`)

**기능**:
- ✅ **Save Session**: 타임스탬프 ID로 자동 저장
  - Annotation points (foreground/background)
  - 모든 프레임 원본 이미지
  - 세그멘테이션 마스크
  - 시각화 이미지
  - JSON 메타데이터

- ✅ **Load Session**: ID로 전체 세션 복원
  - 모든 프레임 및 마스크 로드
  - Annotation points 복원
  - 이전 작업 상태 완전 복구

- ✅ **List Sessions**: 저장된 세션 목록 조회
  - 세션 ID, 비디오명, 프레임 수, 마스크 수 표시

**저장 위치**:
```
outputs/sessions/{YYYYMMDD_HHMMSS}/
├── session_metadata.json
├── frame_0000/
│   ├── original.png
│   ├── mask.png
│   └── visualization.png
└── frame_XXXX/...
```

**현재 설정**:
```yaml
sam2:
  checkpoint: ~/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
  config: configs/sam2.1/sam2.1_hiera_l.yaml
  device: cuda

sam3d:
  checkpoint_dir: ~/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf
  checkpoint_dir_alt: ~/dev/sam-3d-objects/checkpoints/hf

data:
  default_dir: ~/dev/data/markerless_mouse/
  output_dir: ~/dev/sam3d_gui/outputs/
```

**장점**:
- 환경 변수 지원: `${oc.env:HOME}`
- Primary/Alternative 경로 자동 선택
- 한 곳에서 모든 경로 관리

---

## 📊 기능 개선 요약

| 기능 | 이전 | 현재 |
|------|------|------|
| **Propagate** | Contour (부정확) | SAM2 (정확) |
| **프레임 확인** | ❌ 불가능 | ✅ 네비게이션 |
| **프레임 이동** | ❌ 없음 | ✅ 간격 조절 |
| **특정 프레임** | ❌ 없음 | ✅ 번호 입력 |
| **체크포인트** | 하드코딩 | Config 관리 |
| **SAM3D 설치** | 수동 | 스크립트 자동 |
| **세션 저장** | ❌ 없음 | ✅ JSON + 이미지 |
| **세션 로드** | ❌ 없음 | ✅ 완전 복원 |

---

## 🔧 변경된 파일

### 신규 파일:
1. `config/model_config.yaml` - 체크포인트 경로 설정
2. `src/config_loader.py` - Config 로더
3. `download_sam3d.sh` - SAM3D 다운로드 스크립트
4. `README_CHECKPOINTS.md` - 체크포인트 가이드
5. `UPDATES_LOG.md` - 이 문서

### 수정 파일:
1. `src/web_app.py`:
   - `propagate_to_all_frames()` - SAM2 사용 (lines 451-539)
   - `navigate_frame()` - 프레임 네비게이션 추가 (lines 628-687)
   - `save_annotation_session()` - 세션 저장 (lines 597-709)
   - `load_annotation_session()` - 세션 로드 (lines 711-800)
   - `list_saved_sessions()` - 세션 목록 조회 (lines 802-837)
   - UI: 프레임 네비게이션 컨트롤 추가 (lines 798-823)
   - UI: 세션 관리 컨트롤 추가 (lines 1064-1082)
   - Event handlers: 세션 관리 버튼 (lines 1168-1183)
   - Config 기반 초기화

---

## 🚀 사용 방법

### 1. 서버 실행
```bash
cd /home/joon/dev/sam3d_gui
./run.sh
```

### 2. 웹 GUI 접속
- Local: http://localhost:7860
- Network: http://192.168.45.10:7860

### 3. Interactive Mode 워크플로우

#### Step 1: 비디오 로드
1. 비디오 파일 선택 (드롭다운)
2. 시작 시간 & 길이 설정
3. "📹 비디오 로드" 클릭

#### Step 2: Point Annotation
1. **Foreground** 선택 → 객체 위치 3-5번 클릭 (녹색 점)
2. **Background** 선택 → 배경 위치 2-3번 클릭 (빨간색 점)
3. "✂️ Segment Current Frame" 클릭

**결과**:
- "Method: SAM2 (confidence: X.XXX)" 표시
- 녹색 마스크 오버레이

#### Step 3: Propagation (옵션)
1. "🔄 Propagate to All Frames" 클릭
2. 전체 프레임에 SAM2 적용 (진행률 표시)

**결과**:
- "Method: SAM2 (전체 프레임)" 표시
- 모든 프레임에 마스크 생성

#### Step 4: 프레임 네비게이션
1. **이동 간격** 슬라이더로 간격 설정 (1-10)
2. **◀️ 이전** / **▶️ 다음** 버튼으로 이동
3. **프레임 번호** 입력하여 특정 프레임으로 점프

**확인 사항**:
- 각 프레임의 마스크 품질
- 객체 추적 일관성
- 마스크 영역 통계

#### Step 5: 3D Mesh 생성 (옵션)
**전제 조건**: SAM3D 체크포인트 다운로드 필요
```bash
./download_sam3d.sh
```

1. "🎲 Generate 3D Mesh" 클릭
2. PLY 파일 자동 생성 & 다운로드
3. MeshLab으로 확인

#### Step 6: 세션 저장 (권장)
1. **"💾 Save Session"** 클릭
2. 자동으로 타임스탬프 ID 생성 (`YYYYMMDD_HHMMSS`)
3. `outputs/sessions/{session_id}/` 폴더에 저장:
   - `session_metadata.json` - Annotation points, 비디오 정보
   - `frame_XXXX/original.png` - 원본 프레임
   - `frame_XXXX/mask.png` - 세그멘테이션 마스크
   - `frame_XXXX/visualization.png` - 마스크 오버레이 시각화

**세션 로드**:
1. **"📋 목록 조회"** 클릭 → 저장된 세션 목록 확인
2. 세션 ID 입력 (예: `20251124_131200`)
3. **"📂 Load Session"** 클릭
4. 모든 프레임, 마스크, annotation 복원

#### Step 7: 마스크만 저장 (옵션)
1. "💾 Save Masks Only" 클릭
2. `outputs/masks/` 폴더에 PNG만 저장 (세션 정보 제외)

---

## 🐛 알려진 이슈 & 해결 방법

### 이슈 1: SAM3D "config not found"
**증상**: "Generate 3D Mesh" 클릭 시 오류

**해결**:
```bash
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh
```

### 이슈 2: SAM2 로딩 느림
**증상**: 서버 시작 후 1-2분 대기

**원인**: SAM2 모델 로딩 (857MB)

**해결**: 정상 동작, 기다리면 됨

### 이슈 3: Propagate 느림
**증상**: 각 프레임 처리에 1-2초 소요

**원인**: SAM2 inference (정확도 위해 필요)

**최적화 옵션**:
- 짧은 비디오 사용 (3-5초)
- GPU 사용 (CUDA)

---

## 📝 체크포인트 상태

### SAM2 (Interactive Segmentation)
- **위치**: `/home/joon/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt`
- **상태**: ✅ 다운로드 완료 (857MB)
- **Config**: `config/model_config.yaml`

### SAM3D (3D Reconstruction)
- **위치**: `/home/joon/dev/sam-3d-objects/checkpoints/hf/`
- **상태**: ❌ **다운로드 필요**
- **다운로드**: `./download_sam3d.sh`

---

## 🔍 테스트 체크리스트

### Interactive Mode 테스트:
- [x] 비디오 로드 (드롭다운 자동 스캔)
- [x] Foreground point 클릭 (녹색 점 표시)
- [x] Background point 클릭 (빨간색 점 표시)
- [x] Segment Current Frame (SAM2 사용)
- [ ] Propagate to All Frames (SAM2 사용) - **테스트 필요**
- [ ] 프레임 네비게이션 (이전/다음/처음/마지막) - **테스트 필요**
- [ ] 프레임 간격 조절 - **테스트 필요**
- [ ] 특정 프레임 이동 - **테스트 필요**
- [ ] **Save Session** - **테스트 필요**
- [ ] **List Sessions** - **테스트 필요**
- [ ] **Load Session** - **테스트 필요**
- [ ] Generate 3D Mesh (SAM3D 체크포인트 다운로드 후) - **테스트 필요**
- [ ] Save Masks Only - **테스트 필요**

---

**작성일**: 2025-11-24
**최종 업데이트**: 2025-11-24 13:30 KST
**버전**: 2.2
**상태**: 세션 저장/로드 기능 추가 완료

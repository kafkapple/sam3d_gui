# SAM Annotator 비교 분석 및 통합 계획

## 📊 비교 대상

- **SAM Annotator** (`/home/joon/dev/mouse-super-resolution/sam_annotator/`)
- **SAM 3D GUI** (`/home/joon/dev/sam3d_gui/`)

---

## 1. 공통 기능 (Overlapping Features)

### 1.1 SAM 2 Interactive Segmentation

| 기능 | SAM Annotator | SAM 3D GUI | 비고 |
|-----|--------------|-----------|------|
| **Point Annotation** | ✅ Foreground/Background | ✅ Foreground/Background | 동일 |
| **Real-time Mask** | ✅ Click → Generate | ✅ Click → Generate | 동일 |
| **SAM 2.1 Support** | ✅ 4 variants (tiny~large) | ✅ Hiera Large | Annotator가 더 다양 |
| **GPU Auto-detection** | ✅ CUDA/CPU | ✅ auto/cuda/cpu | 동일 |

### 1.2 Video/Image Handling

| 기능 | SAM Annotator | SAM 3D GUI | 비고 |
|-----|--------------|-----------|------|
| **Video Support** | ✅ Direct loading | ✅ Extract frames | Annotator가 메모리 효율적 |
| **Frame Navigation** | ✅ Slider | ✅ Slider + Prev/Next | 3D GUI가 더 편리 |
| **Image Folder** | ✅ Glob pattern | ❌ 미지원 | Annotator 독점 |

### 1.3 Data Persistence

| 기능 | SAM Annotator | SAM 3D GUI | 비고 |
|-----|--------------|-----------|------|
| **Save Annotations** | ✅ JSON + PNG | ✅ Session (JSON) | 형식 다름 |
| **Auto-restore** | ✅ Frame reload 시 | ✅ Load session | Annotator는 자동, 3D는 수동 |
| **Mask Export** | ✅ Binary PNG | ✅ Binary PNG | 동일 |

---

## 2. 차별화 기능 (Unique Features)

### 2.1 SAM Annotator 독점 기능

#### ⭐ 1. Direct Video Loading (메모리 효율)
```python
# On-demand frame extraction via cv2.VideoCapture
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
```

**장점**:
- 디스크 공간 절약 (프레임 추출 불필요)
- 빠른 시작 (사전 처리 없음)
- 메모리 효율적

**벤치마크** (VIDEO_SUPPORT.md):
- 90프레임 비디오: 200MB → 0MB 디스크 사용
- Frame 0 로드: ~50ms
- Frame seeking: ~100ms

#### ⭐ 2. Image Folder Support (Glob Pattern)
```python
# Support for pre-extracted frames
input_pattern = "/data/frames/*.png"
frames = sorted(glob.glob(input_pattern))
```

**Use Case**:
- 이미 추출된 프레임 처리
- 다양한 파일명 패턴 지원
- 외부 전처리 파이프라인 연동

#### ⭐ 3. Hydra Configuration Management
```yaml
# config.yaml
model:
  name: sam2.1_hiera_large
  checkpoint: ${oc.env:HOME}/checkpoints/sam2.1_hiera_large.pt
  device: auto
```

**장점**:
- 타입 안전한 설정 관리
- CLI override: `python -m sam_annotator model.device=cuda`
- 환경 변수 통합: `${oc.env:HOME}`
- 버전 관리 용이

#### ⭐ 4. Multi-Model Support
- SAM 2.1: Tiny, Small, Base+, Large
- 런타임 모델 전환 가능
- 속도 vs 품질 trade-off

#### ⭐ 5. Runtime Input Source Switching
```python
def change_input_source(new_path):
    # 재시작 없이 비디오/이미지 폴더 변경
    self._initialize_input(new_path)
    return self.total_frames
```

#### ⭐ 6. Automatic Annotation Restoration
- 프레임 reload 시 자동으로 이전 annotation 로드
- 파일 기반 persistence
- 중단 후 재개 자동 지원

---

### 2.2 SAM 3D GUI 독점 기능

#### ⭐ 1. SAM 3D Object Reconstruction
```python
# 2D mask → 3D mesh generation
def generate_3d_mesh(masks, frames):
    # SAM 3D Objects 사용
    mesh = sam3d_model(images, masks)
    return mesh  # PLY/OBJ export
```

**SAM Annotator에 없는 핵심 기능!**

#### ⭐ 2. Propagation (Tracking)
```python
# 한 프레임 annotation → 전체 비디오 propagate
def propagate_to_all_frames(annotations):
    video_predictor = build_sam2_video_predictor()
    for frame_idx in range(total_frames):
        mask = video_predictor.propagate(frame_idx)
```

**Use Case**:
- 비디오 객체 추적
- 한 번 annotation으로 전체 프레임 처리
- 시간 절약 (수동 annotation 불필요)

#### ⭐ 3. Quick Mode (자동 처리)
- Automatic motion detection
- Batch processing
- 최소 개입으로 전체 비디오 처리

#### ⭐ 4. Session Management (복잡한 프로젝트)
```json
{
  "session_id": "20251124_131200",
  "video_path": "/path/to/video.mp4",
  "annotations": {...},
  "frame_info": [...]
}
```

**장점**:
- 여러 프로젝트 동시 관리
- 세션 간 전환
- 메타데이터 포함

#### ⭐ 5. HuggingFace Authentication
- `.env` 기반 토큰 관리
- Gated model 자동 다운로드
- OAuth2 인증

#### ⭐ 6. Dropdown-based Session Selection
- GUI에서 저장된 세션 목록 확인
- 드롭다운으로 선택
- Copy-paste 불필요

---

## 3. 장단점 비교

### 3.1 SAM Annotator

**✅ 장점**:
1. **메모리 효율**: Direct video loading (디스크 공간 절약)
2. **유연성**: Image folder, glob pattern 지원
3. **설정 관리**: Hydra 기반 전문적 관리
4. **멀티 모델**: 4가지 SAM 2 variants
5. **자동 복원**: Frame reload 시 자동 annotation 로드
6. **단순성**: 코드가 간결하고 이해하기 쉬움 (annotator.py 429줄)

**❌ 단점**:
1. **단순 기능**: 2D annotation만 지원
2. **수동 작업**: 모든 프레임 수동 annotation 필요
3. **3D 미지원**: Mesh reconstruction 없음
4. **Tracking 없음**: Propagation 기능 없음
5. **Session 관리 약함**: File-based만 지원

**최적 Use Case**:
- **정적 이미지 annotation**
- **소수 프레임 처리** (< 100 frames)
- **빠른 프로토타입**
- **메모리 제약 환경**

---

### 3.2 SAM 3D GUI

**✅ 장점**:
1. **3D Reconstruction**: SAM 3D Objects 통합
2. **Propagation**: Video tracking (한 번 annotation → 전체 프레임)
3. **자동화**: Quick mode (motion detection)
4. **세션 관리**: 복잡한 프로젝트 지원
5. **프레임 네비게이션**: Slider + Prev/Next + Goto
6. **인증 관리**: HuggingFace token 통합

**❌ 단점**:
1. **메모리 사용**: 프레임 추출 필요 (디스크 공간 사용)
2. **설정 복잡**: 두 모델 (SAM 2 + SAM 3D) 관리
3. **코드 복잡**: web_app.py 1300+ 줄
4. **Image folder 미지원**: 비디오만 지원
5. **모델 고정**: SAM 2.1 Hiera Large만

**최적 Use Case**:
- **3D 객체 재구성**
- **비디오 tracking** (propagation)
- **대량 프레임 처리** (> 100 frames)
- **복잡한 annotation 프로젝트**

---

## 4. 통합 계획 (Integration Plan)

### 4.1 목표

**"Best of Both Worlds"**: 두 도구의 장점을 결합한 통합 GUI

```
SAM Unified Annotator
├── Tab 1: 🎨 Interactive Mode (기존 SAM 3D GUI)
├── Tab 2: 🚀 Quick Mode (기존 SAM 3D GUI)
└── Tab 3: 📝 Lite Annotator (SAM Annotator 통합) ← NEW
```

---

### 4.2 Tab 3: Lite Annotator (통합 기능)

#### 통합할 SAM Annotator 기능

**Phase 1: Core Features (필수)**
- ✅ Direct video loading (cv2.VideoCapture)
- ✅ Image folder support (glob pattern)
- ✅ Runtime input source switching
- ✅ Multi-model support (4 SAM variants)
- ✅ Automatic annotation restoration

**Phase 2: Configuration (선택)**
- ✅ Hydra config integration (optional)
- ✅ CLI override support

**Phase 3: Advanced (선택)**
- ⚠️ Point size slider (이미 있음)
- ⚠️ Configurable visualization (이미 있음)

---

### 4.3 구현 방안

#### Option A: 새 탭 추가 (권장)

```python
# src/web_app.py

with gr.Tabs():
    # 기존 탭들
    with gr.Tab("🎨 Interactive Mode"):
        # 현재 Interactive Mode (SAM 2 + SAM 3D)
        pass

    with gr.Tab("🚀 Quick Mode"):
        # 현재 Quick Mode
        pass

    # NEW: SAM Annotator 통합
    with gr.Tab("📝 Lite Annotator"):
        with gr.Row():
            with gr.Column():
                # Left: Input & Frame
                input_source = gr.Textbox(label="Video/Image Folder")
                input_type = gr.Radio(["Video", "Image Folder"], value="Video")
                pattern_input = gr.Textbox(label="Pattern (for images)", value="*.png")
                load_source_btn = gr.Button("Load Source")

                frame_display = gr.Image(label="Frame")
                frame_slider = gr.Slider(label="Frame", minimum=0, maximum=100)

            with gr.Column():
                # Right: Controls & Mask
                point_type = gr.Radio(["Foreground", "Background"], value="Foreground")
                model_select = gr.Dropdown(
                    ["tiny", "small", "base+", "large"],
                    value="large",
                    label="SAM Model"
                )

                generate_btn = gr.Button("Generate Mask")
                save_btn = gr.Button("Save Annotation")
                clear_btn = gr.Button("Clear Points")

                mask_display = gr.Image(label="Mask")
                status_text = gr.Textbox(label="Status")
```

**장점**:
- 기존 코드 영향 최소화
- 독립적 개발 가능
- 사용자가 상황에 맞게 탭 선택

**단점**:
- 코드 중복 가능성
- 탭 전환 시 state 공유 어려움

---

#### Option B: 통합 모드 (고급)

```python
# Unified interface with mode selection
mode = gr.Radio([
    "Interactive (3D)",
    "Quick (Auto)",
    "Lite (Efficient)"
], value="Interactive (3D)")

# Conditional UI based on mode
if mode == "Lite (Efficient)":
    # SAM Annotator features
    pass
```

**장점**:
- UI 일관성
- 코드 재사용
- State 공유 용이

**단점**:
- 복잡한 구현
- 기존 코드 대폭 수정

---

### 4.4 Feature Mapping (기능 매핑)

| SAM Annotator 기능 | SAM 3D GUI 구현 | 통합 방법 |
|-------------------|----------------|----------|
| **Direct video load** | Frame extraction | Add cv2.VideoCapture mode |
| **Image folder** | Video only | Add glob pattern support |
| **Multi-model** | Single model | Add model selector dropdown |
| **Auto-restore** | Manual load | Add auto-load on frame change |
| **Runtime switch** | Fixed at start | Add input source changer |
| **Hydra config** | OmegaConf only | Optional Hydra integration |

---

### 4.5 구현 우선순위

#### P0 (Critical - 즉시 구현)
1. **Direct video loading** (메모리 효율)
2. **Image folder support** (유연성)
3. **Multi-model selection** (속도 vs 품질)

#### P1 (High - 1주 내)
4. **Automatic annotation restoration**
5. **Runtime input source switching**

#### P2 (Medium - 2주 내)
6. **Hydra config integration** (optional)

#### P3 (Low - 추후)
7. **CLI mode for Lite Annotator**

---

## 5. 코드 재사용 전략

### 5.1 공통 모듈 추출

```python
# src/annotator_core.py (NEW)

class BaseAnnotator:
    """SAM Annotator와 SAM 3D GUI 공통 로직"""

    def __init__(self, model_type="large", device="auto"):
        self.model = self._load_sam_model(model_type, device)
        self.points = []
        self.labels = []

    def add_point(self, x, y, label):
        """Add annotation point"""
        self.points.append([x, y])
        self.labels.append(label)

    def generate_mask(self):
        """Generate mask from points"""
        return self.model.predict(self.points, self.labels)

    def clear_points(self):
        """Clear all points"""
        self.points = []
        self.labels = []


class VideoLoader:
    """Direct video loading (SAM Annotator style)"""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, frame_idx):
        """Load single frame on-demand"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None


class ImageFolderLoader:
    """Image folder support (SAM Annotator style)"""

    def __init__(self, folder_path, pattern="*.png"):
        self.frames = sorted(glob.glob(os.path.join(folder_path, pattern)))
        self.total_frames = len(self.frames)

    def get_frame(self, frame_idx):
        """Load image from folder"""
        return cv2.imread(self.frames[frame_idx])
```

### 5.2 통합 Interface

```python
# src/web_app.py (Modified)

class SAMUnifiedGUI:
    def __init__(self):
        self.mode = "interactive"  # interactive, quick, lite
        self.annotator = BaseAnnotator()
        self.loader = None  # VideoLoader or ImageFolderLoader

    def switch_mode(self, mode):
        """Switch between modes"""
        self.mode = mode
        # Update UI accordingly

    def load_input(self, path, input_type):
        """Universal input loader"""
        if input_type == "video":
            self.loader = VideoLoader(path)
        elif input_type == "image_folder":
            self.loader = ImageFolderLoader(path)
        else:
            # Extract frames (existing method)
            self.loader = ExtractedFramesLoader(path)
```

---

## 6. 마이그레이션 가이드

### 6.1 기존 SAM Annotator 사용자를 위한 가이드

**Before (SAM Annotator)**:
```bash
python -m sam_annotator \
  --input /data/video.mp4 \
  --output ./annotations \
  --model large
```

**After (SAM Unified GUI - Lite Mode)**:
```bash
cd /home/joon/dev/sam3d_gui
./run.sh

# In GUI:
# 1. Select "📝 Lite Annotator" tab
# 2. Input: /data/video.mp4
# 3. Model: large
# 4. Click "Load Source"
# 5. Annotate as usual
```

### 6.2 기존 SAM 3D GUI 사용자

**변화 없음**: Interactive Mode와 Quick Mode는 그대로 유지
**추가 옵션**: Lite Mode로 더 빠른 annotation 가능

---

## 7. 문서 통합 계획 (다음 섹션에서 다룸)

이 내용은 "3. 문서 통합" 섹션에서 다룹니다.

---

**작성일**: 2025-11-24  
**버전**: 1.0  
**상태**: ✅ 분석 완료, 통합 계획 수립

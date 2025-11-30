# SAM 3D Mesh 생성 파라미터 가이드

SAM 3D Objects를 사용한 3D 메시 생성 시 조정 가능한 파라미터와 한계점을 설명합니다.

---

## 아키텍처 개요

SAM 3D Objects는 2단계 Diffusion 기반 3D 재구성 파이프라인입니다:

```
Input Image + Mask
       ↓
[Stage 1] Sparse Structure Sampling (ss)
  - 3D 구조의 희소 표현 생성
  - Diffusion 기반 voxel occupancy 예측
       ↓
[Stage 2] Structured Latent (slat) Generation
  - 상세 latent feature 생성
  - Diffusion 기반 feature refinement
       ↓
[Decoder] FlexiCubes Mesh Extraction
  - SDF → Mesh 변환
  - 동적 topology 생성
       ↓
[Post-processing] (선택)
  - Mesh 단순화
  - Hole filling
  - Texture baking
       ↓
Output: PLY/GLB Mesh
```

---

## 조정 가능한 파라미터

### 1. Inference 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `seed` | 42 | int | 랜덤 시드 (재현성 보장) |
| `stage1_inference_steps` | 25 | 5-50 | Stage 1 diffusion steps |
| `stage2_inference_steps` | 25 | 5-50 | Stage 2 diffusion steps |

**영향**:
- Steps 증가 → 품질 향상, 시간 증가
- Steps 감소 → 속도 향상, 품질 저하
- 권장: 빠른 미리보기 10, 일반 25, 고품질 50

### 2. Post-processing 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `with_mesh_postprocess` | False | bool | 후처리 활성화 |
| `simplify_ratio` | 0.95 | 0.5-0.99 | Face 유지 비율 |
| `texture_size` | 1024 | 512-2048 | 텍스처 해상도 |
| `with_texture_baking` | False | bool | 텍스처 베이킹 |
| `use_vertex_color` | True | bool | 버텍스 컬러 사용 |

**영향**:
- `simplify_ratio=0.95`: 원본의 95% face 유지 (5% 제거)
- `simplify_ratio=0.80`: 원본의 80% face 유지 (더 단순화)
- `texture_size`: 클수록 선명, 파일 크기 증가

### 3. 내부 고정 파라미터 (현재 수정 불가)

| 파라미터 | 값 | 위치 |
|---------|-----|------|
| `ss_cfg_strength` | 7 | Classifier-free guidance |
| `slat_cfg_strength` | 5 | Classifier-free guidance |
| `resolution` | 64 | Base grid resolution |
| `qef_reg_scale` | 1e-3 | FlexiCubes smoothing |

---

## 프레임 간 일관성 한계

### 문제점

SAM 3D Objects는 **단일 이미지 기반 모델**로 설계되어 있어:

1. **Topology 가변**: 각 이미지마다 다른 vertex/face 구조 생성
2. **Vertex 수 불일치**: 이미지 콘텐츠에 따라 자동 결정
3. **Correspondence 없음**: 프레임 간 정점 대응 관계 없음

### 동일 객체의 다른 프레임에서 발생하는 차이

```
Frame 1: 12,543 vertices, 24,892 faces
Frame 2: 11,987 vertices, 23,654 faces  ← 다름!
Frame 3: 12,891 vertices, 25,234 faces  ← 다름!
```

### 일관성 확보 방법

| 방법 | 효과 | 구현 | 권장 |
|------|------|------|------|
| **동일 Seed 사용** | 확률적 변동 최소화 | GUI 지원 | ⭐ 권장 |
| **Post-processing 통일** | Face 수 유사화 | GUI 지원 | ⭐ 권장 |
| **대표 프레임만 사용** | 1개 메시로 통일 | 수동 선택 | ⭐ 권장 |
| **Template Fitting** | 기준 메시에 정합 | 별도 구현 필요 | 고급 |
| **Remeshing** | 동일 topology 재생성 | 별도 도구 필요 | 고급 |

---

## 사용 시나리오별 권장 설정

### 빠른 미리보기 (1-2분)

```yaml
seed: 42
stage1_inference_steps: 10
stage2_inference_steps: 10
with_mesh_postprocess: false
with_texture_baking: false
use_vertex_color: true
```

### 일반 품질 (3-5분)

```yaml
seed: 42
stage1_inference_steps: 25
stage2_inference_steps: 25
with_mesh_postprocess: true
simplify_ratio: 0.90
with_texture_baking: false
use_vertex_color: true
```

### 고품질 (10-15분)

```yaml
seed: 42
stage1_inference_steps: 50
stage2_inference_steps: 50
with_mesh_postprocess: true
simplify_ratio: 0.85
texture_size: 2048
with_texture_baking: true
use_vertex_color: false
```

### 메모리 제한 환경 (RTX 3060 12GB)

```yaml
seed: 42
stage1_inference_steps: 15
stage2_inference_steps: 15
with_mesh_postprocess: true
simplify_ratio: 0.95
texture_size: 512
with_texture_baking: false
use_vertex_color: true
```

---

## 설정 저장 형식

메시 생성 시 설정은 JSON 파일로 함께 저장됩니다:

```
outputs/3d_meshes/{session_name}/
├── video001_frame0015_143022.ply
├── video001_frame0015_143022_settings.json  ← 설정 파일
└── ...
```

### settings.json 예시

```json
{
  "timestamp": "2025-11-30T14:30:22",
  "source": {
    "session_name": "mouse_batch_20251128",
    "video_name": "video_001",
    "frame_idx": 15
  },
  "parameters": {
    "seed": 42,
    "stage1_inference_steps": 25,
    "stage2_inference_steps": 25,
    "with_mesh_postprocess": true,
    "simplify_ratio": 0.90,
    "texture_size": 1024,
    "with_texture_baking": false,
    "use_vertex_color": true
  },
  "output": {
    "filename": "video001_frame0015_143022.ply",
    "format": "ply"
  }
}
```

---

## GUI 파라미터 조정

### Interactive Mode

1. **3D Mesh 섹션**에서 파라미터 조정
2. 슬라이더/체크박스로 설정
3. "Generate 3D Mesh" 클릭

### Batch Mode

1. **Preview 섹션**에서 파라미터 조정
2. 개별 프레임: "현재 프레임 3D Mesh"
3. 일괄 처리: "전체 비디오 3D Mesh"

---

## 주의사항

1. **GPU 메모리**: 고품질 설정은 더 많은 VRAM 필요
2. **처리 시간**: inference_steps × 2가 시간에 비례
3. **Texture Baking**: 추가 시간 필요 (1-2분)
4. **일관성**: 동일 객체의 여러 프레임 처리 시 동일 seed 권장

---

## 참고 자료

- SAM 3D Objects 논문: [Meta AI Research]
- FlexiCubes: [NVIDIA Research]
- 프로젝트 소스: `external/sam-3d-objects/`


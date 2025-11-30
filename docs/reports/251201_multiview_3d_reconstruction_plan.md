# Multi-View 3D Reconstruction for Mouse Behavioral Analysis

**날짜**: 2025-12-01
**주제**: 6-Camera Multi-View 기반 생쥐 3D 재구성 및 행동 분석 파이프라인
**목적**: SAM3D GUI에 multi-view 활용 정교화 기능 추가

---

## Executive Summary

6대 동기화 카메라로 촬영된 생쥐 영상에서 3D 재구성 및 행동 분석을 위한 파이프라인 설계.
**우선순위**: 복잡도와 정확도를 고려하여 단계적 구현.

---

## 1. 방법론 비교 분석

### 1.1 방법별 특성 비교표

| 방법 | Mesh 품질 | 속도 | 구현 난이도 | 동적 객체 | Multi-view | 최적 용도 |
|------|----------|------|------------|----------|------------|----------|
| **DeepLabCut + Anipose** | Skeleton만 | ★★★★ | Easy | ★★★★ | 6+ cam | 행동 분석 |
| **Visual Hull** | ★★ | ★★★★ | Easy | ★★ | 필수 | 빠른 실루엣 기반 |
| **SMAL Fitting** | ★★★ | ★★★★ | Medium | ★★★★ | 지원 | 해부학적 정확도 |
| **Non-rigid ICP** | ★★★★ | ★★★ | Medium | ★★★★ | 지원 | 표면 정밀화 |
| **COLMAP MVS** | ★★★★ | ★★★ | Medium | ✗ | 필수 | 정적 고품질 |
| **D-NeRF** | ★★★ | ★★ | High | ★★★★ | 필수 | 신경 렌더링 |
| **4D-GS** | ★★★ | ★★★★ | High | ★★★★ | 필수 | 실시간 렌더링 |

### 1.2 권장 우선순위

```
Phase 1: Keypoint 기반 (1-2주)
├── DeepLabCut 2D keypoint 검출
└── Anipose 6-camera triangulation

Phase 2: Template Deformation (2-4주)
├── SAM-3D single-view mesh → Template
├── Silhouette-based deformation
└── Non-rigid ICP refinement

Phase 3: Multi-view Fusion (3-4주)
├── Multi-view SAM-3D mesh fusion (ICP 정렬)
└── Visual Hull + Photoconsistency

Phase 4: Neural Rendering (Optional, 4-8주)
├── D-NeRF for dynamic scenes
└── 4D Gaussian Splatting
```

---

## 2. Phase 1: Keypoint 기반 3D Triangulation

### 2.1 이론적 배경

**DeepLabCut**: ResNet 기반 pose estimation
- 사용자 정의 keypoint (코, 귀, 척추, 사지, 꼬리 등)
- ~100 프레임 라벨링으로 훈련 가능
- 실시간 추론 (GPU 기준 30-100 fps)

**Anipose**: Multi-camera 3D triangulation
- ChArUco 보드 기반 카메라 캘리브레이션
- Robust triangulation with outlier rejection
- 6+ 카메라 최적화

### 2.2 장단점

**장점**:
- 가장 빠른 구현 (1-2주)
- 검증된 생쥐 행동 분석 파이프라인
- 실시간 처리 가능
- Occlusion에 강건

**단점**:
- Mesh 없음 (skeleton만)
- 수동 keypoint 정의 필요
- 표면 형상 정보 없음

### 2.3 핵심 논문

- Mathis et al. "DeepLabCut" (Nature Protocols 2019)
- Lauer et al. "Multi-animal DeepLabCut" (Nature Methods 2022)
- Karashchuk et al. "Anipose" (ICRA 2021)

---

## 3. Phase 2: Template Mesh Deformation

### 3.1 이론적 배경

**Template Fitting 워크플로우**:
```
1. Reference frame에서 SAM-3D로 template mesh 생성
2. 각 프레임의 2D mask 추출 (SAM2)
3. Silhouette 기반 mesh deformation
4. Non-rigid ICP로 표면 정밀화
```

**Keypoint 활용 방법**:
- DeepLabCut keypoint → SMAL skeleton에 매핑
- 2D keypoint reprojection error 최소화
- Skeleton joint angles 추정 → mesh deformation

**Non-rigid ICP 알고리즘**:
```python
# Pseudo-code
for iteration in range(max_iter):
    # 1. Correspondence: template vertex → target point
    correspondences = find_nearest_neighbors(template, target)

    # 2. Non-rigid transform: local + global deformation
    transform = compute_deformation(correspondences, stiffness)

    # 3. Update template
    template = apply_transform(template, transform)

    # 4. Reduce stiffness (coarse → fine)
    stiffness *= decay_factor
```

### 3.2 Silhouette 기반 Deformation

**이론**:
- Multi-view silhouette 일치 최적화
- Energy function: E = E_silhouette + E_regularization
- E_silhouette: IoU(rendered_mask, target_mask)
- E_regularization: Laplacian smoothness, ARAP

**장점**:
- Texture 불필요 (bare skin 동물에 적합)
- 기하학적 정확도 보장
- Keypoint 없이도 작동

**단점**:
- Concave 구조 재구성 한계
- Silhouette 추출 품질에 민감

### 3.3 SMAL (Skinned Multi-Animal Linear Model)

**한계**: 원본 SMAL은 생쥐 미포함 (대형 사족동물 기반)

**대안**:
1. Transfer learning (큰 동물 → 생쥐)
2. RatBodyFormer (2024, 쥐 특화 모델)
3. 생쥐 CT/3D 스캔으로 fine-tuning

### 3.4 핵심 논문

- Zuffi et al. "3D Menagerie: SMAL" (CVPR 2017)
- Amberg et al. "Optimal Step Non-rigid ICP" (2007)
- Li & Lee "Coarse-to-fine Animal Pose" (NeurIPS 2021)

---

## 4. Phase 3: Multi-view Mesh Fusion

### 4.1 방법 A: SAM-3D Multi-view Fusion

**워크플로우**:
```python
# 1. 각 뷰에서 SAM-3D mesh 생성
meshes = []
for view_idx, (image, mask) in enumerate(multi_view_data):
    mesh = sam3d.reconstruct_3d(image, mask)
    meshes.append(mesh)

# 2. ICP로 mesh 정렬
reference_mesh = meshes[0]
aligned_meshes = [reference_mesh]
for mesh in meshes[1:]:
    aligned = icp_registration(mesh, reference_mesh)
    aligned_meshes.append(aligned)

# 3. Poisson fusion
fused_mesh = poisson_reconstruction(aligned_meshes)
```

**장점**:
- 기존 SAM-3D 파이프라인 활용
- 구현 상대적 간단

**단점**:
- 각 뷰 mesh 품질에 의존
- ICP 정렬 오류 누적 가능

### 4.2 방법 B: Visual Hull + Photoconsistency

**이론**:
```
1. 각 카메라에서 silhouette 추출
2. Silhouette → 3D cone 역투영
3. 6개 cone 교집합 = Visual Hull
4. Photoconsistency로 정밀화
```

**장점**:
- 텍스처 불필요
- GPU 가속으로 실시간 가능
- 캘리브레이션만 있으면 동작

**단점**:
- Concave 구조 불가
- 카메라 수에 품질 의존 (6개면 양호)

### 4.3 방법 C: COLMAP MVS (정적 기준)

**용도**: 특정 프레임의 고품질 정적 재구성
- 행동 분석보다는 형태 참조용
- 안락사 후 표본이나 정지 자세에 적합

---

## 5. Phase 4: Neural Rendering (선택적)

### 5.1 D-NeRF

**이론**:
- Canonical NeRF + Deformation Network
- 시간 t에 따른 변형장 학습
- Marching cubes로 mesh 추출

**장점**:
- 고품질 novel view synthesis
- 부드러운 temporal interpolation

**단점**:
- 훈련 12-48시간
- Mesh 추출 품질 불안정

### 5.2 4D Gaussian Splatting

**이론**:
- 3D Gaussian primitives + temporal deformation
- HexPlane 기반 4D 분해

**장점**:
- 실시간 렌더링 (80+ fps)
- 빠른 훈련 (D-NeRF 대비 10x)

**단점**:
- 최신 기술 (2024), 안정성 미검증
- Mesh 추출 비표준

---

## 6. 구현 계획

### 6.1 모듈 구조

```
src/
├── reconstruction/
│   ├── __init__.py
│   ├── camera_calibration.py    # 카메라 캘리브레이션
│   ├── keypoint_triangulation.py # DeepLabCut + Anipose
│   ├── template_deformation.py   # Non-rigid ICP, Silhouette fitting
│   ├── multiview_fusion.py       # ICP mesh fusion, Visual Hull
│   └── neural_rendering.py       # D-NeRF, 4D-GS (optional)
```

### 6.2 단계별 구현 일정

| Phase | 모듈 | 예상 기간 | 의존성 |
|-------|------|----------|--------|
| 1 | camera_calibration | 3일 | OpenCV, NumPy |
| 1 | keypoint_triangulation | 1주 | DeepLabCut, Anipose |
| 2 | template_deformation | 2주 | PyTorch3D, Open3D |
| 3 | multiview_fusion | 2주 | Open3D, Trimesh |
| 4 | neural_rendering | 4주+ | PyTorch, CUDA |

### 6.3 필요 라이브러리

```bash
# Phase 1
pip install deeplabcut anipose

# Phase 2-3
pip install pytorch3d open3d trimesh

# Phase 4 (Optional)
pip install nerfstudio  # or custom D-NeRF
```

---

## 7. 기술적 요구사항

### 7.1 카메라 캘리브레이션

**필수**:
- Intrinsic: focal length, principal point, distortion
- Extrinsic: camera-to-camera pose (R, t)

**도구**:
- ChArUco board (Anipose 표준)
- MC-Calib (multi-camera toolbox)
- COLMAP (자동 SfM 기반)

### 7.2 시간 동기화

**중요도**: Critical (생쥐 빠른 움직임)

**방법**:
- Hardware sync (GigE camera trigger)
- Software sync (LED flash, clap detection)
- 허용 오차: ±1-2 프레임

### 7.3 조명

- 균일하고 그림자 없는 조명
- DeepLabCut + MVS 모두에 중요

---

## 8. 예상 결과물

| Phase | 출력 | 용도 |
|-------|------|------|
| 1 | 3D keypoint trajectories | 행동 분석, 운동학 |
| 2 | Deformed mesh sequence | 형태 변화 추적 |
| 3 | High-quality fused mesh | 정밀 형태 분석 |
| 4 | Novel view synthesis | 시각화, 발표 |

---

## 9. 핵심 참고자료

### 논문
- DeepLabCut (Nature Protocols 2019)
- Anipose (ICRA 2021)
- SMAL (CVPR 2017)
- D-NeRF (CVPR 2021)
- 4D Gaussian Splatting (CVPR 2024)

### 오픈소스
- https://github.com/DeepLabCut/DeepLabCut
- https://github.com/lambdaloop/anipose
- https://github.com/silviazuffi/SMAL
- https://github.com/albertpumarola/D-NeRF
- https://github.com/hustvl/4DGaussians

---

## 10. 다음 단계

1. **즉시**: Phase 1 모듈 구조 생성
2. **이번 주**: camera_calibration.py 구현
3. **다음 주**: template_deformation.py 기본 구현
4. **이후**: 실제 데이터로 테스트 및 정밀화

---

*작성: Claude Code*
*프로젝트: SAM3D GUI Multi-view Extension*

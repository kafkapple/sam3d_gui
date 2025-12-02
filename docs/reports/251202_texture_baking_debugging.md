# Texture Baking Segfault 디버깅 보고서

**날짜**: 2025-12-02
**상태**: 진행 중 (부분 해결)
**환경**: gpu05 (NVIDIA RTX A6000), Ubuntu, CUDA 11.8, PyTorch 2.0.0

---

## Executive Summary

SAM3D GUI의 Texture Baking 기능에서 Segfault가 발생하는 문제를 디버깅. CUDA 환경 설정, 메시 단순화, float32 패치 등을 통해 상당한 진전을 이루었으나, Gradio 웹 서버 환경에서의 nvdiffrast 호출 시 여전히 Segfault 발생.

**핵심 발견**: 독립 Python 스크립트에서는 nvdiffrast가 정상 작동하지만, Gradio 웹 서버 내에서 호출 시 Segfault 발생.

---

## 1. 문제 상황

### 1.1 증상
- Texture Baking 활성화 시 서버 Segfault (exit code 139)
- `bake_texture()` 함수의 UV rasterization 단계에서 crash
- 로그: `Texture baking (opt): UV: 0%` 또는 `6%`에서 종료

### 1.2 영향 범위
- Texture Baking 기능 사용 불가
- 서버 전체 crash로 인한 서비스 중단

---

## 2. 원인 분석

### 2.1 CUDA 버전 불일치 (해결됨)

**문제**: 시스템에 여러 CUDA 버전 존재
```
/usr/bin/nvcc → CUDA 9.1 (시스템 기본)
/usr/local/cuda-11.8 → PyTorch와 호환되는 버전
/usr/local/cuda-12.x → 최신 버전
```

nvdiffrast JIT 컴파일 시 `/usr/bin/nvcc` (CUDA 9.1) 사용 → PyTorch (CUDA 11.8)와 불일치로 Segfault

**해결**:
```bash
# run.sh에 추가
if [[ -d "/usr/local/cuda-11.8" ]]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
fi
```

### 2.2 conda run 환경변수 미상속 (해결됨)

**문제**: `conda run`은 부모 shell의 환경변수를 상속하지 않음

**해결**:
```bash
# Before
conda run -n sam3d_gui python src/web_app.py

# After
eval "$(conda shell.bash hook)"
conda activate sam3d_gui
python src/web_app.py
```

### 2.3 대형 메시 (해결됨)

**문제**: 원본 메시가 100K+ vertices로 nvdiffrast에서 처리 불가

**해결**: Mesh simplify 적용
```python
# simplify_ratio 기본값 변경: 0.95 → 0.98
# 결과: 100K vertices → 2K vertices
simplify_ratio = 0.98  # 98% face 제거, 2%만 유지
```

### 2.4 fill_holes 내부 nvdiffrast 사용 (해결됨)

**문제**: `fill_holes=True`도 내부적으로 nvdiffrast 호출

**해결**:
```python
# inference_pipeline.py
glb = postprocessing_utils.to_glb(
    ...
    fill_holes=False,  # 비활성화
    ...
)
```

### 2.5 GL Backend float32 문제 (해결됨)

**문제**: GL backend는 float32 입력 필수, 행렬 연산 중 float64로 변환됨

**해결**: 3곳 패치
```python
# 1. utils3d/torch/rasterization.py (site-packages)
pos_clip = (vertices @ mvp.transpose(-1, -2)).float()

# 2. postprocessing_utils.py - dr.texture
render = dr.texture(texture.float(), uv.float(), uv_dr.float())[0]

# 3. postprocessing_utils.py - 최종 rasterize
rastctx, ((uvs * 2 - 1)[None]).float(), faces, ...
```

### 2.6 웹 서버 환경 문제 (미해결)

**문제**: Gradio 웹 서버 내에서만 nvdiffrast Segfault 발생

**테스트 결과**:
| 환경 | nvdiffrast 직접 | utils3d wrapper | Gradio 내 호출 |
|------|-----------------|-----------------|----------------|
| 결과 | ✅ 성공 | ✅ 성공 | ❌ Segfault |

**가능한 원인**:
- Gradio 멀티스레딩과 CUDA context 충돌
- 이전 CUDA 연산(SAM2, gsplat)과 nvdiffrast 간 충돌
- render_multiview(gsplat)와 bake_texture(nvdiffrast) 간 context 충돌

---

## 3. 적용된 변경사항

### 3.1 run.sh
```bash
# CUDA 환경 설정 추가
if [[ -d "/usr/local/cuda-11.8" ]]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
fi

# conda activate 방식으로 변경
eval "$(conda shell.bash hook)"
conda activate sam3d_gui
```

### 3.2 src/web_app.py
```python
# simplify_ratio 기본값 변경
simplify_ratio: float = 0.98  # 0.95 → 0.98
```

### 3.3 inference_pipeline.py
```python
glb = postprocessing_utils.to_glb(
    ...
    fill_holes=False,  # nvdiffrast Segfault 방지
    verbose=True,      # 메시 크기 로깅
    ...
)
```

### 3.4 postprocessing_utils.py (gpu05 직접 수정)
```python
# Backend 변경 (GL → CUDA, 여러 번 변경)
rastctx = utils3d.torch.RastContext(backend="cuda")

# CUDA 동기화 추가
torch.cuda.synchronize()
rastctx = utils3d.torch.RastContext(backend="cuda")

# float32 강제 변환
render = dr.texture(texture.float(), uv.float(), uv_dr.float())[0]

# 디버그 로깅 추가
print(f"DEBUG: bake_texture opt mode starting...", flush=True)
print(f"DEBUG: vertices shape={vertices.shape}, dtype={vertices.dtype}", flush=True)
```

### 3.5 utils3d/torch/rasterization.py (site-packages 패치)
```python
# Line 101, 210
pos_clip = (vertices @ mvp.transpose(-1, -2)).float()  # Force float32
```

---

## 4. 진행 상황

| # | 문제 | 해결책 | 상태 |
|---|------|--------|------|
| 1 | CUDA 9.1 사용 | CUDA_HOME 설정 | ✅ 해결 |
| 2 | conda run 환경변수 | conda activate 방식 | ✅ 해결 |
| 3 | 대형 메시 (224K) | simplify_ratio=0.98 | ✅ 해결 |
| 4 | fill_holes Segfault | fill_holes=False | ✅ 해결 |
| 5 | GL float32 오류 | .float() 패치 | ✅ 해결 |
| 6 | GL context 재사용 | 매 iteration 새 context | ❌ 악화 |
| 7 | Gradio 내 nvdiffrast | CUDA sync 추가 | ❌ 미해결 |

**진전 요약**:
- Segfault → Python Exception (디버깅 가능) → 다시 Segfault
- 224K vertices → 2.3K vertices (단순화 성공)
- 문제 위치 특정: `bake_texture()` UV rasterization loop

---

## 5. 권장 설정

### 5.1 현재 안정적인 설정 (Texture Baking 없이)
```
✅ Vertex Color: ON
❌ Texture Baking: OFF
❌ Mesh Postprocess: OFF (또는 fill_holes=False로 ON)
```

### 5.2 Texture Baking 시도 시 설정
```
✅ Texture Baking: ON
✅ Mesh Postprocess: ON
✅ Simplify Ratio: 0.98
✅ Texture Size: 512 (안전)
✅ Render Views: 16 (안전)
```

---

## 6. 추가 조사 필요 사항

### 6.1 근본 원인 파악
- [ ] Gradio 멀티스레딩 비활성화 테스트
- [ ] CUDA context 격리 방법 조사
- [ ] subprocess로 bake_texture 격리 실행 테스트

### 6.2 대안 검토
- [ ] pytorch3d로 UV rasterization 완전 대체
- [ ] Texture Baking을 별도 프로세스로 분리
- [ ] 다른 texture baking 라이브러리 검토

### 6.3 nvdiffrast 이슈 조사
- [ ] nvdiffrast GitHub issues 확인
- [ ] 웹 서버 환경에서의 사용 사례 조사
- [ ] nvdiffrast 버전 업그레이드/다운그레이드 테스트

---

## 7. 디버깅 명령어 참고

```bash
# CUDA 환경 확인
which nvcc && nvcc --version
python -c "import torch; print(torch.version.cuda)"

# nvdiffrast 캐시 삭제
rm -rf ~/.cache/torch_extensions/*/nvdiffrast*

# nvdiffrast 직접 테스트
python -c "
import torch
import nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext(device='cuda:0')
print('nvdiffrast OK')
"

# utils3d 테스트
python test_utils3d_cuda.py

# 서버 로그 확인
tail -f /tmp/sam3d_debug.log | grep -E 'DEBUG|Texture|UV|Segfault'
```

---

## 8. 관련 파일

| 파일 | 역할 |
|------|------|
| `run.sh` | CUDA 환경 설정, 서버 실행 |
| `src/web_app.py` | UI 설정, simplify_ratio 기본값 |
| `external/sam-3d-objects/.../inference_pipeline.py` | to_glb 호출, fill_holes 설정 |
| `external/sam-3d-objects/.../postprocessing_utils.py` | bake_texture, nvdiffrast 호출 |
| `~/.cache/torch_extensions/py310_cu118/` | nvdiffrast JIT 캐시 |

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-12-01 | CUDA 환경 설정, conda activate 방식 변경 |
| 2025-12-01 | simplify_ratio 기본값 0.98로 변경 |
| 2025-12-01 | fill_holes=False 설정 |
| 2025-12-01 | GL backend float32 패치 |
| 2025-12-02 | CUDA backend로 변경, sync 추가 |
| 2025-12-02 | 문서화 |

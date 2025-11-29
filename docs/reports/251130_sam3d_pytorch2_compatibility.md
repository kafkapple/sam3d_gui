# SAM 3D Objects - PyTorch 2.0 호환성 문제 해결 보고서

**날짜**: 2025-11-30
**환경**: Python 3.10, PyTorch 2.0.0+cu118, CUDA 11.8
**서버**: gpu05

---

## Executive Summary

SAM 3D Objects 모델(Meta)을 PyTorch 2.0 환경에서 실행할 때 발생하는 다양한 호환성 문제를 해결했습니다. 원본 코드는 PyTorch 2.1+ 기준으로 작성되어 있어 PyTorch 2.0에서는 여러 API 차이로 인한 오류가 발생합니다.

**해결 방식**: Fork 기반 관리 (`kafkapple/sam-3d-objects`)
- 런타임 패칭 대신 fork에 직접 수정
- Git submodule로 메인 레포와 연결
- `git pull --recurse-submodules`로 간편한 업데이트

---

## 오류 목록 및 해결 방법

### 1. `torch._dynamo` AttributeError

**오류 메시지**:
```
AttributeError: module 'torch' has no attribute '_dynamo'
```

**원인**: `torch._dynamo`는 PyTorch 2.0에서 실험적 기능으로 일부 환경에서 없을 수 있음

**해결**:
```python
# Before
torch._dynamo.config.cache_size_limit = 64

# After
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.cache_size_limit = 64
```

**수정 파일**:
- `sam3d_objects/pipeline/inference_pipeline.py`
- `sam3d_objects/pipeline/inference_pipeline_pointmap.py`

---

### 2. `torch.nn.attention` ModuleNotFoundError

**오류 메시지**:
```
ModuleNotFoundError: No module named 'torch.nn.attention'
```

**원인**: `torch.nn.attention` 모듈은 PyTorch 2.1+에서만 존재

**해결**:
```python
# PyTorch 버전 감지
TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])
TORCH_NN_ATTENTION_AVAILABLE = (TORCH_MAJOR > 2) or (TORCH_MAJOR == 2 and TORCH_MINOR >= 1)

if BACKEND == "torch_flash_attn":
    if not TORCH_NN_ATTENTION_AVAILABLE:
        print(f"Warning: torch_flash_attn requires PyTorch 2.1+, falling back to sdpa")
        BACKEND = "sdpa"
```

**수정 파일**:
- `sam3d_objects/model/backbone/tdfy_dit/modules/attention/full_attn.py`

---

### 3. Lightning `isinstance()` TypeError

**오류 메시지**:
```
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```

**원인**: Lightning mock 객체와 `isinstance()` 호출 충돌, Union 타입 힌트의 런타임 평가 문제

**해결**:
1. Lightning을 기본 비활성화하고 stub 클래스 사용
2. 타입 힌트에서 `pl.LightningModule` 제거
3. `isinstance` 호출을 `LIGHTNING_AVAILABLE` 조건으로 보호

```python
# Stub 클래스 정의
LIGHTNING_AVAILABLE = False

class _LightningModuleStub:
    pass

class _PLStub:
    LightningModule = _LightningModuleStub

pl = _PLStub()

# isinstance 호출 보호
if LIGHTNING_AVAILABLE and isinstance(model, pl.LightningModule):
    model.on_load_checkpoint(checkpoint)

# 타입 힌트 수정
def load_model_from_checkpoint(
    model: torch.nn.Module,  # Union[pl.LightningModule, torch.nn.Module] 제거
    ...
):
```

**수정 파일**:
- `sam3d_objects/model/io.py`

---

### 4. `tree_map()` 인자 개수 오류

**오류 메시지**:
```
tree_map() takes 2 positional arguments but 3 were given
```

**원인**: PyTorch 2.0의 `tree_map`은 2개 인자만 받음 (`fn`, `pytree`). PyTorch 2.1+는 여러 pytree 지원

**해결**: 호환성 래퍼 함수 생성

```python
# sam3d_objects/model/backbone/tdfy_dit/modules/pytree_compat.py
from torch.utils import _pytree

def tree_map_compat(fn, *pytrees):
    if len(pytrees) == 1:
        return _pytree.tree_map(fn, pytrees[0])
    elif len(pytrees) == 2:
        pt1, pt2 = pytrees
        if isinstance(pt1, dict) and isinstance(pt2, dict):
            return {k: fn(pt1[k], pt2[k]) for k in pt1.keys()}
        else:
            flat1, spec = _pytree.tree_flatten(pt1)
            flat2, _ = _pytree.tree_flatten(pt2)
            results = [fn(a, b) for a, b in zip(flat1, flat2)]
            return _pytree.tree_unflatten(results, spec)
    else:
        # 3+ pytrees
        if all(isinstance(pt, dict) for pt in pytrees):
            keys = pytrees[0].keys()
            return {k: fn(*[pt[k] for pt in pytrees]) for k in keys}
        else:
            flat_lists = [_pytree.tree_flatten(pt)[0] for pt in pytrees]
            spec = _pytree.tree_flatten(pytrees[0])[1]
            results = [fn(*args) for args in zip(*flat_lists)]
            return _pytree.tree_unflatten(results, spec)
```

**수정 파일**:
- `sam3d_objects/model/backbone/tdfy_dit/modules/pytree_compat.py` (신규)
- `sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py`
- `sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py`
- `sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py`
- `sam3d_objects/model/backbone/generator/classifier_free_guidance.py`

---

### 5. `torch.compiler` AttributeError

**오류 메시지**:
```
module 'torch' has no attribute 'compiler'
```

**원인**: `torch.compiler`는 PyTorch 2.1+에서만 존재

**해결**:
```python
# Before
if not torch.compiler.is_compiling():

# After
_is_compiling = getattr(torch, 'compiler', None) and torch.compiler.is_compiling()
if not _is_compiling:
```

**수정 파일**:
- `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py`

---

### 6. `index_add_()` dtype 불일치

**오류 메시지**:
```
index_add_(): self (Float) and source (Half) must have the same scalar type
```

**원인**: Mixed precision 연산에서 Float32와 Float16 텐서 간 `index_add_` 호출

**해결**:
```python
# Before
beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)

# After
beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group.to(beta_sum.dtype))
```

**수정 파일**:
- `sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py`

---

### 7. kaolin Import 오류

**오류 메시지**:
```
RuntimeError: Error loading warp.so
```

**원인**: kaolin의 warp 모듈 로딩 실패 (일부 시스템)

**해결**:
```python
# kaolin is optional
try:
    from kaolin.utils.testing import check_tensor
except (ImportError, RuntimeError, OSError):
    def check_tensor(tensor, *args, **kwargs):
        return tensor
```

**수정 파일**:
- `sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py`

---

## 환경 설정 관련

### 필수 환경 변수

```bash
# run.sh에 설정됨
export LIDRA_SKIP_INIT=1  # sam3d_objects.init 모듈 스킵
```

### Config 설정

`pipeline.yaml`에서 모델 컴파일 비활성화:
```yaml
compile_model: false  # PyTorch 2.0에서 torch.compile 호환성 문제
```

---

## Git Submodule 관리

### 최초 설정 (새 환경)
```bash
git clone https://github.com/kafkapple/sam3d_gui.git
cd sam3d_gui
git submodule update --init
```

### 평소 업데이트
```bash
git pull --recurse-submodules
```

### 충돌 발생 시 (수동 수정 파일이 남아있는 경우)
```bash
rm -rf external/sam-3d-objects
git submodule update --init
```

---

## 수정된 파일 목록 (Fork: kafkapple/sam-3d-objects)

| 파일 | 수정 내용 |
|------|----------|
| `model/io.py` | Lightning 비활성화, stub 클래스, isinstance 보호 |
| `pipeline/inference_pipeline.py` | `_dynamo` hasattr 체크 |
| `pipeline/inference_pipeline_pointmap.py` | `_dynamo` hasattr 체크 |
| `model/backbone/tdfy_dit/modules/attention/full_attn.py` | PyTorch 버전 감지, attention fallback |
| `model/backbone/tdfy_dit/modules/pytree_compat.py` | tree_map 호환성 래퍼 (신규) |
| `model/backbone/tdfy_dit/modules/transformer/modulated.py` | tree_map_compat 사용 |
| `model/backbone/tdfy_dit/modules/attention/modules.py` | tree_map_compat 사용 |
| `model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py` | tree_map_compat 사용 |
| `model/backbone/tdfy_dit/models/structured_latent_flow.py` | torch.compiler 체크 |
| `model/backbone/generator/classifier_free_guidance.py` | tree_map_compat 사용 |
| `model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py` | kaolin fallback, dtype 수정 |

---

## 교훈 및 Best Practices

### 1. PyTorch 버전 호환성 체크 패턴

```python
# 버전 감지
TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

# 모듈 존재 여부 체크
if hasattr(torch, 'module_name'):
    # 사용

# getattr로 안전하게 접근
_feature = getattr(torch, 'feature', None)
if _feature:
    _feature.do_something()
```

### 2. dtype 불일치 방지

```python
# index_add_, scatter_ 등에서 항상 dtype 일치시키기
target.index_add_(0, index, source.to(target.dtype))
```

### 3. 런타임 패칭 vs Fork

| 방식 | 장점 | 단점 |
|------|------|------|
| 런타임 패칭 | 원본 유지 | 복잡, 유지보수 어려움, 실패 위험 |
| Fork | 깔끔, 안정적, 버전 관리 | 업스트림 동기화 필요 |

**권장**: Fork 방식 (이번 케이스에서 채택)

### 4. 웹 앱 재시작 확인

코드 변경 후 반드시 웹 앱 재시작 필요:
- 이전 프로세스가 메모리에 캐시된 코드 사용
- `ps aux | grep web_app.py`로 확인
- 오래된 프로세스 종료 후 재시작

---

## 향후 참고사항

1. **PyTorch 2.1+ 업그레이드 시**: 대부분의 호환성 코드가 불필요해지지만, 하위 호환성 유지를 위해 남겨두는 것이 안전

2. **업스트림 변경사항 반영**: Fork에서 주기적으로 upstream merge 권장
   ```bash
   git remote add upstream https://github.com/facebookresearch/sam-3d-objects.git
   git fetch upstream
   git merge upstream/main
   ```

3. **새로운 오류 발생 시**: 이 문서의 패턴을 참고하여 유사하게 해결

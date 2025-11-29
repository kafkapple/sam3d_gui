# PyTorch 버전 호환성 가이드

외부 라이브러리를 다른 PyTorch 버전에서 사용할 때 발생하는 호환성 문제 해결 가이드입니다.

---

## 핵심 원칙

### 1. Fork 기반 관리 (권장)

```
원본 저장소 (facebookresearch/sam-3d-objects)
    ↓ fork
내 저장소 (kafkapple/sam-3d-objects)  ← 호환성 수정
    ↓ submodule
프로젝트 (sam3d_gui)
```

**장점:**
- 깔끔한 코드베이스
- 버전 관리 가능
- 팀 공유 용이
- 업스트림 변경사항 merge 가능

**vs 런타임 패칭:**
- 런타임 패칭: 복잡, 실패 위험, 유지보수 어려움
- Fork: 안정적, 명확, 추적 가능

### 2. 방어적 코딩 패턴

```python
# 모듈 존재 여부 체크
if hasattr(torch, 'module_name'):
    torch.module_name.do_something()

# getattr로 안전하게 접근
feature = getattr(torch, 'feature', None)
if feature:
    feature.method()

# 버전 감지
TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])
if TORCH_MAJOR >= 2 and TORCH_MINOR >= 1:
    # PyTorch 2.1+ 코드
else:
    # Fallback 코드
```

### 3. dtype 일관성 유지

```python
# 항상 대상 텐서의 dtype으로 변환
target.index_add_(0, index, source.to(target.dtype))
target.scatter_(0, index, source.to(target.dtype))
```

---

## 자주 발생하는 호환성 문제

### 1. 새로운 모듈/API 사용

**증상:** `AttributeError: module 'torch' has no attribute 'xxx'`

**원인:** 최신 PyTorch에만 존재하는 API 사용

**해결 패턴:**
```python
# 패턴 1: hasattr 체크
if hasattr(torch, 'new_feature'):
    torch.new_feature.use()
else:
    # fallback

# 패턴 2: try-except
try:
    from torch.new_module import NewClass
except ImportError:
    class NewClass:  # stub
        pass

# 패턴 3: getattr with default
compiler = getattr(torch, 'compiler', None)
if compiler and compiler.is_compiling():
    pass
```

### 2. 함수 시그니처 변경

**증상:** `TypeError: xxx() takes N positional arguments but M were given`

**원인:** API가 새 버전에서 인자 개수가 변경됨

**해결 패턴:**
```python
# 호환성 래퍼 함수 생성
def compat_function(*args, **kwargs):
    if NEW_VERSION:
        return new_function(*args, **kwargs)
    else:
        # 이전 버전용 구현
        return old_behavior(*args, **kwargs)
```

### 3. 타입 힌트 런타임 평가

**증상:** `TypeError: isinstance() arg 2 must be a type`

**원인:** Union 타입이나 mock 객체가 런타임에 평가될 때 문제

**해결 패턴:**
```python
# 패턴 1: 타입 힌트 단순화
def func(model: torch.nn.Module):  # Union 제거
    pass

# 패턴 2: 조건부 isinstance
if FEATURE_AVAILABLE and isinstance(obj, SomeClass):
    pass

# 패턴 3: 문자열 타입 힌트 (지연 평가)
def func(model: "Union[A, B]"):
    pass
```

### 4. Mixed Precision dtype 불일치

**증상:** `RuntimeError: xxx (Float) and yyy (Half) must have the same scalar type`

**원인:** FP16/FP32 혼합 연산에서 일부 연산이 dtype 불일치를 허용하지 않음

**해결 패턴:**
```python
# 항상 대상의 dtype으로 변환
result = target.index_add_(0, idx, source.to(target.dtype))
result = target.scatter_add_(0, idx, src.to(target.dtype))
```

---

## 문제 해결 워크플로우

### Step 1: 오류 분석

```bash
# 오류 메시지에서 핵심 정보 추출
# 1. 어떤 모듈/함수?
# 2. 어떤 파일, 몇 번째 라인?
# 3. PyTorch 버전 관련?
```

### Step 2: 원인 파악

```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# 해당 API가 언제 도입되었는지 확인
# PyTorch 릴리즈 노트: https://pytorch.org/docs/stable/notes/changelog.html
```

### Step 3: 해결책 적용

1. **Fork에서 직접 수정** (권장)
2. 호환성 래퍼 작성
3. Fallback 구현

### Step 4: 테스트

```bash
# 단위 테스트
python -c "from module import function; function()"

# 통합 테스트
./run.sh  # 웹 앱 재시작 후 기능 테스트
```

### Step 5: 문서화

- 오류 메시지 기록
- 원인 분석
- 해결 방법
- 수정된 파일 목록

---

## 체크리스트

### 새 외부 라이브러리 도입 시

- [ ] 지원하는 PyTorch 버전 확인
- [ ] 현재 환경의 PyTorch 버전과 비교
- [ ] 호환되지 않으면 fork 고려
- [ ] 필요한 수정사항 목록화

### 코드 수정 시

- [ ] `hasattr()` 또는 `try-except`로 보호
- [ ] dtype 변환 명시적으로 처리
- [ ] Fallback 로직 구현
- [ ] 테스트 후 커밋

### 배포 시

- [ ] 서브모듈 최신 상태 확인
- [ ] 환경 변수 설정 확인
- [ ] 웹 앱 재시작 확인
- [ ] 기능 동작 테스트

---

## 참고 자료

- **PyTorch 릴리즈 노트**: https://pytorch.org/docs/stable/notes/changelog.html
- **PyTorch 버전별 API 문서**: https://pytorch.org/docs/
- **SAM 3D 호환성 보고서**: `docs/reports/251130_sam3d_pytorch2_compatibility.md`

---

## 요약

| 상황 | 권장 방법 |
|------|----------|
| 외부 라이브러리 호환성 문제 | Fork 후 직접 수정 |
| 새 API 사용 | `hasattr()` 또는 버전 체크 |
| 함수 시그니처 변경 | 호환성 래퍼 함수 |
| dtype 불일치 | `.to(target.dtype)` 변환 |
| 타입 힌트 문제 | Union 제거 또는 조건부 체크 |

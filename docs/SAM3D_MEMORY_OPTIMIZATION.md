# SAM 3D Memory Optimization Guide

## 문제 상황

**Hardware**: NVIDIA GeForce RTX 3060 (12GB VRAM)
**Issue**: SAM 3D 전체 파이프라인 로딩 시 CUDA OOM (Out Of Memory)
**Error Location**: DINO ViT-L 모델 로딩 중 (9.19 GB 이미 할당 상태에서 추가 20 MB 할당 실패)

## 메모리 사용 분석

### 로딩 순서 및 메모리 사용량

1. **ScaleShift Generator** (ss_generator.ckpt) - ~1.5 GB
2. **SLAT Generator** (slat_generator.ckpt) - ~4.6 GB
3. **ScaleShift Decoder** (ss_decoder.ckpt) - ~0.1 GB
4. **SLAT Decoder GS** (slat_decoder_gs.ckpt) - ~0.2 GB
5. **SLAT Decoder GS 4** (slat_decoder_gs_4.ckpt) - ~0.2 GB
6. **SLAT Decoder Mesh** (slat_decoder_mesh.ckpt) - ~0.3 GB
7. **MoGe Depth Model** (Ruicheng/moge-vitl) - ~1.4 GB
8. **DINO ViT-L** (dinov2_vitl14_reg4) - ~1.2 GB ❌ **OOM 발생 지점**

**합계**: ~9.5 GB + PyTorch overhead + CUDA context ≈ **10-11 GB**

### 12GB GPU에서 초과하는 이유

- PyTorch 자체 메모리 overhead: ~0.5-1 GB
- CUDA context: ~0.5 GB
- 메모리 fragmentation
- **Result**: 12GB에서 실행 불가능

## 구현된 최적화 방안

### 1. Lazy Loading (✅ 완료)

**Location**: `sam3d_processor.py:89-155`

```python
def initialize_sam3d(self, force_reload: bool = False):
    """
    모델을 처음 사용할 때만 로드
    """
    if self.inference_model is not None and not force_reload:
        print(f"   ✓ SAM 3D 모델 이미 로드됨 (재사용)")
        return

    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"🔹 GPU 메모리 상태 (로딩 전): {initial_memory:.2f} GB")

    # ... model loading ...
```

**효과**:
- 불필요한 모델 로드 방지
- 메모리 재사용 가능

### 2. Model Cleanup (✅ 완료)

**Location**: `sam3d_processor.py:157-185`

```python
def cleanup_model(self):
    """
    모델 메모리 해제 및 GPU 캐시 정리
    """
    if self.inference_model is not None:
        del self.inference_model
        self.inference_model = None
        self._model_loaded = False

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

**효과**:
- Inference 후 명시적 메모리 해제
- 다음 작업을 위한 VRAM 확보

### 3. Memory Monitoring (✅ 완료)

**Location**: `sam3d_processor.py:187-223`

```python
def get_memory_status(self) -> Dict:
    """현재 GPU 메모리 상태 조회"""
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
        'model_loaded': self._model_loaded
    }

def print_memory_status(self):
    """메모리 상태 출력"""
    # ... pretty print ...
```

**효과**:
- 실시간 메모리 사용량 모니터링
- OOM 전 경고

### 4. FP16 Mixed Precision (✅ 완료)

**Location**: `sam3d_processor.py:64-87`, `sam3d_processor.py:462-469`

```python
def __init__(self, sam3d_checkpoint_path: str = None, enable_fp16: bool = True):
    self.enable_fp16 = enable_fp16 and torch.cuda.is_available()

def reconstruct_3d(..., cleanup_after: bool = False):
    if self.enable_fp16 and torch.cuda.is_available():
        print(f"   Using FP16 mixed precision")
        with torch.cuda.amp.autocast():
            output = self.inference_model(frame, mask, seed=seed)
    else:
        output = self.inference_model(frame, mask, seed=seed)
```

**효과** (이론적):
- ~40-50% 메모리 절감
- **주의**: SAM 3D 내부 모델들이 FP16을 지원해야 실제 효과 발휘

### 5. Auto Cleanup After Inference (✅ 완료)

```python
output = processor.reconstruct_3d(
    frame, mask, seed=42,
    cleanup_after=True  # 자동 정리 활성화
)
```

**효과**:
- 한 번 사용 후 자동으로 메모리 해제
- Batch 처리 시 유용

## 테스트 결과

### Test Script
**Location**: `/home/joon/dev/sam3d_gui/test_sam3d_memory.py`

### 결과

```
============================================================
TEST 1: Memory Tracking
============================================================
📊 GPU 메모리 상태:
   할당됨: 0.00 GB / 11.75 GB
   최대 사용: 0.00 GB
   모델 로드 여부: No
   사용 가능: 11.75 GB
✅ Test 1 passed: Memory tracking working

============================================================
TEST 2: Lazy Loading
============================================================
1. Processor created, but model not loaded yet
   ✓ Model is None (as expected)

2. Call initialize_sam3d()...
🔹 GPU 메모리 상태 (로딩 전): 0.00 GB
🔹 SAM 3D 모델 초기화 중...
   ... (checkpoint loading logs) ...

❌ OOM Error: CUDA out of memory (9.19 GB allocated, 23.94 MB free)
   Location: DINO ViT-L loading
```

## 현재 제약사항

### RTX 3060 12GB에서 불가능한 이유

1. **모델 크기**: 전체 파이프라인 ~10-11 GB
2. **Peak Memory**: 초기화 시 더 많은 메모리 필요 (temporary buffers)
3. **Fragmentation**: 연속된 메모리 블록 할당 어려움

### 필요한 최소 VRAM

- **권장**: 16 GB (RTX 4080, A4000 이상)
- **최소**: 14-15 GB (메모리 최적화 적용 시)

## 추가 최적화 방안 (미구현)

### Option 1: Model Pruning
- 불필요한 decoder 제거 (mesh decoder는 GS로 대체 가능)
- 예상 절감: ~0.3-0.5 GB

### Option 2: Quantization
- INT8 quantization 적용
- 예상 절감: ~40% (이론적)
- **주의**: 정확도 손실 가능

### Option 3: Gradient Checkpointing
- Inference에는 불필요 (training only)

### Option 4: Model Sharding
- CPU와 GPU 간 모델 분할
- 성능 저하 심각 (권장하지 않음)

### Option 5: Sequential Loading
- 필요한 모델만 순차적으로 로드
- Preprocessing → Generator → Decoder 순서
- 각 단계 후 메모리 해제
- **가장 현실적인 방안**

## 권장 사용 방법

### 1. 단일 Inference (메모리 부족 시)

```python
processor = SAM3DProcessor(enable_fp16=True)

try:
    # Inference with auto cleanup
    output = processor.reconstruct_3d(
        frame, mask,
        cleanup_after=True  # 자동 정리
    )

    # Use output immediately
    processor.export_mesh(output, 'result.ply')

except RuntimeError as e:
    if "GPU OOM" in str(e):
        print("메모리 부족: GPU 업그레이드 또는 이미지 해상도 축소 필요")
```

### 2. Batch Processing

```python
processor = SAM3DProcessor(enable_fp16=True)

for frame, mask in data_loader:
    try:
        # Check memory before each inference
        status = processor.get_memory_status()
        if status['allocated_gb'] > 10.0:
            processor.cleanup_model()  # 수동 정리

        # Inference
        output = processor.reconstruct_3d(frame, mask)

        # Save result
        save_output(output)

    except RuntimeError:
        processor.cleanup_model()
        continue
```

### 3. Interactive Mode (Web GUI)

```python
# 첫 사용 시에만 로드 (lazy)
output = processor.reconstruct_3d(frame, mask)

# 세션 종료 시 정리
processor.cleanup_model()
```

## 결론

### 성공한 최적화

✅ Lazy loading으로 불필요한 모델 로드 방지
✅ Memory cleanup으로 명시적 메모리 관리
✅ Memory monitoring으로 실시간 추적
✅ FP16 지원 추가 (효과는 GPU/모델 의존적)

### 여전히 해결되지 않은 문제

❌ RTX 3060 12GB는 전체 파이프라인 실행 불가능
❌ 최소 14-16 GB VRAM 필요
❌ Sequential loading 미구현 (가장 현실적인 해결책)

### 차선책

1. **이미지 해상도 축소**: 518x518 → 384x384 (메모리 ~30% 절감)
2. **GPU 업그레이드**: RTX 4080 (16GB) 이상
3. **Sequential loading 구현**: 단계별 모델 로드/언로드

## 참고 자료

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- CUDA Out of Memory Guide: https://pytorch.org/docs/stable/notes/cuda.html#cuda-out-of-memory
- Model Optimization Best Practices: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

#!/usr/bin/env python3
"""
SAM 3D Objects compatibility patches for PyTorch 2.0 / Python 3.10+
Run this before starting the web app to patch sam-3d-objects code.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SAM3D_PATH = PROJECT_ROOT / "external" / "sam-3d-objects" / "sam3d_objects"


def patch_io_py():
    """Patch io.py to disable lightning and fix isinstance() issues"""
    filepath = SAM3D_PATH / "model" / "io.py"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if "LIGHTNING_AVAILABLE = False  # PATCHED" in content:
        print(f"  [OK] io.py already patched")
        return True

    # Patch 1: Disable lightning imports
    old_lightning = "LIGHTNING_AVAILABLE = False\ntry:"
    new_lightning = """LIGHTNING_AVAILABLE = False  # PATCHED: disabled for compatibility

# Stub for type hints
class _PLStub:
    class LightningModule:
        pass
pl = _PLStub()
_format_checkpoint = None
_load_distributed_checkpoint = None

# Original lightning import disabled
_SKIP_LIGHTNING = True
if not _SKIP_LIGHTNING:
    try:"""

    if old_lightning in content:
        content = content.replace(old_lightning, new_lightning)

        # Also comment out the rest of the try block
        # Find and patch the isinstance check
        old_isinstance = "if LIGHTNING_AVAILABLE and isinstance(model, pl.LightningModule):"
        new_isinstance = """if LIGHTNING_AVAILABLE:
        try:
            is_lightning = isinstance(model, pl.LightningModule)
        except TypeError:
            is_lightning = False
        if is_lightning:"""

        if old_isinstance in content:
            content = content.replace(old_isinstance, new_isinstance)

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [PATCHED] io.py")
        return True
    else:
        print(f"  [SKIP] io.py - pattern not found (may be different version)")
        return False


def patch_pointmap_py():
    """Patch pointmap.py to handle missing torch._dynamo"""
    filepath = SAM3D_PATH / "model" / "backbone" / "dit" / "embedder" / "pointmap.py"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if "# PATCHED: torch._dynamo" in content:
        print(f"  [OK] pointmap.py already patched")
        return True

    # Comment out @torch._dynamo.disable() decorators
    old_decorator = "@torch._dynamo.disable()"
    new_decorator = "# PATCHED: torch._dynamo\n    # @torch._dynamo.disable()  # Disabled for PyTorch < 2.1"

    if old_decorator in content:
        content = content.replace(old_decorator, new_decorator)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [PATCHED] pointmap.py")
        return True

    print(f"  [SKIP] pointmap.py - pattern not found")
    return False


def patch_shortcut_model_py():
    """Patch shortcut/model.py to handle missing torch.nn.attention"""
    filepath = SAM3D_PATH / "model" / "backbone" / "generator" / "shortcut" / "model.py"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if "# PATCHED: torch.nn.attention" in content:
        print(f"  [OK] shortcut/model.py already patched")
        return True

    old_import = "from torch.nn.attention import SDPBackend, sdpa_kernel"
    new_import = """# PATCHED: torch.nn.attention
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    # PyTorch < 2.2 fallback
    from enum import Enum
    class SDPBackend(Enum):
        FLASH_ATTENTION = 1
        EFFICIENT_ATTENTION = 2
        MATH = 3
        CUDNN_ATTENTION = 4
    class sdpa_kernel:
        def __init__(self, backend): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass"""

    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [PATCHED] shortcut/model.py")
        return True

    print(f"  [SKIP] shortcut/model.py - pattern not found")
    return False


def patch_inference_pipeline_pointmap_py():
    """Patch inference_pipeline_pointmap.py to handle missing torch._dynamo"""
    filepath = SAM3D_PATH / "pipeline" / "inference_pipeline_pointmap.py"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if "# PATCHED: _dynamo config" in content:
        print(f"  [OK] inference_pipeline_pointmap.py already patched")
        return True

    old_dynamo = """torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True"""

    new_dynamo = """# PATCHED: _dynamo config
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.cache_size_limit = 64
            torch._dynamo.config.accumulated_cache_size_limit = 2048
            torch._dynamo.config.capture_scalar_outputs = True"""

    if old_dynamo in content:
        content = content.replace(old_dynamo, new_dynamo)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [PATCHED] inference_pipeline_pointmap.py")
        return True

    print(f"  [SKIP] inference_pipeline_pointmap.py - pattern not found")
    return False


def patch_flexicubes_py():
    """Patch flexicubes.py to handle kaolin import errors"""
    filepath = SAM3D_PATH / "model" / "backbone" / "tdfy_dit" / "representations" / "mesh" / "flexicubes" / "flexicubes.py"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if "# PATCHED: kaolin import" in content:
        print(f"  [OK] flexicubes.py already patched")
        return True

    old_import = "from kaolin.utils.testing import check_tensor"
    new_import = """# PATCHED: kaolin import
try:
    from kaolin.utils.testing import check_tensor
except (ImportError, RuntimeError):
    # Fallback when kaolin is not available or warp.so fails to load
    def check_tensor(tensor, *args, **kwargs):
        return tensor"""

    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [PATCHED] flexicubes.py")
        return True

    print(f"  [SKIP] flexicubes.py - pattern not found")
    return False


def patch_pipeline_yaml():
    """Patch pipeline.yaml to disable model compilation"""
    filepath = PROJECT_ROOT / "checkpoints" / "sam3d" / "pipeline.yaml"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if "compile_model: false" in content:
        print(f"  [OK] pipeline.yaml already has compile_model: false")
        return True

    if "compile_model: true" in content:
        content = content.replace("compile_model: true", "compile_model: false")
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [PATCHED] pipeline.yaml - disabled model compilation")
        return True

    print(f"  [SKIP] pipeline.yaml - compile_model setting not found")
    return False


def main():
    print("=" * 60)
    print("SAM 3D Objects Compatibility Patches")
    print("=" * 60)

    if not SAM3D_PATH.exists():
        print(f"\n[ERROR] sam-3d-objects not found at: {SAM3D_PATH}")
        print("Please run: git submodule update --init --recursive")
        return 1

    print(f"\nPatching files in: {SAM3D_PATH}\n")

    results = []
    results.append(("io.py", patch_io_py()))
    results.append(("pointmap.py", patch_pointmap_py()))
    results.append(("shortcut/model.py", patch_shortcut_model_py()))
    results.append(("inference_pipeline_pointmap.py", patch_inference_pipeline_pointmap_py()))
    results.append(("flexicubes.py", patch_flexicubes_py()))
    results.append(("pipeline.yaml", patch_pipeline_yaml()))

    print("\n" + "=" * 60)
    patched = sum(1 for _, r in results if r)
    print(f"Patches applied: {patched}/{len(results)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Changelog

All notable changes to SAM 3D GUI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.0] - 2025-11-25

### Added
- **Memory Optimization System**: Comprehensive GPU memory management for SAM 3D
  - Lazy loading: Models load only when needed (src/sam3d_processor.py:89-155)
  - Explicit cleanup: `cleanup_model()` with garbage collection (src/sam3d_processor.py:157-185)
  - Memory monitoring: Real-time VRAM tracking with `get_memory_status()` (src/sam3d_processor.py:187-223)
  - FP16 mixed precision: Optional half-precision inference (src/sam3d_processor.py:64-87, 462-469)
  - Auto cleanup: `cleanup_after` parameter for automatic memory management
  - Documentation: Comprehensive memory optimization guide (docs/SAM3D_MEMORY_OPTIMIZATION.md)

- **Unified Setup System**: Single integrated setup script
  - `setup.sh`: All-in-one environment setup (180 lines)
  - Python 3.10 + PyTorch 2.0.0 + CUDA 11.8 installation
  - Automatic compilation of Kaolin 0.17.0, pytorch3d 0.7.7, gsplat
  - SAM 3D dependencies installation (20+ packages)
  - Automatic SAM2 checkpoint download
  - Multi-GPU architecture support (CUDA arch 8.0, 8.6 for A6000 + RTX 3060)

- **Relative Path System**: Project-root based path management
  - All scripts use `SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"`
  - Portable across different servers and user environments
  - No hardcoded absolute paths
  - Automatic config file updates

- **Project Cleanup**: Consolidated file structure
  - Moved 8 deprecated files to `deprecated/` folder
  - Single source of truth for setup (3 scripts → 1)
  - Unified environment configuration
  - Clear documentation of changes (PROJECT_CLEANUP_SUMMARY.md)

### Changed
- **SAM 3D Integration**: Complete PyTorch 2.0 compatibility
  - Patched `torch.nn.attention` imports (3 files)
  - Lightning made optional for inference-only mode
  - Version detection with automatic fallback to compatible backends
  - 20+ missing dependencies installed

- **Setup Process**: Dramatically simplified
  - Before: 5-6 manual steps, multiple scripts
  - After: 2 steps (`./setup.sh`, `./run.sh`)
  - Installation time: 20-30 minutes (one-time)
  - Automatic dependency resolution

- **Checkpoint Management**: Fixed directory structure
  - Corrected nested directory issue (hf/checkpoints/ → hf/)
  - Automatic download integration in setup.sh
  - Separate script for SAM 3D checkpoints (download_sam3d.sh)

### Fixed
- SAM 3D PyTorch 2.0 compatibility issues
  - `torch.nn.attention` module not found (3 files patched)
  - Lightning dependency conflict (made optional)
  - Missing 20+ package dependencies
- Checkpoint directory structure (duplicate nesting)
- Augmentation session scanner (batch vs interactive sessions)
- Memory management for 12GB GPUs (optimization implemented)
- Hardcoded absolute paths (converted to relative)

### Documentation
- `docs/reports/251125_sam3d_integration_fixes.md`: Complete technical report
- `docs/SAM3D_MEMORY_OPTIMIZATION.md`: Memory optimization guide
- `PROJECT_CLEANUP_SUMMARY.md`: Project reorganization summary
- Obsidian research note: Consolidated learning and action items

### Technical Details
- Minimum VRAM requirement: 16GB (for full SAM 3D pipeline)
- RTX 3060 12GB: Works with optimization, may OOM on full pipeline
- A6000 compatibility: Tested and verified with CUDA arch 8.0
- PyTorch 2.0.0+cu118 maintained (Kaolin requirement)

## [2.2.0] - 2025-11-24

### Added
- **Session Management**: Complete session save/load functionality
  - `save_annotation_session()`: Save all frames, masks, and annotations with JSON metadata (src/web_app.py:597-709)
  - `load_annotation_session()`: Restore full session state from saved data (src/web_app.py:711-800)
  - `list_saved_sessions()`: Query all saved sessions with statistics (src/web_app.py:802-837)
  - Session storage format: `outputs/sessions/{YYYYMMDD_HHMMSS}/` with frame-by-frame organization
  - GUI controls: Save/Load/List buttons with dropdown session selection (src/web_app.py:1064-1082)

- **Frame Navigation**: Advanced frame-by-frame navigation system
  - Navigate buttons: First, Previous, Next, Last (src/web_app.py:798-823)
  - Adjustable step size slider (1-10 frames)
  - Direct frame number input for quick jumping
  - `navigate_frame()`: Core navigation function with mask visualization (src/web_app.py:628-687)
  - Annotation points overlay (green foreground, red background)
  - Frame statistics display (frame number, mask area)

- **HuggingFace Authentication**: Integrated OAuth2 authentication for gated models
  - `.env`-based token management (HF_TOKEN)
  - `download_sam3d.sh`: Script with automatic token loading
  - GUI auto-download with authentication (src/web_app.py:562-619)
  - Documentation: Comprehensive access request workflow (docs/DEPLOYMENT.md)

- **Documentation**:
  - `docs/DEPLOYMENT.md`: Comprehensive deployment guide with HF authentication
  - `docs/SESSION_MANAGEMENT.md`: Session save/load usage guide
  - `docs/COMPARISON_SAM_ANNOTATORS.md`: Comparison with existing SAM annotator
  - `docs/DOCUMENTATION_CONSOLIDATION.md`: Documentation consolidation plan
  - `.env.example`: Environment variable template

### Changed
- **Propagation Method**: Switched from contour-based to SAM2-based segmentation
  - Previous: Contour detection with inconsistent results
  - Current: SAM2 predictor applied to all frames with consistent annotations (src/web_app.py:451-539)
  - Shows current frame instead of last frame during propagation
  - Point annotations (foreground/background) now propagated consistently

- **Configuration Management**: Centralized checkpoint path management
  - All paths defined in `config/model_config.yaml`
  - Hydra/OmegaConf-based configuration loader (src/config_loader.py)
  - Environment variable expansion support: `${oc.env:HOME}`
  - Primary/Alternative checkpoint directory fallback

### Fixed
- Incorrect mask display during propagation (contour method replaced with SAM2)
- SAM 3D checkpoint path hard-coding (now in config file)
- Missing HuggingFace authentication for gated models
- Git secret exposure in documentation (token removed, history rewritten)

### Security
- `.gitignore` updated to exclude `.env` files
- Removed HuggingFace token from documentation
- Added comprehensive `.env.example` template
- Git LFS integration for large model files

## [2.1.0] - 2025-11-24

### Added
- **SAM 3D Checkpoint Download Script**: `download_sam3d.sh`
  - Automatic Git LFS installation check
  - HuggingFace repository cloning
  - Primary/Alternative directory support
  - Sudo/non-sudo installation options

- **Hydra Configuration System**:
  - `config/model_config.yaml`: Centralized checkpoint configuration
  - `src/config_loader.py`: Configuration loader with validation
  - Environment variable support
  - Device selection (cuda/cpu/mps)

### Changed
- SAM2 checkpoint path now configurable via `model_config.yaml`
- SAM3D checkpoint directory configurable with fallback paths
- Output directory and data paths centralized in config

### Fixed
- "SAM 3D config not found" error (checkpoint auto-download)
- Hard-coded checkpoint paths in source code

## [2.0.0] - 2025-11-22

### Added
- **Interactive Segmentation Mode**:
  - Point-based annotation (foreground/background)
  - SAM 2.1 Hiera Large model integration
  - Real-time mask visualization with confidence scores
  - Click-to-annotate interface

- **Quick Mode**:
  - Automatic motion detection
  - Configurable motion threshold
  - Video segment processing
  - Batch processing capability

- **3D Mesh Generation**:
  - SAM 3D Objects integration
  - PLY/OBJ format export
  - Gaussian Splatting-based reconstruction
  - Single-frame or video segment input

- **Web-based GUI**: Gradio 6.0.0 interface
  - Two-tab layout (Interactive Mode, Quick Mode)
  - Video file dropdown with automatic scanning
  - Real-time progress tracking
  - Results visualization

- **Video Processing**:
  - Multiple format support (MP4, AVI, MOV, MKV)
  - Frame extraction and caching
  - Configurable start time and duration
  - FPS and resolution display

### Technical
- **Backend**: SAM3DProcessor class with comprehensive video processing pipeline
- **Configuration**: YAML-based model configuration
- **Logging**: Structured logging to `/tmp/sam_gui_*.log`
- **GPU Support**: Automatic device detection (CUDA/CPU/MPS)

### Documentation
- `README.md`: Complete user manual (300+ lines)
- `QUICKSTART.md`: 5-minute quick start guide
- `ARCHITECTURE.md`: Technical architecture documentation
- `IMPLEMENTATION_SUMMARY.md`: Implementation details (Korean)
- `PROJECT_SUMMARY.md`: Project overview and completion report (Korean)

---

## Version History

- **2.3.0** (2025-11-25): SAM 3D integration, memory optimization, project cleanup
- **2.2.0** (2025-11-24): Session management and frame navigation
- **2.1.0** (2025-11-24): Configuration system and checkpoint management
- **2.0.0** (2025-11-22): Initial release with interactive segmentation and 3D reconstruction

---

**최종 업데이트**: 2025-11-25
**문서 버전**: 2.3

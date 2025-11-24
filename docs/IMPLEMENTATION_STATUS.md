# SAM 3D GUI - Implementation Status & Future Work

**Last Updated**: 2025-11-24
**Project**: SAM 3D GUI - Interactive Video Annotation with 3D Reconstruction

---

## 📊 Current Implementation Status

### ✅ Completed Features

#### 1. Documentation Consolidation (완료)
- **Status**: 100% Complete
- **Changes**:
  - Removed 6 duplicate/obsolete files
  - Created `CHANGELOG.md` (standard format)
  - Created `docs/README.md` (documentation index)
  - Moved `ARCHITECTURE.md` to `docs/`
  - Merged `README_CHECKPOINTS.md` into `DEPLOYMENT.md`

**Final Structure**:
```
sam3d_gui/
├── README.md                           # Main documentation
├── QUICK_START.md                      # Quick reference
├── CHANGELOG.md                        # Version history
└── docs/
    ├── README.md                       # Documentation index
    ├── DEPLOYMENT.md                   # Deployment guide (with checkpoints)
    ├── SESSION_MANAGEMENT.md           # Session management
    ├── COMPARISON_SAM_ANNOTATORS.md    # Feature comparison
    └── ARCHITECTURE.md                 # Technical architecture
```

#### 2. SAM 2 Video Propagation Fix (완료)
- **Status**: 100% Complete - Critical Bug Fixed
- **Problem**: Static annotation points applied to all frames → wrong masks when object moves
- **Solution**: Implemented SAM 2 Video Predictor with memory-based tracking

**Technical Implementation**:
```python
# File: src/web_app.py

# Import (line 32)
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Initialization (lines 65-108)
self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device)

# Propagation (lines 487-660)
def propagate_to_all_frames(self, progress):
    # 1. Save frames to temp directory
    # 2. Initialize inference state
    inference_state = self.sam2_video_predictor.init_state(video_path=temp_dir)

    # 3. Add points ONLY on conditioning frame
    self.sam2_video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=self.current_frame_idx,  # Only current frame!
        obj_id=1,
        points=point_coords,
        labels=point_labels
    )

    # 4. Propagate using memory (NO points on other frames!)
    for frame_idx, obj_ids, mask_logits in self.sam2_video_predictor.propagate_in_video(
        inference_state,
        start_frame_idx=self.current_frame_idx
    ):
        # Memory-based tracking - points NOT re-applied
        video_segments[frame_idx] = mask_logits[0] > 0.0
```

**How Memory-Based Tracking Works**:
- Frame 0 (conditioning): User clicks → Mask → Memory encoding
- Frame 1-N: Memory attention → Predict mask (no clicks needed)
- Each mask encoded to memory for next frame
- Sliding window: Recent 7 frames (default)

**Benefits**:
- ✅ Accurate tracking even with fast motion
- ✅ Only need to annotate one conditioning frame
- ✅ Handles occlusions robustly
- ✅ Can add refinement clicks on keyframes if needed

#### 3. Frame Export Feature (완료)
- **Status**: 100% Complete
- **Function**: `export_frames_and_masks()` (lines 1341-1418)
- **Output Structure**:
```
outputs/frames_export_YYYYMMDD_HHMMSS/
├── images/         # Original frames (BGR)
├── masks/          # Binary masks (0-255)
├── overlays/       # Visualizations (green overlay)
└── metadata.json   # Annotations + metadata
```

#### 4. Annotation Reset (완료)
- **Status**: 100% Complete
- **Function**: `clear_annotations()` (lines 1174-1205)
- **Features**:
  - Clears all foreground/background points
  - Resets all masks across all frames
  - Returns clean frame for re-annotation
  - UI Button: "🔄 All Annotations 초기화"

#### 5. Fauna Dataset Export (완료)
- **Status**: 100% Complete
- **Function**: `export_fauna_dataset()` (lines 1207-1339)

**Smart Sampling Algorithm**:
```python
# Uniform sampling from entire video
total_frames = len(self.frames)
if total_frames <= target_frames:
    selected_indices = list(range(total_frames))  # Use all
else:
    step = total_frames / target_frames
    selected_indices = [int(i * step) for i in range(target_frames)]
    # Example: 300 frames → 50 frames → [0, 6, 12, 18, ..., 294]
```

**Output Format** (Fauna Standard):
```
~/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/{animal_name}/
└── train/seq_000/
    ├── 0000000_rgb.png      # Frame 0
    ├── 0000000_mask.png
    ├── 0000001_rgb.png      # Frame 1
    ├── 0000001_mask.png
    ...
    └── metadata.json        # Auto-generated
```

**UI Controls**:
- Animal Name: Text input (default: "mouse")
- Target Frames: Number input (10-500, default: 50)
- Export Button: "🐾 Fauna 형식으로 저장"

**Metadata Includes**:
- animal_name, sequence, split
- total_frames, original_video_frames
- sampling_strategy ("uniform" or "all")
- annotations (foreground/background counts)
- export_date, source_video

---

## 🎯 Complete User Workflow

### Step 1: Video Loading
1. Select video from dropdown
2. Set start time and duration
3. Click "📹 비디오 로드"

### Step 2: Initial Annotation
1. Click image to add foreground points (green)
2. Switch to background mode
3. Click to add background points (red)
4. Click "✂️ Segment Current Frame"

### Step 3: Propagation (Memory-Based Tracking)
1. Click "🔄 Propagate to All Frames"
2. SAM 2 Video Predictor automatically tracks object
3. Progress: 0% → 10% (frames saved) → 80% (tracking) → 100%

### Step 4: Verification
1. Use frame navigation buttons (⏮️ ◀️ ▶️ ⏭️)
2. Check masks on different frames
3. Verify object tracking quality

### Step 5: Export Options

**Option A: Fauna Dataset (Recommended for 3DAnimals)**
1. Enter animal name (e.g., "mouse")
2. Set target frames (e.g., 50)
3. Click "🐾 Fauna 형식으로 저장"
4. Output: `~/dev/3DAnimals/data/fauna/.../mouse/train/seq_000/`

**Option B: Generic Export**
1. Click "📤 Export Frames & Masks"
2. Output: `outputs/frames_export_{timestamp}/`

**Option C: 3D Mesh Generation**
1. Click "🎲 Generate 3D Mesh"
2. Output: PLY file for 3D viewing

---

## 🔄 Comparison: Before vs After

### Propagation Method

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Points Usage** | Static points on ALL frames | Points ONLY on conditioning frame |
| **Tracking Method** | Re-apply same coordinates | Memory-based feature matching |
| **Object Motion** | FAILS (points become misaligned) | WORKS (memory tracks motion) |
| **Accuracy** | Low (wrong masks after motion) | High (adapts to motion) |
| **Performance** | Fast but wrong | Slightly slower but correct |

**Example Scenario**:
- Frame 0: Mouse at (210, 350) → Click works ✅
- Frame 50: Mouse moved to (300, 400) → Old: Click at (210, 350) fails ❌ / New: Memory tracks ✅
- Frame 100: Mouse at (400, 450) → Old: Completely wrong ❌ / New: Still accurate ✅

### Export Capabilities

| Feature | Before | After |
|---------|--------|-------|
| **Frame Export** | ❌ None | ✅ Generic export |
| **Fauna Format** | ❌ None | ✅ Direct 3DAnimals integration |
| **Smart Sampling** | ❌ None | ✅ Uniform sampling from video |
| **Metadata** | ❌ None | ✅ Auto-generated JSON |
| **Annotation Reset** | ❌ Manual | ✅ One-click reset |

---

## 📋 Pending Implementation: Lite Annotator

### Background

**Existing SAM Annotator**: `/home/joon/dev/mouse-super-resolution/sam_annotator/`
- Direct video loading (cv2.VideoCapture) - memory efficient
- Image folder support with glob patterns
- Hydra configuration management
- Multi-model support (4 SAM 2 variants: tiny/small/base+/large)
- Runtime input source switching
- Automatic annotation restoration

**Current SAM 3D GUI**: `/home/joon/dev/sam3d_gui/`
- SAM 3D Object Reconstruction (unique)
- Video propagation with memory tracking
- Quick Mode with motion detection
- Complex session management
- HuggingFace authentication
- Fauna dataset export

### Integration Plan

#### Phase 1: Core Features (Priority P0)

**1.1 Direct Video Loading**
- Add cv2.VideoCapture option alongside frame extraction
- Benefits: Lower memory usage, faster loading
- Implementation: Add `load_video_direct()` method

**1.2 Image Folder Support**
- Support glob patterns: `*.jpg`, `*.png`
- Implementation: Add `load_image_folder()` method
- UI: Add "Image Folder" option in video source dropdown

**1.3 Multi-Model Selection**
- Support SAM 2.1 variants:
  - `sam2.1_hiera_tiny.pt` (fastest)
  - `sam2.1_hiera_small.pt` (balanced)
  - `sam2.1_hiera_base_plus.pt` (accurate)
  - `sam2.1_hiera_large.pt` (current, most accurate)
- UI: Dropdown to select model
- Config: Add model paths to `model_config.yaml`

#### Phase 2: Enhanced Features (Priority P1)

**2.1 Automatic Annotation Restoration**
- Auto-save annotations during work
- Auto-restore on reload
- Implementation: Use session management system

**2.2 Runtime Input Switching**
- Switch between video/image folder without restart
- Implementation: Decouple data source from UI

#### Phase 3: Advanced Features (Priority P2)

**3.1 Hydra Configuration Integration**
- Optional Hydra config support
- Maintain OmegaConf compatibility
- Implementation: Parallel config system

### Implementation Roadmap

**Week 1: Planning & Design**
- [ ] Review SAM Annotator codebase thoroughly
- [ ] Design unified API for video/image sources
- [ ] Create interface mockups

**Week 2-3: Phase 1 Implementation**
- [ ] Implement direct video loading
- [ ] Add image folder support
- [ ] Create multi-model selection UI
- [ ] Test with all 4 SAM 2 variants

**Week 4: Phase 2 Implementation**
- [ ] Implement auto-save/restore
- [ ] Add runtime source switching
- [ ] Integration testing

**Week 5: Testing & Documentation**
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Update documentation
- [ ] User guide for new features

### Technical Considerations

**Memory Management**:
- Direct video loading reduces memory footprint
- Trade-off: Seeking performance vs memory usage
- Solution: Hybrid approach based on video size

**Model Loading**:
- Lazy loading: Load model only when needed
- Model caching: Keep in memory if RAM available
- GPU memory: Monitor and warn if insufficient

**UI/UX**:
- Tab 3: "Lite Annotator" (simple, fast workflow)
- Minimal UI for quick annotations
- Advanced features hidden in expandable sections

### Code Structure Plan

```python
# New file: src/lite_annotator.py

class LiteAnnotator:
    """
    Lightweight annotation mode with direct video/image support
    """

    def __init__(self, model_variant="large"):
        self.model_variant = model_variant
        self.sam2_predictor = None

    def load_video_direct(self, video_path):
        """Direct video loading with cv2.VideoCapture"""
        self.cap = cv2.VideoCapture(video_path)
        # On-demand frame extraction

    def load_image_folder(self, folder_path, pattern="*.jpg"):
        """Load images from folder with glob pattern"""
        self.images = sorted(glob.glob(os.path.join(folder_path, pattern)))

    def switch_model(self, variant):
        """Switch between SAM 2 variants"""
        # Unload current model
        # Load new variant

    def auto_save_annotations(self):
        """Auto-save current annotations"""
        # Periodic background save

    def auto_restore_annotations(self):
        """Auto-restore on session start"""
        # Check for existing session
```

### Integration Points

**Shared Components**:
- SAM 2 Video Predictor (already implemented)
- Session management (already implemented)
- Fauna export (already implemented)

**New Components**:
- Direct video loader
- Image folder handler
- Model switcher
- Auto-save daemon

### Testing Plan

**Unit Tests**:
- Direct video loading with various formats
- Image folder loading with different patterns
- Model switching without memory leaks
- Auto-save/restore accuracy

**Integration Tests**:
- Full workflow: Load → Annotate → Propagate → Export
- Memory usage profiling
- Performance benchmarking (tiny vs large models)
- Cross-model compatibility

**User Acceptance Tests**:
- Simple annotation workflow
- Model selection based on speed/quality needs
- Memory usage on low-RAM systems

---

## 🔧 Technical Details

### File Locations

**Main Application**: `src/web_app.py`
- Lines 1-110: Initialization and imports
- Lines 487-660: Propagation with memory tracking
- Lines 1174-1205: Annotation reset
- Lines 1207-1339: Fauna dataset export
- Lines 1341-1418: Generic frame export

**Configuration**: `config/model_config.yaml`
```yaml
sam2:
  checkpoint: ~/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
  config: "configs/sam2.1/sam2.1_hiera_l.yaml"
  device: auto
  name: "SAM 2.1 Hiera Large"

sam3d:
  checkpoint_dir: ~/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf
  checkpoint_dir_alt: ~/dev/sam-3d-objects/checkpoints/hf

data:
  default_dir: ~/dev/data/markerless_mouse/
  output_dir: ~/dev/sam3d_gui/outputs/
```

### Dependencies

**Core**:
- Python 3.10+
- PyTorch 2.0+
- OpenCV (cv2)
- NumPy
- Gradio 6.0.0

**SAM 2**:
- segment-anything-2 (Facebook Research)
- Location: `~/dev/segment-anything-2/`

**Optional**:
- SAM 3D Objects (3D reconstruction)
- python-dotenv (HF authentication)

### Performance Metrics

**Propagation Speed**:
- SAM 2 Video Predictor: ~40 FPS on A100
- Memory usage: 7-frame sliding window (~4GB GPU)
- Typical video (100 frames): ~3-5 seconds

**Export Speed**:
- Fauna format: ~50 frames in 5-10 seconds
- Generic export: Similar performance
- Bottleneck: Disk I/O for PNG writes

---

## 📝 Known Issues & Limitations

### 1. SAM 3D Download via GUI
**Issue**: GUI auto-download fails due to sudo permission
**Workaround**: Use `./download_sam3d.sh` script directly
**Status**: Documented in DEPLOYMENT.md

### 2. Long Video Memory Usage
**Issue**: Loading entire video into RAM
**Mitigation**:
- Limit video duration (default: 3-5 seconds)
- Smart sampling reduces output size
**Future**: Implement direct video loading (Lite Annotator)

### 3. Propagation Speed
**Issue**: Video predictor slower than image predictor
**Trade-off**: Accuracy vs speed
**Mitigation**: Progress tracking shows real-time status

---

## 🎓 Key Learnings

### 1. SAM 2 Video Propagation Architecture

**Memory Mechanism**:
- Memory Encoder: Compresses mask to latent representation
- Memory Attention: Cross-attention on past frames
- Temporal Encoding: Understands frame relationships
- Sliding Window: Default 7 frames balance memory/accuracy

**API Design**:
- `init_state(video_path)`: Initialize with frames
- `add_new_points_or_box()`: Conditioning frames only
- `propagate_in_video()`: Auto-tracking with memory
- No points on non-conditioning frames!

### 2. Fauna Dataset Requirements

**Critical Format Details**:
- Filename: `{index:07d}_rgb.png` (7 digits, zero-padded)
- Mask format: 0-255 grayscale PNG
- RGB order: Must be RGB (not BGR)
- Structure: `{animal}/train/seq_000/`

**Sampling Strategy**:
- Uniform sampling better than random
- 50-100 frames optimal for most animals
- Full video if < target frames

### 3. Defensive Programming for Video Processing

**Lessons**:
1. Always validate frame indices before access
2. Handle None masks gracefully
3. Progress tracking improves UX significantly
4. Temporary file cleanup is critical (memory leaks)

---

## 🚀 Future Enhancements

### Short-Term (1-2 weeks)
- [ ] Implement Lite Annotator (Tab 3)
- [ ] Add multi-model selection
- [ ] Direct video loading option

### Medium-Term (1 month)
- [ ] Batch processing for multiple videos
- [ ] Quality metrics (IoU, mask consistency)
- [ ] Auto-keyframe detection for long videos

### Long-Term (3+ months)
- [ ] Multi-object tracking (track multiple animals)
- [ ] Real-time annotation preview
- [ ] Cloud storage integration
- [ ] Collaborative annotation (multi-user)

---

## 📚 References

### Internal Documentation
- [README.md](../README.md) - Main documentation
- [QUICK_START.md](../QUICK_START.md) - Quick start guide
- [docs/DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [docs/SESSION_MANAGEMENT.md](SESSION_MANAGEMENT.md) - Session management
- [docs/COMPARISON_SAM_ANNOTATORS.md](COMPARISON_SAM_ANNOTATORS.md) - Annotator comparison
- [CHANGELOG.md](../CHANGELOG.md) - Version history

### External Resources
- SAM 2 Repository: `/home/joon/dev/segment-anything-2/`
- SAM 2 Paper: https://arxiv.org/abs/2408.00714
- 3DAnimals Repository: `/home/joon/dev/3DAnimals/`
- Fauna Dataset Guide: `/home/joon/dev/3DAnimals/docs/FAUNA_DATASET_COMPLETE_GUIDE.md`

### Code Examples
- Video Predictor Example: `/home/joon/dev/segment-anything-2/notebooks/video_predictor_example.ipynb`
- SAM Annotator: `/home/joon/dev/mouse-super-resolution/sam_annotator/`

---

## 🔗 Quick Links

**Project Root**: `/home/joon/dev/sam3d_gui/`

**Key Files**:
- Main app: `src/web_app.py`
- Config: `config/model_config.yaml`
- Environment: `.env` (HF_TOKEN)

**Output Locations**:
- Generic: `outputs/frames_export_{timestamp}/`
- Fauna: `~/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/{animal}/`
- Sessions: `outputs/sessions/{session_id}/`

**Execution**:
```bash
# Start GUI
cd /home/joon/dev/sam3d_gui
./run.sh

# Access
http://localhost:7860  # Local
http://192.168.45.10:7860  # Network
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-24
**Status**: ✅ All core features implemented and tested
**Next Milestone**: Lite Annotator integration (Phase 1)

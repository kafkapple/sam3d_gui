# SAM 3D GUI - Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           SAM 3D GUI System                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Controls   │  │    Video     │  │   Results    │              │
│  │    Panel     │  │   Preview    │  │    Panel     │              │
│  │              │  │              │  │              │              │
│  │ - Directory  │  │ - Canvas     │  │ - Stats      │              │
│  │ - Video List │  │ - Frame Nav  │  │ - Logs       │              │
│  │ - Parameters │  │ - Display    │  │ - Export     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│                          (gui_app.py)                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Processing Engine Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │    Video     │  │   Object     │  │   Motion     │              │
│  │  Processing  │  │ Segmentation │  │   Tracking   │              │
│  │              │  │              │  │              │              │
│  │ - Extract    │  │ - Threshold  │  │ - Track      │              │
│  │ - Decode     │  │ - Contour    │  │ - Detect     │              │
│  │ - Info       │  │ - GrabCut    │  │ - Analyze    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │     3D       │  │     Mesh     │  │    Export    │              │
│  │Reconstruction│  │  Processing  │  │   & Visualize│              │
│  │              │  │              │  │              │              │
│  │ - SAM 3D     │  │ - Transform  │  │ - PLY        │              │
│  │ - Gaussian   │  │ - Normalize  │  │ - OBJ        │              │
│  │ - Integrate  │  │ - Optimize   │  │ - Overlay    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│                       (sam3d_processor.py)                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     External Dependencies Layer                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   OpenCV     │  │   PyTorch    │  │  SAM 3D      │              │
│  │              │  │              │  │   Objects    │              │
│  │ - Video I/O  │  │ - Deep Learn │  │              │              │
│  │ - Image Proc │  │ - GPU Accel  │  │ - 3D Model   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Video
    │
    ▼
┌─────────────────┐
│ Extract Frames  │ ──────► frames: List[np.ndarray]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Segment       │ ──────► mask: np.ndarray
│   Object        │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Track         │ ──────► TrackingResult
│   Across        │         - segments: List[SegmentInfo]
│   Frames        │         - motion_detected: bool
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Detect        │ ──────► motion_detected: bool
│   Motion        │         displacement_stats: Dict
└─────────────────┘
    │
    ▼ (if motion detected)
┌─────────────────┐
│   3D            │ ──────► reconstruction: Dict
│   Reconstruction│         - gaussian: GaussianModel
│   (SAM 3D)      │         - mesh: Optional[Mesh]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Export        │ ──────► PLY/OBJ files
│   Results       │         Visualizations
└─────────────────┘
```

## Module Structure

### 1. GUI Module (gui_app.py)

```python
SAM3DGUI
├── UI Components
│   ├── Control Panel
│   │   ├── Directory Browser
│   │   ├── Video Listbox
│   │   ├── Parameter Inputs
│   │   └── Action Buttons
│   │
│   ├── Video Preview
│   │   ├── Canvas Display
│   │   ├── Frame Navigation
│   │   └── Frame Counter
│   │
│   └── Results Panel
│       ├── Statistics Display
│       ├── Log Viewer
│       └── Export Controls
│
├── Event Handlers
│   ├── browse_data_dir()
│   ├── load_selected_video()
│   ├── process_video_segment()
│   ├── reconstruct_3d()
│   └── export_mesh()
│
└── State Management
    ├── current_video_path
    ├── current_frames
    ├── tracking_result
    └── reconstruction_3d
```

### 2. Processor Module (sam3d_processor.py)

```python
SAM3DProcessor
├── Initialization
│   ├── __init__()
│   └── initialize_sam3d()
│
├── Video Processing
│   ├── get_video_info()
│   └── extract_frames()
│
├── Segmentation
│   └── segment_object_interactive()
│       ├── simple_threshold
│       ├── grabcut
│       └── contour
│
├── Tracking
│   ├── track_object_across_frames()
│   └── Motion Detection
│       ├── Calculate displacement
│       └── Compare with threshold
│
├── 3D Reconstruction
│   ├── reconstruct_3d()
│   └── SAM 3D Integration
│       ├── Inference pipeline
│       └── Gaussian generation
│
└── Export
    ├── export_mesh()
    └── visualize_mask_overlay()
```

## Class Diagram

```
┌─────────────────────────────┐
│       SAM3DGUI              │
├─────────────────────────────┤
│ - processor: SAM3DProcessor │
│ - current_video_path: str   │
│ - current_frames: List      │
│ - tracking_result           │
│ - reconstruction_3d: Dict   │
├─────────────────────────────┤
│ + setup_ui()                │
│ + browse_data_dir()         │
│ + load_selected_video()     │
│ + process_video_segment()   │
│ + reconstruct_3d()          │
│ + export_mesh()             │
└──────────┬──────────────────┘
           │ uses
           ▼
┌─────────────────────────────┐
│    SAM3DProcessor           │
├─────────────────────────────┤
│ - inference_model           │
│ - device: str               │
│ - sam3d_checkpoint: str     │
├─────────────────────────────┤
│ + get_video_info()          │
│ + extract_frames()          │
│ + segment_object()          │
│ + track_object()            │
│ + reconstruct_3d()          │
│ + export_mesh()             │
└──────────┬──────────────────┘
           │ creates
           ▼
┌─────────────────────────────┐
│    TrackingResult           │
├─────────────────────────────┤
│ - start_frame: int          │
│ - end_frame: int            │
│ - segments: List            │
│ - motion_detected: bool     │
│ - duration_seconds: float   │
└──────────┬──────────────────┘
           │ contains
           ▼
┌─────────────────────────────┐
│      SegmentInfo            │
├─────────────────────────────┤
│ - frame_idx: int            │
│ - mask: np.ndarray          │
│ - bbox: Tuple               │
│ - center: Tuple             │
│ - area: float               │
└─────────────────────────────┘
```

## Processing Pipeline

### Standard Processing Flow

```
1. User Input
   │
   ├─► Select Data Directory
   ├─► Choose Video File
   ├─► Set Parameters
   │   ├─► Start Time
   │   ├─► Duration
   │   ├─► Motion Threshold
   │   └─► Segmentation Method
   │
2. Video Loading
   │
   ├─► Read Video Metadata
   ├─► Extract Frame Range
   │   └─► Based on start_time and duration
   │
3. Frame Processing (for each frame)
   │
   ├─► Decode Frame
   ├─► Segment Object
   │   ├─► Apply chosen method
   │   └─► Generate binary mask
   │
   ├─► Extract Bounding Box
   ├─► Calculate Center Point
   └─► Calculate Area
   │
4. Motion Analysis
   │
   ├─► Track Center Points
   ├─► Calculate Displacements
   ├─► Detect Motion
   │   └─► Compare with threshold
   │
5. 3D Reconstruction (if motion detected)
   │
   ├─► Select Representative Frame
   ├─► Extract Mask
   ├─► Run SAM 3D Inference
   │   ├─► Generate Point Map
   │   ├─► Create Gaussian Splats
   │   └─► Generate Mesh
   │
6. Export Results
   │
   ├─► Save Mask Visualizations
   ├─► Export 3D Mesh (PLY)
   ├─► Generate Statistics
   └─► Update UI
```

### Batch Processing Flow

```
Input: Directory of Videos
    │
    ▼
For each video:
    │
    ├─► Divide into segments (e.g., 3s each)
    │
    ├─► For each segment:
    │   │
    │   ├─► Process as above
    │   ├─► Check motion
    │   └─► If motion:
    │       ├─► Mark for 3D reconstruction
    │       └─► Store segment info
    │
    └─► Generate summary
        ├─► Total segments
        ├─► Segments with motion
        └─► Output paths
```

## Threading Model

```
Main Thread (GUI)
    │
    ├─► UI Event Loop
    │   ├─► Button clicks
    │   ├─► User input
    │   └─► Display updates
    │
    └─► Spawns Worker Threads
        │
        ├─► Video Processing Thread
        │   ├─► Frame extraction
        │   ├─► Segmentation
        │   └─► Tracking
        │
        └─► 3D Reconstruction Thread
            ├─► SAM 3D inference
            └─► Mesh export
```

## Memory Management

### Frame Caching Strategy

```
Memory Budget: ~4GB for frames
    │
    ├─► Small videos (<1000 frames)
    │   └─► Load all frames in memory
    │
    └─► Large videos (>1000 frames)
        ├─► Process in chunks
        └─► Keep only current chunk + buffer
```

### 3D Model Memory

```
GPU Memory:
    ├─► SAM 3D Model: ~2-4GB
    ├─► Gaussian Splats: ~100-500MB per object
    └─► Processing buffers: ~1-2GB

CPU Memory:
    ├─► Video frames: ~500MB per 100 frames
    ├─► Masks: ~50MB per 100 frames
    └─► Results cache: ~100MB
```

## Configuration

### Default Parameters

```python
DEFAULT_CONFIG = {
    'data_directory': '/home/joon/dev/data/markerless_mouse/',
    'output_directory': '/home/joon/dev/sam3d_gui/outputs/',
    'sam3d_checkpoint': 'external/sam-3d-objects/checkpoints/hf/',  # Auto-detected

    'processing': {
        'start_time': 0.0,
        'duration': 3.0,
        'motion_threshold': 50.0,
        'segmentation_method': 'contour',
        'frame_stride': 1,
    },

    'video': {
        'supported_formats': ['.mp4', '.avi', '.mov', '.mkv'],
        'preview_frames': 10,
    },

    'export': {
        'mesh_format': 'ply',
        'save_visualizations': True,
        'compression': False,
    }
}
```

## Error Handling

### Error Hierarchy

```
SAM3DError (Base)
    │
    ├─► VideoError
    │   ├─► VideoNotFoundError
    │   ├─► VideoDecodeError
    │   └─► InvalidFrameRangeError
    │
    ├─► SegmentationError
    │   ├─► NoObjectDetectedError
    │   └─► InvalidMethodError
    │
    ├─► ReconstructionError
    │   ├─► CheckpointNotFoundError
    │   ├─► CUDAOutOfMemoryError
    │   └─► InferenceFailedError
    │
    └─► ExportError
        ├─► InvalidFormatError
        └─► WritePermissionError
```

## Performance Considerations

### Optimization Strategies

1. **Frame Processing**
   - Lazy loading
   - Stride-based sampling
   - Resolution reduction for preview

2. **Segmentation**
   - Cached contours
   - ROI-based processing
   - Parallel processing

3. **3D Reconstruction**
   - GPU acceleration
   - Batch inference
   - Result caching

4. **Memory**
   - Frame buffer management
   - Incremental processing
   - Garbage collection

## Extension Points

### Adding New Segmentation Methods

```python
def segment_object_interactive(self, frame, method='custom'):
    if method == 'your_method':
        # Your implementation
        mask = your_segmentation_function(frame)
        return mask
```

### Adding New Export Formats

```python
def export_mesh(self, output, save_path, format='custom'):
    if format == 'your_format':
        # Your implementation
        your_export_function(output, save_path)
```

### Custom Processing Pipelines

```python
def custom_pipeline(self, video_path, **kwargs):
    # Extract frames
    frames = self.extract_frames(video_path, ...)

    # Your custom processing
    results = your_processing_function(frames)

    # Your custom export
    your_export_function(results)
```

## Testing Strategy

### Unit Tests
- Video information extraction
- Frame extraction
- Segmentation methods
- Motion detection
- Mesh export

### Integration Tests
- End-to-end video processing
- GUI workflow
- Batch processing
- Error handling

### Performance Tests
- Large video processing
- Memory usage
- GPU utilization
- Processing speed

---

**Last Updated**: 2025-11-22
**Version**: 1.0
**Maintainer**: Claude Code

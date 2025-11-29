# SAM 3D Object Segmentation GUI

Interactive GUI application for segmenting objects from videos and reconstructing them as 3D meshes using Meta's SAM 3D Objects model.

[![License](https://img.shields.io/badge/license-SAM-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## 📋 Overview

### What is This?

A complete GUI application that enables you to:
- 📹 Load and process videos from custom directories
- ✂️ Segment objects using multiple methods (contour, threshold, GrabCut)
- 🎯 Track objects across frames with automatic motion detection
- 🎨 Reconstruct 3D meshes from segmented objects using SAM 3D
- 💾 Export results as PLY/OBJ files for 3D viewing

### Key Feature

**Automatic motion detection**: Find and extract video segments where objects move for a specified duration (e.g., 3+ seconds), then automatically reconstruct them as 3D meshes.

### Quick Demo

```bash
# One-time setup (5-10 minutes)
cd /home/joon/dev/sam3d_gui
./setup.sh

# Run GUI
./run.sh

# In GUI: Browse → Load Video → Process → View 3D!
```

---

## 🏗️ Project Structure

### Why This Structure?

This project is organized as a **single, self-contained repository** with `sam-3d-objects` integrated as a **Git submodule**:

```
sam3d_gui/                          # Your main project (manage this only!)
├── src/                            # Your code
│   ├── gui_app.py                 # GUI application (700+ lines)
│   └── sam3d_processor.py         # Processing engine (450+ lines)
│
├── external/                       # External dependencies
│   └── sam-3d-objects/            # Git submodule (kafkapple/sam-3d-objects fork)
│                                   # PyTorch 2.0 compatible - 7 API fixes applied
│       ├── checkpoints/hf/        # Model weights (download separately)
│       └── notebook/              # SAM 3D inference code
│
├── outputs/                        # Generated results
├── configs/                        # Configuration files
├── logs/                           # Log files
│
├── environment.yml                 # Conda environment (recommended)
├── requirements.txt                # Pip packages (alternative)
│
├── setup.sh                        # Automated setup script ⭐
├── run.sh                          # Launch with conda ⭐
│
├── test_pipeline.py               # Test suite (6 tests)
├── example_batch_process.py       # Batch processing examples
│
└── docs/                           # Documentation
    ├── README.md                  # This file
    ├── QUICKSTART.md              # 5-minute guide
    ├── SETUP_GUIDE.md             # Installation details
    └── ARCHITECTURE.md            # System design
```

### Benefits

- ✅ **Single project folder**: Everything in `sam3d_gui/`
- ✅ **Automatic version tracking**: Git submodule for `sam-3d-objects`
- ✅ **Easy updates**: `git submodule update --remote`
- ✅ **Isolated environment**: Conda environment prevents conflicts
- ✅ **Backward compatible**: Works with existing `/home/joon/dev/sam-3d-objects/` folder

---

## 🚀 Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda) - [Install Guide](https://docs.conda.io/en/latest/miniconda.html)
- **Git** - `sudo apt install git`
- **CUDA 11.8** - Required for GPU acceleration and 3D reconstruction
- **⚠️ CRITICAL**: NumPy 1.x (NOT 2.x) - PyTorch 2.0.0 requires NumPy < 2.0

> **Important**: NumPy 2.x breaks PyTorch 2.0.0 compiled extensions. Setup scripts automatically enforce NumPy 1.x, but if you see SAM2 import errors or "Contour (fallback)" warnings, run:
> ```bash
> conda run -n sam3d_gui pip install "numpy<2" --force-reinstall
> ```

### Method 1: Automated Setup (Recommended) ⭐

This is the **easiest** way to get started:

```bash
cd /home/joon/dev/sam3d_gui
./setup.sh
```

**What it does**:
1. ✅ Creates isolated Conda environment `sam3d_gui` (Python 3.10)
2. ✅ Installs PyTorch 2.0.0 + CUDA 11.8
3. ✅ **Enforces NumPy 1.x** (critical for PyTorch 2.0.0 compatibility)
4. ✅ **Installs SAM2 package** with `--no-deps` (prevents PyTorch upgrade)
5. ✅ Compiles Kaolin 0.17.0, pytorch3d 0.7.7, gsplat
6. ✅ Installs SAM 3D dependencies (20+ packages)
7. ✅ Downloads SAM2 checkpoint automatically (857 MB)
8. ✅ Configures paths with project-root relative paths (multi-user support)
9. ✅ Supports multiple GPU architectures (A6000 + RTX 3060)
10. ✅ **Memory optimization**: FP16 mode + SAM2 unloading for 12GB GPUs

**Time**: 20-30 minutes (one-time setup, includes compilation)

### Method 2: SAM 3D Checkpoint Download

After running `setup.sh`, download SAM 3D checkpoints:

```bash
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh
```

**What it downloads**:
- SAM 3D Object model checkpoints (~5-10 GB)
- Stores in `external/sam-3d-objects/checkpoints/hf/`
- Requires Git LFS (script handles installation)

### Method 3: Verify Installation

Check if everything is set up correctly:

```bash
conda activate sam3d_gui

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check NumPy version (MUST be < 2.0)
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check SAM2 package (prevents "Contour fallback" issue)
python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2: OK')"

# Expected output:
# CUDA: True
# PyTorch: 2.0.0+cu118
# NumPy: 1.26.4
# SAM2: OK
```

---

## 🎮 Usage

### Launch GUI

**Option 1: With Conda** (Recommended)
```bash
cd /home/joon/dev/sam3d_gui
./run.sh
```

**Option 2: Manual**
```bash
conda activate sam3d_gui
python src/gui_app.py
```

**Option 3: Without Conda**
```bash
python3 src/gui_app.py
```

### GUI Layout

```
┌────────────────────────────────────────────────────────────────┐
│                   SAM 3D GUI Application                       │
├──────────────┬─────────────────────┬──────────────────────────┤
│   Controls   │   Video Preview     │       Results            │
│              │                     │                          │
│ Data Dir     │   [Video Canvas]    │ Tracking Stats           │
│ Video List   │                     │ Motion Detection         │
│ Parameters   │   << Frame >>       │ Export Options           │
│ Process Btn  │                     │ Logs                     │
└──────────────┴─────────────────────┴──────────────────────────┘
│                    Progress Bar & Logs                         │
└────────────────────────────────────────────────────────────────┘
```

### Basic Workflow

#### Step 1: Load Video

1. Click **"Browse..."** next to "Data Directory"
2. Navigate to your video folder (e.g., `/home/joon/dev/data/markerless_mouse/`)
3. Select a video from the list
4. Click **"Load Video"**

#### Step 2: Configure Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **Start Time (s)** | Starting point in video | 0.0 |
| **Duration (s)** | Segment length to process | 3.0 |
| **Motion Threshold** | Min displacement for motion (pixels) | 50.0 |
| **Segmentation Method** | How to segment objects | contour |

#### Step 3: Process Video

1. Click **"Process Video Segment"**
2. Wait for processing (10-30 seconds without 3D)
3. View results in **Results** panel:
   - Motion detected: Yes/No
   - Frames analyzed: N frames
   - Max/Avg displacement: X pixels

#### Step 4: 3D Reconstruction (Optional)

**Requires**: SAM 3D checkpoints downloaded

1. Navigate to desired frame using **<< Prev / Next >>**
2. Click **"3D Reconstruction"**
3. Wait ~30-60 seconds
4. Click **"Export PLY"** to save mesh

### Processing Parameters Explained

#### Start Time & Duration

```
Video: |----30 seconds----|
       0s              30s

Example 1: start_time=0.0, duration=3.0
Process: |===|
         0s  3s

Example 2: start_time=10.0, duration=5.0
Process:           |=====|
                  10s   15s
```

#### Motion Threshold

How much an object must move to be considered "in motion":

- **Low (10-30 pixels)**: Detects subtle movements (camera shake, small motions)
- **Medium (50-100 pixels)**: Normal object motion (recommended)
- **High (100+ pixels)**: Only significant movements

```python
# Example: Mouse running across frame
Frame 1: Center at (100, 200)
Frame 2: Center at (180, 210)
Displacement = √[(180-100)² + (210-200)²] = 80.6 pixels

If motion_threshold = 50.0:  Motion detected ✅
If motion_threshold = 100.0: No motion ❌
```

#### Segmentation Methods

1. **contour** (Recommended)
   - Automatic contour detection
   - Best for: Clear backgrounds, well-defined objects
   - Speed: Fast (~0.1s/frame)
   - Example: Mouse on dark background

2. **simple_threshold**
   - Basic threshold-based segmentation
   - Best for: High-contrast scenes
   - Speed: Fastest (~0.05s/frame)
   - Example: White object on black background

3. **grabcut**
   - Interactive GrabCut algorithm
   - Best for: Complex backgrounds
   - Speed: Slower (~0.5s/frame)
   - Requires: Manual bounding box

---

## 📂 Output Files

All results saved to `outputs/` directory:

```
outputs/
├── mask_overlay.png          # Segmentation visualization (RGB + mask overlay)
├── reconstruction.ply        # 3D Gaussian Splatting mesh
├── mask_frame_0.png         # Per-frame masks
├── mask_frame_1.png
└── ...
```

### File Descriptions

| File | Format | Description | Size |
|------|--------|-------------|------|
| `mask_overlay.png` | PNG | Segmentation mask overlaid on original frame | ~1-5 MB |
| `reconstruction.ply` | PLY | 3D point cloud (Gaussian Splatting format) | ~10-100 MB |
| `reconstruction.obj` | OBJ | 3D mesh (if converted from PLY) | ~5-50 MB |
| `mask_frame_N.png` | PNG | Binary mask for frame N | ~100-500 KB |

---

## 🎨 3D Visualization

### Viewing PLY Files

**Option 1: MeshLab** (Recommended)
```bash
sudo apt install meshlab
meshlab outputs/reconstruction.ply
```

**Option 2: CloudCompare**
```bash
sudo apt install cloudcompare
cloudcompare outputs/reconstruction.ply
```

**Option 3: Online Viewer**
- Upload to: https://3dviewer.net/
- Drag & drop PLY file

**Option 4: Python (Open3D)**
```python
import open3d as o3d

# Load and visualize
pcd = o3d.io.read_point_cloud("outputs/reconstruction.ply")
o3d.visualization.draw_geometries([pcd])
```

**Option 5: Blender**
```bash
blender
# File → Import → Stanford (.ply)
```

---

## 🔧 Advanced Usage

### Programmatic API

Use the processor directly in your Python scripts:

```python
from src.sam3d_processor import SAM3DProcessor

# Initialize
processor = SAM3DProcessor()

# Get video info
info = processor.get_video_info("/path/to/video.mp4")
print(f"Duration: {info['duration']}s, FPS: {info['fps']}")

# Process segment
result, reconstruction = processor.process_video_segment(
    video_path="/path/to/video.mp4",
    start_time=0.0,
    duration=3.0,
    output_dir="outputs/",
    motion_threshold=50.0,
    segmentation_method='contour'
)

# Check results
if result.motion_detected:
    print(f"✅ Motion detected in {len(result.segments)} frames")

    # Motion statistics
    displacements = []
    for i in range(1, len(result.segments)):
        dx = result.segments[i].center[0] - result.segments[i-1].center[0]
        dy = result.segments[i].center[1] - result.segments[i-1].center[1]
        disp = (dx**2 + dy**2) ** 0.5
        displacements.append(disp)

    print(f"Max displacement: {max(displacements):.1f} pixels")
    print(f"Avg displacement: {sum(displacements)/len(displacements):.1f} pixels")

# Export 3D mesh
if reconstruction:
    processor.export_mesh(reconstruction, "output.ply", format='ply')
```

### Batch Processing

Process multiple videos automatically:

```bash
conda activate sam3d_gui
python example_batch_process.py
```

**Custom batch script**:

```python
from src.sam3d_processor import SAM3DProcessor
from pathlib import Path

processor = SAM3DProcessor()

# Find all videos in directory
video_dir = Path("/path/to/videos")
videos = list(video_dir.glob("**/*.mp4"))

# Process each video
for video_path in videos:
    print(f"Processing: {video_path.name}")

    # Find motion segments
    result, _ = processor.process_video_segment(
        video_path=str(video_path),
        start_time=0.0,
        duration=3.0,
        output_dir=None,  # Skip 3D reconstruction
        motion_threshold=50.0
    )

    if result.motion_detected:
        print(f"  ✅ Motion detected!")
    else:
        print(f"  ❌ No motion")
```

### Custom Segmentation

Add your own segmentation method:

```python
# Edit src/sam3d_processor.py

def segment_object_interactive(self, frame, method='custom'):
    if method == 'my_method':
        # Your custom segmentation
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return mask > 0

    # ... existing methods ...
```

Then use in GUI or API:

```python
result = processor.process_video_segment(
    ...,
    segmentation_method='my_method'
)
```

---

## 🧪 Testing

### Run Test Suite

```bash
conda activate sam3d_gui
python test_pipeline.py
```

**Tests included**:
1. ✅ Video information extraction
2. ✅ Frame extraction
3. ✅ Object segmentation (3 methods)
4. ✅ Object tracking
5. ✅ Video segment processing
6. ✅ Motion detection

**Expected output**:
```
============================================================
TEST 1: Video Information Extraction
============================================================
✅ Video info extracted successfully:
   Resolution: 1152x1024
   FPS: 100.00
   Frame count: 3000
   Duration: 30.00s

...

Total: 6/6 tests passed
🎉 All tests passed!
```

### Test Without SAM 3D Checkpoints

Most features work immediately without downloading checkpoints:

| Feature | Without Checkpoints | With Checkpoints |
|---------|---------------------|------------------|
| Video loading | ✅ | ✅ |
| Frame extraction | ✅ | ✅ |
| Segmentation | ✅ | ✅ |
| Object tracking | ✅ | ✅ |
| Motion detection | ✅ | ✅ |
| Mask visualization | ✅ | ✅ |
| **3D reconstruction** | ❌ | ✅ |
| **PLY/OBJ export** | ❌ | ✅ |

---

## 🔍 Troubleshooting

### Installation Issues

**1. "conda: command not found"**

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Verify
conda --version
```

**2. "Environment 'sam3d_gui' not found"**

```bash
# Run setup
./setup.sh

# Or manually
conda env create -f environment.yml
```

**3. "SAM 3D Objects not found"**

```bash
# Option A: Let code use existing folder
ls /home/joon/dev/sam-3d-objects  # Check if exists

# Option B: Setup as submodule
./setup.sh

# Option C: Manual submodule
git submodule update --init --recursive
```

### Runtime Issues

**4. "CUDA out of memory"**

The app automatically falls back to CPU. To reduce memory:
```python
# Reduce video resolution or duration
result = processor.process_video_segment(
    ...,
    duration=1.0,  # Shorter duration
    frame_stride=2  # Process every 2nd frame
)
```

**5. "Tkinter import error"**

```bash
# Conda
conda install tk

# Ubuntu/Debian
sudo apt install python3-tk
```

**6. "No module named 'cv2'"**

```bash
conda activate sam3d_gui
conda install opencv

# Or with pip
pip install opencv-python
```

**7. "Import error: inference"**

This means SAM 3D Objects is not found. Check:

```bash
# Check submodule
ls external/sam-3d-objects

# Check standalone
ls /home/joon/dev/sam-3d-objects

# Fix: Run setup
./setup.sh
```

### GUI Issues

**8. GUI doesn't start / blank window**

```bash
# Check display
echo $DISPLAY

# Try different backend
export MPLBACKEND=TkAgg
python src/gui_app.py
```

**9. Video preview not showing**

- Check video codec: `ffprobe video.mp4`
- Try different video format
- Update OpenCV: `conda update opencv`

### Performance Issues

**10. Processing is very slow**

```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA version
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**11. High memory usage**

- Reduce video duration
- Increase frame stride
- Process shorter segments
- Close other applications

### Critical Issues (NumPy & SAM2)

**12. "Contour (fallback)" instead of SAM2 segmentation**

This usually indicates SAM2 package import failure. Check:

```bash
# Verify SAM2 can be imported
conda run -n sam3d_gui python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2: OK')"

# If you see NumPy compatibility warnings, check NumPy version
conda run -n sam3d_gui python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# If NumPy >= 2.0, downgrade immediately:
conda run -n sam3d_gui pip install "numpy<2" --force-reinstall

# Verify fix
conda run -n sam3d_gui python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2: OK')"
```

**Why this happens:**
- NumPy 2.x breaks PyTorch 2.0.0 compiled extensions
- SAM2 import fails silently with NumPy 2.x
- System falls back to contour-based segmentation
- **Solution**: Always use NumPy 1.x (< 2.0)

**13. "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"**

```bash
# Fix: Downgrade NumPy
conda run -n sam3d_gui pip install "numpy<2" --force-reinstall

# Ignore these warnings (they're expected):
# - opencv-python requires numpy>=2 (works fine with 1.x)
# - sam-2 requires torch>=2.5.1 (we use --no-deps, so this is OK)
```

**14. CUDA Out of Memory on RTX 3060 12GB**

The application includes automatic memory optimization for 12GB GPUs:

**Features enabled by default:**
- **FP16 Mode**: Reduces SAM 3D memory usage by ~50% (10GB → 5GB)
- **SAM2 Unloading**: Frees ~3GB before SAM 3D inference, reloads after

**If still getting OOM:**
```python
# Check current memory usage
nvidia-smi

# The app will automatically:
# 1. Unload SAM2 models before 3D reconstruction
# 2. Use FP16 mixed precision for SAM 3D
# 3. Reload SAM2 after 3D reconstruction completes
```

**Expected memory profile on RTX 3060:**
- Idle: ~1-2GB
- SAM2 segmentation: ~3-4GB
- SAM 3D reconstruction: ~5-7GB (with FP16 + SAM2 unloaded)
- Total peak: ~7GB (well within 12GB limit)

**15. "ModuleNotFoundError: No module named 'sam2'"**

SAM2 package not installed. This should be handled by setup scripts, but if it persists:

```bash
# Install SAM2 without dependencies (prevents PyTorch upgrade)
conda run -n sam3d_gui pip install --no-deps git+https://github.com/facebookresearch/segment-anything-2.git

# Verify installation
conda run -n sam3d_gui python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2: OK')"

# Check PyTorch wasn't upgraded
conda run -n sam3d_gui python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should be: 2.0.0+cu118
```

---

## 📚 Documentation

### Main Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **README.md** | Complete user manual (this file) | All users |
| **QUICKSTART.md** | 5-minute quick start guide | New users |
| **SETUP_GUIDE.md** | Installation details, Git submodule, Conda | Developers |
| **ARCHITECTURE.md** | System design, data flow | Developers |
| **IMPLEMENTATION_SUMMARY.md** | Implementation details (Korean) | Korean developers |
| **PROJECT_SUMMARY.md** | Project overview (Korean) | Korean users |

### Code Examples

| File | Purpose |
|------|---------|
| **test_pipeline.py** | API usage examples, 6 test cases |
| **example_batch_process.py** | Batch processing, motion detection |

### Quick Links

- [5-Minute Quick Start](QUICKSTART.md)
- [Installation Guide](SETUP_GUIDE.md)
- [System Architecture](ARCHITECTURE.md)
- [Git Submodule Explained](SETUP_GUIDE.md#git-submodule-이해하기)
- [Conda Environment](SETUP_GUIDE.md#conda-환경-관리)

---

## 🔗 Project Management

### Git Submodule

This project uses **Git submodule** for `sam-3d-objects` (PyTorch 2.0 compatible fork):

**Fork repository**: https://github.com/kafkapple/sam-3d-objects

**Daily update (recommended)**:
```bash
git pull --recurse-submodules
```

**First time setup**:
```bash
git clone https://github.com/kafkapple/sam3d_gui.git
cd sam3d_gui
git submodule update --init
```

**If submodule conflicts occur** (after manual edits):
```bash
rm -rf external/sam-3d-objects
git submodule update --init
```

**Advantages**:
- ✅ Single project directory
- ✅ Automatic version tracking
- ✅ PyTorch 2.0 compatibility fixes included
- ✅ No runtime patching needed

### Conda Environment

**Activate**:
```bash
conda activate sam3d_gui
```

**Deactivate**:
```bash
conda deactivate
```

**List environments**:
```bash
conda env list
```

**Recreate environment**:
```bash
conda env remove -n sam3d_gui
conda env create -f environment.yml
```

**Export environment** (for sharing):
```bash
conda env export > environment_exact.yml
```

---

## 🎯 Example Workflows

### Workflow 1: Quick Motion Detection (2 minutes)

**Goal**: Find if an object moves in a video segment

```bash
# 1. Setup (one-time)
./setup.sh

# 2. Launch GUI
./run.sh

# 3. In GUI:
#    - Browse: /home/joon/dev/data/markerless_mouse/
#    - Select: mouse_1/Camera1/0.mp4
#    - Set duration: 3.0s
#    - Click: "Process Video Segment"

# 4. Results panel shows:
#    Motion detected: Yes/No
#    Max displacement: X pixels
```

### Workflow 2: Batch Process Directory (10 minutes)

**Goal**: Find all motion segments in multiple videos

```bash
conda activate sam3d_gui
python example_batch_process.py

# Output:
# ✅ mouse_1/Camera1/0.mp4: 3 motion segments
#    - 0.0s to 3.0s
#    - 6.0s to 9.0s
#    - 12.0s to 15.0s
# ✅ mouse_1/Camera1/3000.mp4: 2 motion segments
# ...
```

### Workflow 3: Full 3D Reconstruction (30 minutes)

**Goal**: Extract 3D mesh from moving object

**Prerequisites**: SAM 3D checkpoints downloaded

```bash
# 1. Download checkpoints (one-time)
# Follow: https://github.com/facebookresearch/sam-3d-objects

# 2. Launch GUI
./run.sh

# 3. Process video
#    - Load video
#    - Process segment (detect motion)

# 4. 3D reconstruction
#    - Navigate to interesting frame
#    - Click "3D Reconstruction"
#    - Wait ~30-60 seconds

# 5. Export and view
#    - Click "Export PLY"
#    - Open: meshlab outputs/reconstruction.ply
```

### Workflow 4: Custom Processing

**Goal**: Integrate into your own pipeline

```python
from src.sam3d_processor import SAM3DProcessor
from pathlib import Path

# Initialize
processor = SAM3DProcessor()

# Your video processing pipeline
videos = Path("/data").glob("*.mp4")

for video in videos:
    # Extract motion segments
    result, _ = processor.process_video_segment(
        video_path=str(video),
        start_time=0.0,
        duration=5.0,
        motion_threshold=30.0
    )

    # Your custom processing
    if result.motion_detected:
        # Extract features
        # Train model
        # Generate report
        pass
```

---

## 🚢 Deployment Guide

### Repository Structure

This project follows a clean Git repository structure with large files properly excluded:

```
sam3d_gui/
├── .gitignore                  # ✅ Excludes checkpoints, outputs, logs
├── src/                        # ✅ Source code (tracked)
├── config/                     # ✅ Configuration files (tracked)
├── docs/                       # ✅ Documentation (tracked)
│   ├── DEPLOYMENT.md           # Detailed deployment guide
│   └── SESSION_MANAGEMENT.md   # Session management guide
├── download_sam3d.sh           # ✅ Checkpoint download script (tracked)
├── run.sh                      # ✅ Launcher (tracked)
├── QUICK_START.md              # ✅ Quick start guide (tracked)
│
├── checkpoints/                # ❌ Excluded from Git (SAM 2, 5-10GB)
├── external/sam-3d-objects/    # ❌ Excluded from Git (SAM 3D, 5-10GB)
├── outputs/                    # ❌ Excluded from Git (results)
└── *.log                       # ❌ Excluded from Git (logs)
```

### Git LFS and Checkpoint Management

#### Understanding the Requirement

SAM 3D Objects requires **Git LFS** (Large File Storage) for downloading 5-10GB model checkpoints from HuggingFace.

**Why sudo is needed for auto-download:**
- GUI's auto-download feature uses `sudo apt-get install git-lfs`
- System package installation requires root privileges
- Without sudo: "Permission denied" errors

**Solutions by environment:**

| Environment | Sudo Available? | Solution | Method |
|-------------|----------------|----------|---------|
| **Development Server** | ✅ Yes | Auto-download | Use GUI's "Generate 3D Mesh" button |
| **Restricted Server** | ❌ No | Pre-download | Use `./download_sam3d.sh` script |
| **Production Server** | ❌ No | Conda install | `conda install -c conda-forge git-lfs` |
| **Docker** | ✅ Yes (as root) | Dockerfile | Include in Docker build |

#### Method 1: Auto-Download (Sudo Required)

**Best for**: Development machines with sudo access

```bash
# 1. Run GUI
./run.sh

# 2. Click "Generate 3D Mesh"
# 3. Checkpoints auto-download if missing
# → GUI shows progress: 0% → 10% → 30% → 90% → 100%
```

#### Method 2: Pre-Download Script (No Sudo)

**Best for**: Production servers without sudo

```bash
# 1. Install Git LFS via Conda (no sudo needed)
conda activate sam3d_gui
conda install -c conda-forge git-lfs
git lfs install

# 2. Run pre-download script
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh

# 3. GUI will use pre-downloaded checkpoints
./run.sh
```

**What `download_sam3d.sh` does:**
- Checks for Git LFS
- Offers installation options (sudo or conda)
- Downloads from HuggingFace to `~/dev/sam-3d-objects/checkpoints/hf/`
- Verifies download completion

#### Method 3: Manual Download and Copy

**Best for**: Air-gapped or highly restricted environments

```bash
# On a machine with internet and Git LFS
cd ~/Downloads
git clone https://huggingface.co/facebook/sam-3d-objects
tar -czf sam3d_checkpoints.tar.gz sam-3d-objects/

# Copy to target server
scp sam3d_checkpoints.tar.gz user@server:/home/user/

# On target server
cd ~
tar -xzf sam3d_checkpoints.tar.gz
mv sam-3d-objects ~/dev/sam-3d-objects/checkpoints/hf
```

### Checkpoint Paths (Config Management)

Checkpoint locations are managed in `config/model_config.yaml`:

```yaml
sam2:
  checkpoint: "${oc.env:HOME}/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
  device: "auto"  # GPU auto-detection with CPU fallback

sam3d:
  checkpoint_dir: "${oc.env:HOME}/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf"
  checkpoint_dir_alt: "${oc.env:HOME}/dev/sam-3d-objects/checkpoints/hf"
```

**The app automatically checks both locations:**
1. Primary: `sam3d_gui/external/sam-3d-objects/` (submodule location)
2. Fallback: `~/dev/sam-3d-objects/` (standalone installation)

### Deployment Checklist

**For new server deployment:**

- [ ] Install Conda environment: `conda env create -f environment.yml`
- [ ] Activate environment: `conda activate sam3d_gui`
- [ ] Install Git LFS: `conda install -c conda-forge git-lfs` (if no sudo)
- [ ] Download checkpoints: `./download_sam3d.sh`
- [ ] Verify config paths: `cat config/model_config.yaml`
- [ ] Test GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Launch GUI: `./run.sh`
- [ ] Access GUI: `http://localhost:7860`
- [ ] Test 3D mesh generation

**For Git repository clone:**

```bash
# Clone repository
git clone https://github.com/your-username/sam3d_gui.git
cd sam3d_gui

# Setup environment
conda env create -f environment.yml
conda activate sam3d_gui

# Download checkpoints
conda install -c conda-forge git-lfs
git lfs install
./download_sam3d.sh

# Run
./run.sh
```

### Port Management

By default, the GUI runs on port 7860. If the port is already in use:

```bash
# Method 1: Environment variable
GRADIO_SERVER_PORT=7861 ./run.sh

# Method 2: Export (persistent for session)
export GRADIO_SERVER_PORT=7862
./run.sh

# Method 3: Kill existing server
pkill -f "web_app.py"
./run.sh
```

### GPU Configuration

**Auto-detection (default):**
- Config: `device: "auto"`
- Behavior: Uses CUDA if available, falls back to CPU
- Supported GPUs: RTX 3060, A6000, any CUDA-compatible GPU

**Verify GPU usage:**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Monitor GPU usage
nvidia-smi
```

**Force CPU mode:**
```yaml
# Edit config/model_config.yaml
sam2:
  device: "cpu"  # Change from "auto" to "cpu"
```

### Detailed Documentation

For comprehensive deployment information, see:
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Full deployment guide with troubleshooting
- **[QUICK_START.md](QUICK_START.md)** - Quick reference for server operations
- **[docs/SESSION_MANAGEMENT.md](docs/SESSION_MANAGEMENT.md)** - Session save/load functionality

---

## 🤝 Contributing

### Project Structure

**Your code** (src/):
- `src/gui_app.py` - GUI application
- `src/sam3d_processor.py` - Processing engine

**External** (external/):
- `external/sam-3d-objects/` - SAM 3D Objects (PyTorch 2.0 compatible fork: https://github.com/kafkapple/sam-3d-objects)

### Adding Features

**1. New segmentation method**:

Edit `src/sam3d_processor.py`:
```python
def segment_object_interactive(self, frame, method='my_method'):
    if method == 'my_method':
        # Your implementation
        return mask
```

**2. New export format**:

Edit `src/sam3d_processor.py`:
```python
def export_mesh(self, output, save_path, format='my_format'):
    if format == 'my_format':
        # Your implementation
        pass
```

**3. GUI enhancements**:

Edit `src/gui_app.py`:
- Follow existing 3-panel structure
- Use `TodoWrite` for progress tracking
- Thread long operations

---

## 📄 License

This project builds upon [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), licensed under the [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE).

---

## 📞 Support & Resources

### Get Help

- **Troubleshooting**: See [Troubleshooting](#-troubleshooting) section above
- **Documentation**: Browse [📚 Documentation](#-documentation)
- **Examples**: Check `test_pipeline.py` and `example_batch_process.py`

### External Resources

- **SAM 3D Objects**: https://github.com/facebookresearch/sam-3d-objects
- **SAM 3D Paper**: https://arxiv.org/abs/2511.16624
- **SAM 3D Demo**: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d

---

## 🎉 Quick Reference

```bash
# ========================================
# Installation (one-time)
# ========================================
./setup.sh              # Automated setup (recommended)
conda env create -f environment.yml  # Manual setup

# ========================================
# Running
# ========================================
./run.sh                # Launch with conda
python src/gui_app.py   # Direct launch

# ========================================
# Testing
# ========================================
python test_pipeline.py              # Run tests
python example_batch_process.py      # Batch example

# ========================================
# Conda Environment
# ========================================
conda activate sam3d_gui    # Activate
conda deactivate            # Deactivate
conda env list              # List environments

# ========================================
# Git Submodule
# ========================================
git submodule update --remote      # Update submodule
git submodule status               # Check status

# ========================================
# 3D Visualization
# ========================================
meshlab outputs/reconstruction.ply        # MeshLab
cloudcompare outputs/reconstruction.ply   # CloudCompare
```

---

## PyTorch 2.0 Compatibility

This project uses a **forked sam-3d-objects** repository with PyTorch 2.0 compatibility fixes. The original Meta repository requires PyTorch 2.1+.

### Fixed Issues (7 total)

| Issue | Error Message | Solution |
|-------|---------------|----------|
| `torch._dynamo` | `AttributeError: module 'torch' has no attribute '_dynamo'` | `hasattr()` check |
| `torch.nn.attention` | `ModuleNotFoundError` | Version detection + fallback |
| Lightning isinstance | `TypeError: isinstance() arg 2 must be a type` | Stub classes + guard |
| `tree_map` args | `takes 2 positional arguments but 3 were given` | Compatibility wrapper |
| `torch.compiler` | `has no attribute 'compiler'` | `getattr()` check |
| `index_add_` dtype | `self (Float) and source (Half) must have same type` | `.to(dtype)` conversion |
| kaolin import | `RuntimeError: Error loading warp.so` | try/except fallback |

### Detailed Documentation

See **[docs/reports/251130_sam3d_pytorch2_compatibility.md](docs/reports/251130_sam3d_pytorch2_compatibility.md)** for:
- Complete error messages and stack traces
- Root cause analysis for each issue
- Code-level solutions with before/after examples
- Best practices for future PyTorch version compatibility

### Upstream Sync (Optional)

To merge upstream changes from Meta's repository:
```bash
cd external/sam-3d-objects
git remote add upstream https://github.com/facebookresearch/sam-3d-objects.git
git fetch upstream
git merge upstream/main
# Resolve conflicts if any, keeping compatibility fixes
git push origin main
```

---

**Status**: ✅ Complete and ready to use
**Version**: 2.4 (PyTorch 2.0 Compatibility)
**Last Updated**: 2025-11-30

**Happy 3D Segmentation!** 🎉

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
│   └── sam-3d-objects/            # Git submodule (auto-managed)
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
- **CUDA** (optional, for GPU acceleration) - Recommended for 3D reconstruction

### Method 1: Automated Setup (Recommended) ⭐

This is the **easiest** way to get started:

```bash
cd /home/joon/dev/sam3d_gui
./setup.sh
```

**What it does**:
1. ✅ Creates isolated Conda environment `sam3d_gui`
2. ✅ Installs all dependencies (PyTorch, OpenCV, etc.)
3. ✅ Sets up `sam-3d-objects` as Git submodule in `external/`
4. ✅ Configures paths automatically
5. ✅ Creates output directories

**Time**: 5-10 minutes (one-time setup)

### Method 2: Manual Installation

If you prefer manual control:

```bash
cd /home/joon/dev/sam3d_gui

# Step 1: Create Conda environment
conda env create -f environment.yml

# Step 2: Activate environment
conda activate sam3d_gui

# Step 3: Setup Git submodule (optional)
git submodule add https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects
git submodule update --init --recursive

# Step 4: Create directories
mkdir -p outputs configs logs
```

### Method 3: Minimal Setup (No Conda)

If you already have Python packages installed:

```bash
cd /home/joon/dev/sam3d_gui

# Install only missing packages
pip install -r requirements.txt

# Run directly
python src/gui_app.py
```

### Existing `sam-3d-objects` Folder?

**Good news**: The code automatically detects both locations!

If you already have `/home/joon/dev/sam-3d-objects/`:
- ✅ **Nothing to do!** Code will use it automatically
- ✅ `external/` can stay empty
- ✅ Optionally run `./setup.sh` to convert to submodule

**Detection order**:
1. `sam3d_gui/external/sam-3d-objects/` (submodule) - checked first
2. `/home/joon/dev/sam-3d-objects/` (standalone) - fallback

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

This project uses **Git submodule** for `sam-3d-objects`:

**Update submodule**:
```bash
git submodule update --remote
```

**Clone this project (with submodule)**:
```bash
git clone --recursive https://github.com/your/repo.git
```

**If you already cloned**:
```bash
git submodule update --init --recursive
```

**Advantages**:
- ✅ Single project directory
- ✅ Automatic version tracking
- ✅ Easy updates
- ✅ Team collaboration

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
- `external/sam-3d-objects/` - SAM 3D Objects (submodule)

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

**Status**: ✅ Complete and ready to use
**Version**: 2.0 (Conda + Git Submodule)
**Last Updated**: 2025-11-22

**Happy 3D Segmentation!** 🎉

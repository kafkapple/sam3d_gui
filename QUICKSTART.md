# Quick Start Guide

**Get started with SAM 3D GUI in 5 minutes!**

## ⚡ One-Command Installation

```bash
cd /home/joon/dev/sam3d_gui
./setup.sh
```

That's it! The script will:
- ✅ Create conda environment `sam3d_gui`
- ✅ Install all dependencies
- ✅ Setup sam-3d-objects as git submodule
- ✅ Configure paths automatically

**Time**: 5-10 minutes

During installation, you'll be asked:
```
Do you want to install PyTorch3D and Kaolin? (y/n)
```
- Choose **y** if you want 3D reconstruction (recommended)
- Choose **n** for video processing only

---

## 🚀 Launch GUI

```bash
./run.sh
```

Or manually:
```bash
conda activate sam3d_gui
python src/gui_app.py
```

---

## 📹 Process Your First Video (2 minutes)

### Step 1: Browse Data Directory
- Click **"Browse..."** next to "Data Directory"
- Navigate to your videos (e.g., `/home/joon/dev/data/markerless_mouse/`)

### Step 2: Load Video
- Select a video from the list (e.g., `mouse_1/Camera1/0.mp4`)
- Click **"Load Video"**

### Step 3: Set Parameters
| Parameter | Value | Why? |
|-----------|-------|------|
| Start Time | `0.0` | Start from beginning |
| Duration | `3.0` | Process 3 seconds |
| Motion Threshold | `50.0` | Detect movement > 50 pixels |
| Segmentation | `contour` | Best for clear backgrounds |

### Step 4: Process
- Click **"Process Video Segment"**
- Wait 10-30 seconds
- View results in Results panel:
  ```
  Motion detected: Yes ✅
  Frames analyzed: 300
  Max displacement: 127.3 pixels
  ```

### Step 5: 3D Reconstruction (Optional)
**Only if you installed PyTorch3D & Kaolin**

- Navigate frames with **<< Prev / Next >>**
- Click **"3D Reconstruction"**
- Wait ~30-60 seconds
- Click **"Export PLY"**
- Open with: `meshlab outputs/reconstruction.ply`

---

## 📂 Output Files

All saved to `outputs/`:
```
outputs/
├── mask_overlay.png          # Segmentation visualization
├── reconstruction.ply        # 3D mesh (if reconstructed)
└── mask_frame_N.png         # Per-frame masks
```

---

## 🎨 View 3D Results

### Option 1: MeshLab
```bash
sudo apt install meshlab
meshlab outputs/reconstruction.ply
```

### Option 2: Online
Upload to: https://3dviewer.net/

### Option 3: Python
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("outputs/reconstruction.ply")
o3d.visualization.draw_geometries([pcd])
```

---

## ✅ What Works Without 3D Checkpoints

**Immediately after `./setup.sh`:**
- ✅ Video loading and preview
- ✅ Frame extraction
- ✅ Object segmentation (3 methods)
- ✅ Object tracking
- ✅ Motion detection
- ✅ Mask visualization

**Requires SAM 3D checkpoints** (download separately):
- ❌ 3D mesh reconstruction
- ❌ Gaussian Splatting
- ❌ PLY/OBJ export

Most features work immediately!

---

## 🔧 Troubleshooting

### "conda: command not found"
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### "Environment sam3d_gui not found"
```bash
./setup.sh
```

### "No module named 'cv2'"
```bash
conda activate sam3d_gui
conda install opencv
```

### GUI doesn't start
```bash
# Check tkinter
python3 -c "import tkinter; print('OK')"

# If error:
sudo apt install python3-tk
```

### "SAM 3D Objects not found"
This is normal! The code auto-detects:
1. `external/sam-3d-objects/` (created by setup.sh)
2. `/home/joon/dev/sam-3d-objects/` (if you already have it)

If both missing, run:
```bash
./setup.sh
```

---

## 📚 Next Steps

### Learn More
- **[README.md](README.md)** - Complete documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[test_pipeline.py](test_pipeline.py)** - API examples

### Try Advanced Features
```bash
# Batch processing
conda activate sam3d_gui
python example_batch_process.py

# Run tests
python test_pipeline.py
```

### Programmatic Usage
```python
from src.sam3d_processor import SAM3DProcessor

processor = SAM3DProcessor()

# Process video
result, reconstruction = processor.process_video_segment(
    video_path="/path/to/video.mp4",
    start_time=0.0,
    duration=3.0,
    motion_threshold=50.0
)

# Check results
if result.motion_detected:
    print(f"✅ Motion detected!")
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Test
```bash
./setup.sh    # One-time
./run.sh      # Every time
# Browse → Load → Process → Done!
```

### Workflow 2: Find All Motion Segments
```bash
conda activate sam3d_gui
python example_batch_process.py
```

### Workflow 3: 3D Reconstruction
Requires: SAM 3D checkpoints downloaded
```bash
./run.sh
# Load video → Process → Navigate to frame → 3D Reconstruction → Export PLY
```

---

## 💡 Key Concepts

### Motion Detection
The app tracks object center points across frames:
```
Frame 1: Object at (100, 200)
Frame 2: Object at (180, 210)
Displacement = √[(180-100)² + (210-200)²] = 80.6 pixels

If threshold = 50.0 → Motion detected ✅
If threshold = 100.0 → No motion ❌
```

### Segmentation Methods
1. **contour** - Automatic (recommended for clear backgrounds)
2. **simple_threshold** - Fast (high contrast scenes)
3. **grabcut** - Accurate (complex backgrounds, slower)

### Project Structure
```
sam3d_gui/                    # Your project
├── external/
│   └── sam-3d-objects/      # Auto-managed by setup.sh
├── src/
│   ├── gui_app.py           # GUI
│   └── sam3d_processor.py   # Processing engine
└── outputs/                  # Your results
```

---

## 🎉 Quick Reference

```bash
# Installation (one-time)
./setup.sh

# Launch GUI
./run.sh

# Activate environment (manual)
conda activate sam3d_gui

# Run tests
python test_pipeline.py

# Batch processing
python example_batch_process.py

# Update sam-3d-objects
git submodule update --remote
```

---

**That's it! You're ready to use SAM 3D GUI.**

For detailed information, see [README.md](README.md)

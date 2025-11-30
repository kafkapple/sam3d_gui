# View-Invariant Mouse Behavior Embedding Research Plan

**Date**: 2024-12-01
**Project**: SAM3D GUI - Multi-View 3D Mouse Reconstruction
**Goal**: Monocular RGB to 3D Mouse Reconstruction with View-Invariant Behavior Embedding

---

## 1. Research Objective

### 1.1 Problem Statement
Mouse behavior analysis in neuroscience requires consistent representation across different camera viewpoints. Current methods depend heavily on camera setup and viewpoint, making cross-session and cross-experiment comparison difficult.

### 1.2 Goal
Train a model that:
1. **Training**: Utilizes multi-view (6 cameras) rich supervision (RGB, mask, mesh, depth, keypoints)
2. **Inference**: Reconstructs 3D mouse shape from **monocular RGB only**
3. **Output**: Generates view-invariant behavior embeddings for downstream analysis

### 1.3 Key Innovation
- Learn 3D geometric priors from multi-view data during training
- Deploy with single-camera input at inference time
- Produce consistent embeddings regardless of viewpoint

---

## 2. Data Pipeline

### 2.1 Training Data (Multi-View)
For each frame, we have 6 synchronized camera views:

| Data Type | Description | Generation Method |
|-----------|-------------|-------------------|
| RGB | Raw camera images | Direct capture |
| Mask | Binary segmentation | SAM2 + propagation |
| Mesh | 3D surface mesh | SAM-3D (TRELLIS) |
| Depth | Depth map | Render from mesh or stereo |
| Keypoints | 2D/3D landmarks | DeepLabCut/SLEAP |

### 2.2 Data Structure
```
training_data/
├── session_001/
│   ├── frame_0000/
│   │   ├── cam_1/
│   │   │   ├── rgb.png
│   │   │   ├── mask.png
│   │   │   ├── depth.png
│   │   │   └── keypoints.json
│   │   ├── cam_2/
│   │   ├── ...
│   │   ├── cam_6/
│   │   ├── mesh.obj          # Single 3D mesh for frame
│   │   └── keypoints_3d.json # Triangulated 3D keypoints
│   ├── frame_0001/
│   └── ...
├── session_002/
└── ...
```

### 2.3 Current Data (mouse_batch_20251128_163151)
- **2 mice** (subject_id: m1, m2)
- **6 cameras** per mouse
- **6 start frames** per camera
- **Total**: 2 × 6 × 6 = 72 video clips
- **Frames per clip**: ~100 frames

---

## 3. Technical Approach

### 3.1 Phase 1: Multi-View 3D Supervision Generation

#### Camera Calibration
```python
from src.reconstruction import CameraCalibrator, MultiCameraSystem

# Load existing Anipose calibration
camera_system = load_anipose_calibration("calibration.toml")

# Or calibrate from ChArUco board
calibrator = CameraCalibrator(board_size=(5, 7), square_size=0.02)
intrinsics = calibrator.calibrate(images)
```

#### 3D Keypoint Triangulation
```python
# Triangulate 2D keypoints to 3D
points_3d = camera_system.triangulate(
    cam_ids=['cam1', 'cam2', 'cam3'],
    points_2d=[kp_cam1, kp_cam2, kp_cam3]
)
```

#### Multi-View Mesh Fusion
```python
from src.reconstruction import MultiViewFusion, fuse_sam3d_meshes

# Fuse per-view SAM-3D meshes
fusion = MultiViewFusion(fusion_method='poisson')
fused_mesh = fuse_sam3d_meshes(
    meshes=[mesh_cam1, mesh_cam2, ...],
    cameras=camera_system
)
```

### 3.2 Phase 2: View-Invariant Model Architecture

#### Option A: Pose-Splatter Style (3D Gaussian Splatting)
```
RGB → Encoder → 3D Gaussians → Differentiable Splatting → Novel Views
                     ↓
              Embedding (view-invariant)
```

Improvements to consider:
1. **Temporal Consistency**: Add temporal loss between consecutive frames
2. **Volume Shape Carving**: Use multi-view silhouettes to constrain 3D shape
3. **3D GS → 2D GS**: Consider 2D Gaussian Splatting for efficiency

#### Option B: Neural Radiance Fields (D-NeRF)
```
RGB → Feature Encoder → Dynamic NeRF → Novel View Synthesis
                              ↓
                        Latent Code (view-invariant)
```

#### Option C: Template Mesh Deformation
```
RGB → CNN → Deformation Field → Template Mesh → Canonical Pose
                                      ↓
                             Embedding (view-invariant)
```

### 3.3 Phase 3: Embedding Extraction

The model produces embeddings that are:
- **View-Invariant**: Same pose → same embedding regardless of camera
- **Pose-Discriminative**: Different poses → different embeddings
- **Temporally Smooth**: Similar poses at t and t+1 → similar embeddings

#### Embedding Network
```python
class ViewInvariantEmbedding(nn.Module):
    def __init__(self, dim=128):
        self.encoder = ResNet50(pretrained=True)
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )

    def forward(self, rgb):
        features = self.encoder(rgb)
        embedding = self.projector(features)
        return F.normalize(embedding, dim=-1)
```

### 3.4 Training Losses

#### Multi-View Consistency Loss
```python
# Embeddings from different views of same frame should match
L_multiview = ||E(view_1) - E(view_2)||^2
```

#### 3D Reconstruction Loss
```python
# Rendered views should match ground truth
L_recon = L1(rendered_rgb, gt_rgb) + L1(rendered_mask, gt_mask)
```

#### Temporal Smoothness Loss
```python
# Consecutive frames should have similar embeddings
L_temporal = ||E(frame_t) - E(frame_{t+1})||^2 * (1 - motion_magnitude)
```

#### Contrastive Loss (Optional)
```python
# Different poses should have different embeddings
L_contrastive = InfoNCE(E(pose_A), E(pose_B), E(negative_samples))
```

---

## 4. Implementation Roadmap

### Stage 1: Data Preparation (Current)
- [x] SAM2 mask generation pipeline
- [x] SAM-3D mesh generation pipeline
- [x] Batch processing for 72 videos
- [ ] Camera calibration integration
- [ ] 3D keypoint triangulation
- [ ] Multi-view mesh fusion

### Stage 2: Supervision Generation
- [ ] Generate consistent 3D meshes per frame
- [ ] Render depth maps from meshes
- [ ] Generate multi-view consistent data

### Stage 3: Model Development
- [ ] Implement encoder architecture
- [ ] Implement 3D representation (Gaussian Splatting or NeRF)
- [ ] Implement embedding extraction head

### Stage 4: Training
- [ ] Multi-view consistency pre-training
- [ ] Full model training with all losses
- [ ] Ablation studies

### Stage 5: Evaluation
- [ ] View-invariance metrics
- [ ] Downstream task evaluation (behavior classification)
- [ ] Comparison with baseline methods

---

## 5. Related Work

### 5.1 Multi-View Animal Reconstruction
- **SMAL/SMALST**: Statistical body model for animals
- **Fauna**: Learning 3D animal shape from video
- **BARC**: Breed-adaptive animal reconstruction from a single image

### 5.2 View-Invariant Representations
- **SimCLR/MoCo**: Contrastive learning for view-invariant features
- **NeRF-W**: Appearance and geometry disentanglement
- **3D Gaussian Splatting**: Efficient novel view synthesis

### 5.3 Animal Behavior Analysis
- **SLEAP**: Multi-animal pose estimation
- **DeepLabCut**: Markerless pose estimation
- **VAME**: Variational Animal Motion Encoding

---

## 6. Evaluation Metrics

### 6.1 View Invariance
- **Cross-View Consistency**: Cosine similarity of embeddings across views
- **Retrieval Accuracy**: Same-pose retrieval from different viewpoints

### 6.2 3D Reconstruction Quality
- **Mask IoU**: 2D silhouette overlap
- **Chamfer Distance**: 3D mesh accuracy
- **PSNR/SSIM**: Rendered image quality

### 6.3 Behavior Analysis
- **Action Classification**: F1 score on behavior labels
- **Temporal Segmentation**: Frame-level behavior labeling

---

## 7. Resources

### 7.1 Hardware
- Training: GPU with 12GB+ VRAM (RTX 3060 12GB available)
- Data storage: ~100GB for full dataset

### 7.2 Software Dependencies
- PyTorch 2.0+
- CUDA 11.8
- Open3D (mesh processing)
- SAM2 (segmentation)
- TRELLIS (3D generation)

### 7.3 Project Files
- `/home/joon/dev/sam3d_gui/` - Main application
- `/home/joon/dev/sam3d_gui/src/reconstruction/` - Multi-view reconstruction modules
- `/home/joon/dev/pose-splatter/` - Reference implementation

---

## 8. Next Steps

1. **Immediate**: Complete camera calibration integration with existing Anipose data
2. **Short-term**: Generate multi-view consistent 3D supervision
3. **Mid-term**: Implement and train view-invariant embedding model
4. **Long-term**: Evaluate on behavior analysis downstream tasks

---

## 9. Notes

### Pose-Splatter Improvements to Consider
1. **Temporal Consistency**: Current model treats each frame independently. Add temporal loss or recurrent architecture.
2. **Volume Shape Carving**: Use multi-view silhouettes to better constrain 3D Gaussian positions.
3. **3D GS → 2D GS**: 2D Gaussian Splatting may be more efficient for surface-based objects like mice.

### Key Technical Decisions
- **Canonical Space**: Define canonical mouse pose for consistent embedding
- **Articulation Model**: How to handle joint articulation (keypoint-based vs. skeleton-based)
- **Scale Normalization**: Ensure consistent scale across different subjects

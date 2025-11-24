#!/bin/bash
# SAM 3D GUI - Complete Setup Script

set -e  # Exit on error

echo "=========================================="
echo "SAM 3D GUI - Complete Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓${NC} Conda found: $(conda --version)"
echo ""

# Step 1: Create conda environment
echo "=========================================="
echo "Step 1: Creating Conda Environment"
echo "=========================================="

if conda env list | grep -q "^sam3d_gui "; then
    echo -e "${YELLOW}Warning: Environment 'sam3d_gui' already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n sam3d_gui -y
    else
        echo "Using existing environment"
        SKIP_ENV_CREATE=1
    fi
fi

if [ -z "$SKIP_ENV_CREATE" ]; then
    echo "Creating new conda environment from environment.yml..."
    conda env create -f environment.yml
    echo -e "${GREEN}✓${NC} Base environment created successfully"
else
    echo -e "${YELLOW}Skipping environment creation${NC}"
fi

echo ""

# Step 1.5: Install PyTorch3D and Kaolin (optional, for SAM 3D)
echo "=========================================="
echo "Step 1.5: Installing PyTorch3D & Kaolin (Optional)"
echo "=========================================="

echo ""
echo "PyTorch3D and Kaolin are required for SAM 3D 3D reconstruction."
echo "They are optional - you can skip this and still use video processing, "
echo "segmentation, and motion tracking."
echo ""
read -p "Do you want to install PyTorch3D and Kaolin? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch3D and Kaolin..."

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate sam3d_gui

    # Install PyTorch3D from conda-forge
    echo "Installing PyTorch3D..."
    conda install -c fvcore -c iopath -c conda-forge pytorch3d -y || {
        echo -e "${YELLOW}Warning: Could not install PyTorch3D from conda${NC}"
        echo "Trying alternative method..."

        # Try pip install with pre-built wheels
        pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" || {
            echo -e "${YELLOW}Warning: PyTorch3D installation failed${NC}"
            echo "You can install it manually later if needed."
        }
    }

    # Install Kaolin
    echo "Installing Kaolin..."
    # Kaolin from PyPI is outdated, using GitHub
    pip install kaolin==0.17.0 || {
        echo -e "${YELLOW}Warning: Kaolin installation failed${NC}"
        echo "You can install it manually later if needed."
    }

    echo -e "${GREEN}✓${NC} PyTorch3D and Kaolin installation attempted"
else
    echo -e "${YELLOW}Skipping PyTorch3D and Kaolin installation${NC}"
    echo "You can still use all features except 3D reconstruction."
fi

echo ""

# Step 2: Setup sam-3d-objects as submodule
echo "=========================================="
echo "Step 2: Setting up SAM 3D Objects"
echo "=========================================="

if [ -d "external/sam-3d-objects" ]; then
    echo -e "${YELLOW}Warning: sam-3d-objects already exists${NC}"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd external/sam-3d-objects
        git pull
        cd ../..
        echo -e "${GREEN}✓${NC} Updated sam-3d-objects"
    fi
else
    echo "Setting up sam-3d-objects as git submodule..."

    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        echo "Initializing git repository..."
        git init
    fi

    # Create external directory
    mkdir -p external

    # Add as submodule or clone directly
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "Adding as git submodule..."
        git submodule add https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects 2>/dev/null || {
            echo -e "${YELLOW}Submodule already exists or git error, trying direct clone...${NC}"
            if [ ! -d "external/sam-3d-objects" ]; then
                git clone https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects
            fi
        }
        git submodule update --init --recursive 2>/dev/null || true
    else
        echo "Cloning sam-3d-objects..."
        git clone https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects
    fi

    echo -e "${GREEN}✓${NC} SAM 3D Objects setup complete"
fi

echo ""

# Step 3: Update processor to use submodule path
echo "=========================================="
echo "Step 3: Configuring Paths"
echo "=========================================="

echo -e "${GREEN}✓${NC} Code already configured to auto-detect paths"
echo "   Priority: external/sam-3d-objects/ → /home/joon/dev/sam-3d-objects/"
echo ""

# Step 4: Instructions for checkpoints
echo "=========================================="
echo "Step 4: SAM 3D Checkpoints (Optional)"
echo "=========================================="

CHECKPOINT_DIR="external/sam-3d-objects/checkpoints/hf"

if [ -f "$CHECKPOINT_DIR/pipeline.yaml" ]; then
    echo -e "${GREEN}✓${NC} SAM 3D checkpoints found"
else
    echo -e "${YELLOW}Note: SAM 3D checkpoints not found${NC}"
    echo ""
    echo "To enable 3D reconstruction, you need to download the checkpoints:"
    echo ""
    echo "1. Visit: https://github.com/facebookresearch/sam-3d-objects"
    echo "2. Follow the checkpoint download instructions"
    echo "3. Place checkpoints in: $CHECKPOINT_DIR"
    echo ""
    echo "The application will work WITHOUT checkpoints for:"
    echo "  ✓ Video processing"
    echo "  ✓ Object segmentation"
    echo "  ✓ Motion tracking"
    echo ""
    echo "Checkpoints are REQUIRED for:"
    echo "  ✗ 3D mesh reconstruction"
    echo ""
fi

echo ""

# Step 5: Create output directories
echo "=========================================="
echo "Step 5: Creating Directories"
echo "=========================================="

mkdir -p outputs
mkdir -p configs
mkdir -p logs

echo -e "${GREEN}✓${NC} Directories created"
echo ""

# Final instructions
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment and run the GUI:"
echo ""
echo -e "${GREEN}  conda activate sam3d_gui${NC}"
echo -e "${GREEN}  python src/gui_app.py${NC}"
echo ""
echo "Or use the launcher script:"
echo ""
echo -e "${GREEN}  ./run.sh${NC}"
echo ""
echo "For more information, see:"
echo "  - README.md        (Complete documentation)"
echo "  - QUICKSTART.md    (Quick start guide)"
echo ""
echo "Project structure:"
echo "  sam3d_gui/                   (Your project)"
echo "  ├── src/                     (Your code)"
echo "  ├── external/                (External dependencies)"
echo "  │   └── sam-3d-objects/      (Git submodule)"
echo "  └── outputs/                 (Generated files)"
echo ""

# Summary
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo ""
conda activate sam3d_gui 2>/dev/null || true
python -c "
import sys
print('Python:', sys.version.split()[0])

try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
except:
    print('PyTorch: Not installed')

try:
    import cv2
    print('OpenCV:', cv2.__version__)
except:
    print('OpenCV: Not installed')

try:
    import pytorch3d
    print('PyTorch3D: Installed ✓')
except:
    print('PyTorch3D: Not installed (optional)')

try:
    import kaolin
    print('Kaolin: Installed ✓')
except:
    print('Kaolin: Not installed (optional)')
" 2>/dev/null || echo "Environment not activated"

echo ""
echo -e "${GREEN}Ready to use!${NC}"

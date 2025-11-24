#!/bin/bash
# SAM 3D GUI - 통합 웹 인터페이스 Launcher

echo "=========================================="
echo "SAM 3D GUI - Unified Web Interface"
echo "=========================================="
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "^sam3d_gui "; then
    echo "Error: Conda environment 'sam3d_gui' not found"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    echo ""
    exit 1
fi

# Activate conda environment and run
echo "Activating conda environment: sam3d_gui"
echo "Launching web interface..."
echo ""
echo "📱 Access the GUI at:"
echo "  - Local:  http://localhost:7860"
echo "  - Network: http://$(hostname -I | awk '{print $1}'):7860"
echo ""
echo "🎬 Features:"
echo "  Tab 1: 🚀 Quick Mode - 자동 세그멘테이션 & 모션 감지"
echo "  Tab 2: 🎨 Interactive Mode - Point annotation & Propagation"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Use conda run for better compatibility
conda run -n sam3d_gui python src/web_app.py

echo ""
echo "Server stopped."

#!/bin/bash

echo "🧹 Cleaning up CUDA/nvidia packages from venv"
echo ""

# Activate venv
source venv/bin/activate

# Show current disk usage
echo "Current venv size:"
du -sh venv/
echo ""

# Uninstall torch with CUDA
echo "Uninstalling PyTorch with CUDA..."
pip uninstall -y torch torchvision torchaudio

# Uninstall nvidia-related packages
echo ""
echo "Uninstalling nvidia packages..."
pip list | grep -i nvidia | awk '{print $1}' | xargs -r pip uninstall -y

# Reinstall PyTorch CPU-only
echo ""
echo "Installing PyTorch CPU-only..."
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Reinstall sentence-transformers
echo ""
echo "Reinstalling sentence-transformers..."
pip install sentence-transformers==3.3.1

# Clean pip cache
echo ""
echo "Cleaning pip cache..."
pip cache purge

# Show final disk usage
echo ""
echo "Final venv size:"
du -sh venv/
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Space saved:"
echo "Run: df -h"

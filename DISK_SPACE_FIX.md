# Disk Space Issue: CUDA/nvidia Packages

## Problem

`sentence-transformers` package pulls in PyTorch, which by default installs CUDA/nvidia GPU support (~2-3GB). Since your server doesn't have a GPU, this wastes disk space.

## Solution

Use CPU-only PyTorch to save ~2GB of disk space.

## Quick Fix on Server

### Option 1: Clean Existing venv
```bash
cd /home/vagarwal/tiagent/tiagent
./cleanup-cuda.sh
```

This will:
- Uninstall PyTorch with CUDA
- Remove nvidia packages
- Install PyTorch CPU-only
- Clean pip cache

### Option 2: Fresh venv (Recommended)
```bash
cd /home/vagarwal/tiagent/tiagent
rm -rf venv/
./deploy-pm2.sh
```

The updated `deploy-pm2.sh` now uses `requirements-cpu.txt` which installs CPU-only versions.

## Check Disk Space

### Before Cleanup
```bash
# Check venv size
du -sh venv/
# Likely: ~3-4GB

# Check for nvidia packages
source venv/bin/activate
pip list | grep -i nvidia
pip list | grep -i torch
```

### After Cleanup
```bash
# Check venv size
du -sh venv/
# Should be: ~1-1.5GB

# Check PyTorch version
source venv/bin/activate
pip list | grep torch
# Should show: torch 2.2.0+cpu
```

## Files Changed

- **`requirements-cpu.txt`** - New CPU-only requirements
- **`deploy-pm2.sh`** - Updated to use requirements-cpu.txt
- **`cleanup-cuda.sh`** - Script to remove CUDA packages
- **`requirements.txt`** - Updated with missing langchain-google-genai

## Disk Space Savings

- **Before**: ~3-4GB venv (with CUDA)
- **After**: ~1-1.5GB venv (CPU-only)
- **Saved**: ~2GB

## Why This Happened

When you run `pip install sentence-transformers`, it installs:
1. PyTorch (default: GPU/CUDA version ~2GB)
2. nvidia-cublas, nvidia-cuda-runtime, etc. (~500MB-1GB)
3. Other dependencies

By specifying `torch==2.2.0+cpu` FIRST (before sentence-transformers), pip uses the CPU-only version instead.

## Prevention

Always use `requirements-cpu.txt` on servers without GPUs:
```bash
pip install -r requirements-cpu.txt
```

## Manual Cleanup Commands

If you want to do it manually:
```bash
source venv/bin/activate

# Uninstall CUDA packages
pip uninstall -y torch torchvision torchaudio
pip list | grep nvidia | awk '{print $1}' | xargs pip uninstall -y

# Install CPU-only
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Reinstall sentence-transformers
pip install sentence-transformers==3.3.1

# Clean cache
pip cache purge
```

## Verify It's Working

After cleanup, test the backend:
```bash
source venv/bin/activate
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output:
```
PyTorch version: 2.2.0+cpu
CUDA available: False
```

This is correct! You don't need CUDA on a CPU-only server.

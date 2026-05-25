#!/bin/bash
set -e

ENV_NAME="align3d_test"

echo "Cloning repositories..."
git clone https://github.com/RotemGat/CopyTransformPaste.git
cd CopyTransformPaste

git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/RotemGat/nvdiffmodeling.git

echo "Creating conda environment..."
conda create -y -n ${ENV_NAME} python=3.9

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Installing PyTorch..."
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing local nvdiffrast..."
pip install --no-build-isolation -e ./nvdiffrast

echo "Adding local nvdiffmodeling to PYTHONPATH..."
export PYTHONPATH="$PWD/nvdiffmodeling:$PYTHONPATH"

echo "Installing PyTorch3D..."
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo
echo "========== VALIDATION =========="

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import clip
import nvdiffrast.torch
import pytorch3d
import trimesh
import kornia
import pymeshlab
import diffusers
print("✓ Core libraries imported")
PY

echo
echo "Running hotdog example..."
python main.py --config configs/PairBench3D/hotdog.yaml

echo
echo "====================================="
echo "INSTALLATION SUCCESSFUL"
echo "====================================="
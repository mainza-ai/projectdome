#!/bin/bash
set -e

echo "=== Step 1: Creating venv with Python 3.12 via uv ==="
/opt/homebrew/bin/uv venv venv --python 3.12

echo "=== Step 2: Activating virtual environment ==="
source venv/bin/activate

echo "=== Step 3: Updating git submodules ==="
git submodule update --init --recursive

echo "=== Step 4: Installing GNM shape package ==="
/opt/homebrew/bin/uv pip install -e vendor/GNM/gnm/shape

echo "=== Step 5: Installing dev and additional dependencies ==="
/opt/homebrew/bin/uv pip install trimesh numpy matplotlib onnxruntime

echo "=== Step 6: Verifying installation ==="
python -c "from gnm.shape.gnm_numpy import GNM; print('GNM Head installed.')"

echo "=== Step 7: Exporting web buffers ==="
./venv/bin/python tools/export_basis.py

echo ""
echo "=== Setup complete! Run ./venv/bin/python src/server.py ==="

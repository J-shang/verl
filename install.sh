#!/bin/bash
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

cd $SCRIPT_DIR/../

# install vllm v0.6.3
git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout v0.6.3 && python setup.py develop && cd ..

# install & patch ray
pip install ray==2.38.0
RAY_DIR=$(python -c "import ray; print(ray.__path__[0])")
TARGET_FILE="$RAY_DIR/_private/accelerators/amd_gpu.py"
NEW_FILE="$SCRIPT_DIR/patch/amd_gpu.py"
cp "$NEW_FILE" "$TARGET_FILE"

cd $SCRIPT_DIR && pip install -e .

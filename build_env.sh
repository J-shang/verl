wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p miniconda3
source miniconda3/bin/activate
conda init
conda create -y -n verl python=3.12
cp -r /usr/local/lib/python3.12/dist-packages/* miniconda3/envs/verl/lib/python3.12/site-packages/

conda activate verl

pip install ray==2.38.0

pip uninstall -y vllm
export CCACHE_DIR=/scratch/nishang/ccahe
git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout v0.6.3 && python setup.py develop && cd ..

git clone https://github.com/pytorch/tensordict.git && cd tensordict && pip install . && cd ..

git clone https://github.com/J-shang/verl.git && cd verl && git checkout nishang/add-simplerl-reward && cd ..

RAY_DIR=$(python -c "import ray; print(ray.__path__[0])")
TARGET_FILE="$RAY_DIR/_private/accelerators/amd_gpu.py"
NEW_FILE="verl/patch/amd_gpu.py"
cp "$NEW_FILE" "$TARGET_FILE"

cd verl && pip install -e .

rm miniconda3/envs/verl/bin/../lib/libstdc++.so.6

# download qwen math model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B')"

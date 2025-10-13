#!/bin/bash
#SBATCH --job-name=cjs-training
#SBATCH -o cjs-training-%J.txt
#SBATCH -M kale
#SBATCH -p gpu
#SBATCH --cpus-per-gpu=2
#SBATCH -G1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=168:00:00

### Finetunes the cjs model
cd /wrk/users/aarneris/NetLLM

### Install Anaconda to manage conda environments (if necessary)
# Needed as conda uses the home path for some operations
export HOME=/wrk/users/aarneris/NetLLM
cd miniconda3
# download miniconda installer if not loaded yet
# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -u -b

# Activate conda if installed
. bin/activate

cd /wrk/users/aarneris/NetLLM/NetLLM/cluster_job_scheduling
conda create -n cjs_netllm python==3.11.9 -y
conda activate cjs_netllm

# Run the commands as described in the original repo

##################################

# For Pytorch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# For PyG
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric

# For Gymnasium
conda install swig -y
pip install "Gymnasium[all]"

# For other packages
pip install numpy==1.26.4
pip install transformers==4.37.1
pip install munch==4.0.0
pip install openprompt==1.0.1
pip install peft==0.13.2

##################################



# Run the training from start
# Adding model dir as an argument
python run_plm.py \
    --train \
    --test \
    --seed 666 \
    --plm-type llama \
    --plm-size base \
    --device cuda:0 \
    --device-out cuda:0 \
    --state-feature-dim 256 \
    --K 20 \
    --gamma 1.0 \
    --lr 0.0001 \
    --model-dir ../downloaded_plms/llama/base/ \
    --num-iters 40 
    
 # --freeze-encoder not supported

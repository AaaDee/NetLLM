#!/bin/bash
#SBATCH --job-name=cjs-training-lm3
#SBATCH -o cjs-training-lm3-%J.txt
#SBATCH -M ukko
#SBATCH -p gpu
#SBATCH --cpus-per-gpu=2
#SBATCH -G1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=48:00:00


### Finetunes the cluster job scheduling model
cd  /turso/wrk-vakka/users/aarneris/NetLLM/

### Install Anaconda to manage conda environments (if necessary)
# Needed as conda uses the home path for some operations
export HOME=/turso/wrk-vakka/users/aarneris/NetLLM/
cd miniconda3
# download miniconda installer if not loaded yet
# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -u -b

# Activate conda if installed
. bin/activate

cd /turso/wrk-vakka/users/aarneris/NetLLM/NetLLM/cluster_job_scheduling
conda env create -f environment_3.yaml
conda activate cjs_netllm_3

conda install -c conda-forge gxx -y

# For Gymnasium
conda install swig -y
pip install "Gymnasium[all]"

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.8.0+cu129
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.8.0+cu129
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.8.0+cu129
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.8.0+cu129
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple_torch_geometric

pip install numpy==2.1.0
pip install munch==4.0.0
pip install openprompt==1.0.1
pip install peft==0.13.2
pip install torch_geometric==2.7.0

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
    --model-dir ../../Llama-3.1-8B\
    --num-iters 40 
    
 # --freeze-encoder not supported

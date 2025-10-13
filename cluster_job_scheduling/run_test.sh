#!/bin/bash
#SBATCH --job-name=cjs-test
#SBATCH -o cjs-test-%J.txt
#SBATCH -M kale
#SBATCH -p gpu
#SBATCH --cpus-per-gpu=2
#SBATCH -G1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=168:00:00


### Finetunes the adaptive bitrate streaming model
cd /wrk-kappa/users/aarneris/NetLLM

### Install Anaconda to manage conda environments (if necessary)
# Needed as conda uses the home path for some operations
export HOME=/wrk-kappa/users/aarneris/NetLLM
cd miniconda3
# download miniconda installer if not loaded yet
# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -u -b

# Activate conda if installed
. bin/activate

cd /wrk-kappa/users/aarneris/NetLLM/NetLLM/cluster_job_scheduling
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
    --test \
    --plm-type llama \
    --plm-size base \
    --state-feature-dim 256 \
    --device cuda:0 \
    --model-dir ./artifacts/ft_plms/llama_base/artifacts_exp_pool_exp_pool_ss_None/peft_128_K_20_gamma_1.0_sfd_256_exec_50_stage_100_lr_0.0001_wd_0.0001_warm_2000_iters_40_steps_10000_seed_666/early_stop_-1_best_model


  # --peft-rank 128 \ unsupported
    
 # --freeze-encoder not supported

#!/bin/bash
#SBATCH --job-name=abr-training
#SBATCH -o abr-training-lm4-%J.txt
#SBATCH -M ukko
#SBATCH -p gpu
#SBATCH --cpus-per-gpu=2
#SBATCH -G1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=48:00:00


### Finetunes the adaptive bitrate streaming model
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

cd /wrk/users/aarneris/NetLLM/NetLLM/adaptive_bitrate_streaming
conda env create -f environment.yaml
conda activate abr_netllm

pip install -r requirements.txt
# Has to be installed separately due to a dependency conflict
pip install huggingface-hub==0.23.0 --no-dependencies

# Run the training from start
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama4 --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 --plm-path ../../llama4/Llama-4-Scout-17B-16E
 

# Run interactive
# srun --interactive --cpus-per-gpu=2 --mem-per-cpu=8000 -G1 -t08:00:00 -p gpu -M kale --pty bash
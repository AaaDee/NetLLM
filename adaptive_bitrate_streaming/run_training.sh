#!/bin/bash
#SBATCH --job-name=abr-training
#SBATCH -o abr-training-%J.txt
#SBATCH  -M kale
#SBATCH  -p gpu
#SBATCH --cpus-per-gpu=2
#SBATCH -G=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=96:00:00


### Finetunes the adaptive bitrate streaming model
cd /turso/wrk-vakka/users/aarneris/NetLLM

### Install Anaconda to manage conda environments (if necessary)
# Needed as conda uses the home path for some operations
export HOME=/turso/wrk-vakka/users/aarneris/NetLLM
cd miniconda3
# download miniconda installer if not loaded yet
# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -u -b

# Activate conda if installed
. bin/activate

cd /turso/wrk-vakka/users/aarneris/NetLLM/NetLLM/adaptive_bitrate_streaming
conda env create -f environment.yaml
conda activate abr_netllm

pip install -r requirements.txt
# Has to be installed separately due to a dependency conflict
pip install huggingface-hub==0.23.0 --no-dependencies

# Run the training from start
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 

# Run interactive
# srun --interactive --cpus-per-gpu=2 --mem-per-cpu=8000 -G1 -t08:00:00 -pgpu -M ukko --pty bash
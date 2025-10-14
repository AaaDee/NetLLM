#!/bin/bash
#SBATCH --job-name=viewport-training-lm4
#SBATCH -o viewport-training-lm4-%J.txt
#SBATCH  -M ukko
#SBATCH  -p gpu
#SBATCH --cpus-per-gpu=2
#SBATCH -G1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=48:00:00


### Finetunes the viewport preciction model
cd /turso/wrk-vakka/users/aarneris/NetLLM

### Install Anaconda to manage conda environments (if necessary)
# Needed as conda uses the home path for some operations
export HOME=/turso/wrk-vakka/users/aarneris/NetLLM
cd miniconda3
# download miniconda installer if not loaded yet
# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -u -b
. bin/activate

cd /turso/wrk-vakka/users/aarneris/NetLLM/NetLLM/viewport_prediction
conda env create -f environment.yaml
conda activate vp_netllm

pip install -r requirements.txt
# Has to be installed separately due to a dependency conflict
pip install huggingface-hub==0.23.0 --no-dependencies

pip install transformers==4.57.0

# Run the training from start
python run_plm.py --adapt --train-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama4 --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --plm-path ../../llama4/Llama-4-Scout-17B-16E
 


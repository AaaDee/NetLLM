### Runs the adaptive bitrate streaming test code with a llm model checkpoint
srun --interactive -c4 --mem=32G -G1 -t04:00:00 -pgpu-oversub -M ukko --pty bash
cd /turso/wrk-vakka/users/aarneris/NetLLM

### Install Anaconda to manage conda environments (if necessary)
# Needed as conda uses the home path for some operations
export HOME=/turso/wrk-vakka/users/aarneris/NetLLM
cd miniconda3
# download miniconda installer if not loaded yet
# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -u -b
. bin/activate

cd /turso/wrk-vakka/users/aarneris/NetLLM/NetLLM/adaptive_bitrate_streaming
conda env create -f environment.yaml
conda activate vp_netllm

pip install -r requirements.txt
# Has to be installed separately due to a dependency conflict
pip install huggingface-hub==0.23.0 --no-dependencies
# Torch also needs to be installed separately
python -m pip install torch==1.10.2
# Test command as presented in the NetLLM repository

# Run with the best model checkpoint
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir  data/ft_plms/llama_base/artifacts_exp_pools_ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_80_seed_100003/early_stop_-1_best_model

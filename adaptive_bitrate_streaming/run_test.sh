### Runs the adaptive bitrate streaming test code with a llm model checkpoint
srun --interactive -c4 --mem=32G -G4 -t04:00:00 -pgpu-oversub -M ukko --pty bash
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
# Note that you will need the model checkpoint downloaded separately, please see instructions in the abr repository
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir  data/ft_plms/try_llama2_7b



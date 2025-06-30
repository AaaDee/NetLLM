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

cd /turso/wrk-vakka/users/aarneris/NetLLM/NetLLM/viewport_prediction
conda env create -f environment.yaml
conda activate vp_netllm

pip install -r requirements.txt
# Has to be installed separately due to a dependency conflict
pip install huggingface-hub==0.23.0 --no-dependencies
# Test command as presented in the NetLLM repository
# Note that you will need the model checkpoint downloaded separately, please see instructions in the vp repository
python run_old.py --test --test-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --model-path data/ft_plms/try_llama2_7b

# Copy files back to local
rsync -av --progress -e "ssh -A aarneris@pangolin.it.helsinki.fi ssh" aarneris@turso.cs.helsinki.fi:/wrk-vakka/users/aarneris/NetLLM/NetLLM/viewport_prediction/data/results/llama_base/ .



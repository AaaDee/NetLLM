# Runs the abr training for llama3
# To be run on a CUDA docker image (or similar) with most of the tools preinstalled

conda env create -f environment_3.yaml
conda activate abr_netllm_3

# Run the training from start
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 --plm-path ../../Llama-3.1-8B
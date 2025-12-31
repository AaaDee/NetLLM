# Run viewport prediction on a CUDA docker image

conda env create -f environment_3.yaml
conda activate cjs_netllm_3

# Run the training from start
python run_plm.py --adapt --train-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --plm-path ../../Llama-3.1-8B
 


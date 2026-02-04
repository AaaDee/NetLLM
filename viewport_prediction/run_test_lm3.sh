conda activate vp_netllm_3

python run_plm.py --test --test-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --model-path data/ft_plms/llama_base_low_rank/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_24_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True/best_model --plm-path ../../Llama-3.1-8B

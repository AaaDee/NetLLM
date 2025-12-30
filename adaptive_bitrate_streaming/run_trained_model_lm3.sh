## To be run on a cuda image with the files ready on a typical remote machine. Not runnable on Turso as the prerequisities are not installed.

cd /workspace/Thesis/NetLLM/adaptive_bitrate_streaming
conda activate abr_netllm_3


# Run with the best model checkpoint (from training)
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir  data/ft_plms/llama_base/artifacts_exp_pools_ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_80_seed_100003/early_stop_-1_best_model --plm-path ../../Llama-3.1-8B


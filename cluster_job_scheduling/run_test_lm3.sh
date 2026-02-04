# Assume that the environmnet created during the training is present
conda activate cjs_netllm_3

# Run the training from start
# Adding model dir as an argument
python run_plm.py \
    --test \
    --plm-type llama \
    --plm-size base \
    --state-feature-dim 256 \
    --device cuda:0 \
    --model-dir ./artifacts/ft_plms/llama_base/artifacts_exp_pool_exp_pool_ss_None/peft_128_K_20_gamma_1.0_sfd_256_exec_50_stage_100_lr_0.0001_wd_0.0001_warm_2000_iters_40_steps_10000_seed_666/early_stop_-1_best_model \
    --plm-dir ../../Llama-3.1-8B 

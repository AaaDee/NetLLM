#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np

path = '/home/aarne/Thesis/NetLLM/cluster_job_scheduling/artifacts/results/tpch/exe_50_cap_200_rate_4e-05_md_2000.0_wd_1000.0_env_seed_1/early_stop_-1_results_dt_llama_base_peft_128_K_20_gamma_1.0_tgt_scale_1.0_seed_1.pkl'

with open(path, 'rb') as f:
   raw_data = pickle.load(f)
    
    
data = raw_data['job_durations2']


mean = np.mean(data)
sem = stats.sem(data)
n = len(data)

# Using 95% confidence
lower_cutoff = stats.t.ppf(0.025, n - 1, loc = mean, scale = sem)
upper_cutoff = stats.t.ppf(0.975, n - 1, loc = mean, scale = sem) 

results = {
    'mean': mean,
    'ci_lower': lower_cutoff,
    'ci_upper': upper_cutoff,
    'interval_size': upper_cutoff - mean
}
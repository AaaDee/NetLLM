#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import statistics
import scipy.stats as stats
import os

def calculate_cjs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(input_path, 'rb') as f:
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

    paper_values = f"mean: {results['mean']} interval size: {results['interval_size']}"

    f = open(f"{output_path}/paper_values.txt", "w")
    f.write(paper_values)

input_path = './data/trained/CJS/early_stop_-1_results_dt_llama_base_peft_128_K_20_gamma_1.0_tgt_scale_1.0_seed_1.pkl'
output_path = '../results/trained/CJS'

calculate_cjs(input_path, output_path)

input_path = './data/llama3/CJS/early_stop_-1_results_dt_llama_base_peft_128_K_20_gamma_1.0_tgt_scale_1.0_seed_1.pkl'
output_path = '../results/llama3/CJS'

calculate_cjs(input_path, output_path)
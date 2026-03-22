
import os
import re
import pandas as pd
import statistics
import scipy.stats as stats

def calculate_vp_result(input_filepath, output_filepath):
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
        
    df = pd.read_csv(input_filepath, sep=',')
         
    print(statistics.mean(df['mae']))
    
    mean = statistics.mean(df['mae'])
    sem = stats.sem(df['mae'])
    n = len(df['mae'])
    sd = statistics.stdev(df['mae'])
    
    # Using 95% confidence
    lower_cutoff = stats.t.ppf(0.025, n - 1, loc = mean, scale = sem)
    upper_cutoff = stats.t.ppf(0.975, n - 1, loc = mean, scale = sem)
    
    interval_size = upper_cutoff - mean
    
    paper_values = f"mean: {mean} interval size: {interval_size}, sd: {sd}, n: {n}"
    f = open(f"{output_filepath}/paper_values.txt", "w")
    f.write(paper_values)


input_path = './data/checkpoint/VP/his_10_fut_20_ss_15_epochs_40_bs_32_lr_0.0002_seed_1_rank_32_teacher_forcing_False_scheduled_sampling_True_results.csv'
output_path = '../results/checkpoint/VP'

calculate_vp_result(input_path, output_path)

input_path = './data/trained/VP/his_10_fut_20_axes_ss_15_epochs_40_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_results.csv'
output_path = '../results/trained/VP'

calculate_vp_result(input_path, output_path)

input_path = './data/llama3/VP/his_10_fut_20_axes_ss_15_epochs_40_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_results.csv'
output_path = '../results/llama3/VP'

calculate_vp_result(input_path, output_path)
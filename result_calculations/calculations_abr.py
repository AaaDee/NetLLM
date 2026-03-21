from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import os


def calculate_abr_result(input_filepath, output_filepath):
    # create output dir
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
        
   ### Read data

    
    
    folder_path = Path(input_path)
    
    dfs = []
    
    # Column names as documented in test.py
    column_names = ['time_stamp', 'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time', 'smoothness', 'reward']
    
    for file in folder_path.iterdir():
        df = pd.read_csv(file, sep='\t', names=column_names)
        # Drop first row as in the original paper
        df = df.iloc[1:]
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    ### Calculate results
    results = {}
    
    for col in combined_df.select_dtypes(include='number'):
        data = combined_df[col]
        mean = np.mean(data)
        sem = stats.sem(data)
        n = len(data)
    
        # Using 95% confidence
        lower_cutoff = stats.t.ppf(0.025, n - 1, loc = mean, scale = sem)
        upper_cutoff = stats.t.ppf(0.975, n - 1, loc = mean, scale = sem) 
       
        results[col] = {
            'mean': mean,
            'ci_lower': lower_cutoff,
            'ci_upper': upper_cutoff,
            'interval_size': upper_cutoff - mean
        }
    
    ci_df = pd.DataFrame(results).T
    print(ci_df)
    
    # ABR results for model checkpoint replication
    print(results['reward']['mean'])
    print(results['reward']['interval_size'])
    
    csv_path = f"{output_filepath}/results.csv"
    ci_df.to_csv(csv_path)
    paper_values = f"mean: {results['reward']['mean']} interval size: {results['reward']['interval_size']}"
    
    f = open(f"{output_filepath}/paper_values.txt", "w")
    f.write(paper_values)
    
    
    
input_path = './data/checkpoint/ABR/early_stop_-1_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_100003'
output_path = '../results/checkpoint/ABR'

calculate_abr_result(input_path, output_path)

input_path = './data/trained/ABR/early_stop_-1_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_100003'
output_path = '../results/trained/ABR'

calculate_abr_result(input_path, output_path)

input_path = './data/llama3/ABR/early_stop_-1_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_100003'
output_path = '../results/llama3/ABR'

calculate_abr_result(input_path, output_path)


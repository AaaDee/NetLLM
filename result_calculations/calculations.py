from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

### Read data

# Note that test conditions and seed are identified by the filepath
folder = '../adaptive_bitrate_streaming/artifacts/results/fcc-test_video1/trace_num_100_fixed_True/llama_base/early_stop_-1_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_100003'
folder_path = Path(folder)

dfs = []

# Column names as documented in test.py
column_names = ['time_stamp', 'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time', 'smoothness', 'reward']

for file in folder_path.iterdir():
    df = pd.read_csv(file, sep='\t', names=column_names)
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

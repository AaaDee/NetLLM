
import re
import pandas as pd
    # create output dir
#    if not os.path.exists(output_filepath):
#        os.makedirs(output_filepath)
        
   ### Read data

 input_path = '../viewport_prediction/results/llama_base/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_40_bs_32_lr_0.0002_seed_1_rank_32_teacher_forcing_False_scheduled_sampling_True_results.csv'

 with open(input_path, "r") as f:
     lines = f.readlines()

     
print(lines)  

# Any decimal value
pattern = r"-?\d*\.\d+"


df = pd.DataFrame(values)


# Values are recorded in groups of 20 for each axis
df_1 = df.iloc[:, :20]
df_2 = df.iloc[:, 20:40]
df_3 = df.iloc[:, 40:]

# Split odd (pred) and even (gt) rows

df_1_pred = df_1.iloc[::2]
df_1_gt = df_1.iloc[1::2]

df_2_pred = df_2.iloc[::2]
df_2_gt = df_2.iloc[1::2]

df_3_pred = df_3.iloc[::2]
df_3_gt = df_3.iloc[1::2]

# Format into single columns
df_1_pred = pd.DataFrame(df_1_pred.values.ravel(), columns=["1P"])
df_1_gt = pd.DataFrame(df_1_gt.values.ravel(), columns=["1G"])

df_2_pred = pd.DataFrame(df_2_pred.values.ravel(), columns=["2P"])
df_2_gt = pd.DataFrame(df_2_gt.values.ravel(), columns=["2G"])

df_3_pred = pd.DataFrame(df_3_pred.values.ravel(), columns=["3P"])
df_3_gt = pd.DataFrame(df_3_gt.values.ravel(), columns=["3G"])


combined = pd.concat([df_1_pred, df_1_gt, df_2_pred, df_2_gt, df_3_pred, df_3_gt], axis=1)

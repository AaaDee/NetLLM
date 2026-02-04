#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed calculations for the VP task, which use the raw data instead of the preprocessed files
"""

# Detailed vp
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

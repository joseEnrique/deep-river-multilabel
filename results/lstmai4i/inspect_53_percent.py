import pandas as pd
import numpy as np

# Paths
path_bce = 'lstm_bce.csv'
path_focal = 'lstm_focal_full_20251209_035057.csv'

# Load
df_bce = pd.read_csv(path_bce)
df_focal = pd.read_csv(path_focal)

# Normalize columns
df_bce.columns = df_bce.columns.str.lower().str.replace(' ', '_')
if 'model' in df_bce.columns:
    df_bce = df_bce.rename(columns={'model': 'name'})
df_focal.columns = df_focal.columns.str.lower().str.replace(' ', '_')

# Merge
merge_cols = ['hidden_dim', 'num_layers', 'bidirectional', 'lr', 'dropout', 'window_size']
df_merged = pd.merge(
    df_focal,
    df_bce[merge_cols + ['micro_f1', 'macro_f1']],
    on=merge_cols,
    how='left',
    suffixes=('_focal', '_bce')
)

df_merged['delta_micro_f1'] = df_merged['micro_f1_focal'] - df_merged['micro_f1_bce']

# 1. Global Stat
wins = (df_merged['delta_micro_f1'] > 0).sum()
total = len(df_merged)
print(f"Global Win Rate: {wins}/{total} ({wins/total:.1%})")

# 2. Specific Config: Alpha=0.5, Gamma=3.0
subset = df_merged[
    (df_merged['focal_alpha'] == 0.5) &
    (df_merged['focal_gamma'] == 3.0)
]
wins_sub = (subset['delta_micro_f1'] > 0).sum()
total_sub = len(subset)
print(f"Alpha=0.5, Gamma=3.0 Win Rate: {wins_sub}/{total_sub} ({wins_sub/total_sub:.1%})")

# 3. Most Common Winning Parameters
winners = df_merged[df_merged['delta_micro_f1'] > 0]
print("\nMost Common Winning Alpha:")
print(winners['focal_alpha'].value_counts(normalize=True))
print("\nMost Common Winning Gamma:")
print(winners['focal_gamma'].value_counts(normalize=True))

# 4. Check for '53%' roughly in other configs
pivot = df_merged.pivot_table(
    values='delta_micro_f1', 
    index='focal_alpha', 
    columns='focal_gamma', 
    aggfunc=lambda x: (x > 0).mean()
)
print("\nWin Rates by (Alpha, Gamma):")
print(pivot)

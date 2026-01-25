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

# Focus on Best Config
target_alpha = 0.5
target_gamma = 3.0
print(f"--- Analysis for Alpha={target_alpha}, Gamma={target_gamma} ---")

subset = df_merged[
    (df_merged['focal_alpha'] == target_alpha) &
    (df_merged['focal_gamma'] == target_gamma)
].copy()

def analyze_factor(df, factor):
    stats = df.groupby(factor).agg(
        total_experiments=('delta_micro_f1', 'count'),
        wins=('delta_micro_f1', lambda x: (x > 0).sum()),
        win_rate=('delta_micro_f1', lambda x: (x > 0).mean()),
        avg_delta=('delta_micro_f1', 'mean')
    )
    print(f"\n--- Effect of {factor} ---")
    print(stats)

# Analyze various factors
analyze_factor(subset, 'window_size')
analyze_factor(subset, 'num_layers')
analyze_factor(subset, 'hidden_dim')
analyze_factor(subset, 'dropout')

# Combined analysis for complexity
subset['complexity'] = subset['num_layers'].astype(str) + "x" + subset['hidden_dim'].astype(str)
analyze_factor(subset, 'complexity')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plots style
sns.set_theme(style="whitegrid")
# plt.rcParams['figure.figsize'] = [12, 6]

def main():
    # Paths to CSV files
    # Assuming script is run from /home/quique/tesis/deep-river-tesis/results/lstm/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_bce = os.path.join(base_dir, '../../lstm_bce.csv')
    path_focal = os.path.join(base_dir, 'lstm_focal_full_20251209_035057.csv')

    print(f"Loading BCE from: {path_bce}")
    print(f"Loading Focal from: {path_focal}")

    # Load DataFrames
    df_bce = pd.read_csv(path_bce)
    df_focal = pd.read_csv(path_focal)

    # Normalize column names for BCE
    df_bce.columns = df_bce.columns.str.lower().str.replace(' ', '_')
    if 'model' in df_bce.columns:
        df_bce = df_bce.rename(columns={'model': 'name'})
    
    # Normalize column names for Focal (just in case)
    df_focal.columns = df_focal.columns.str.lower().str.replace(' ', '_')

    # Columns to merge on
    # Inspect columns first to be sure
    print("BCE Cols:", df_bce.columns.tolist())
    print("Focal Cols:", df_focal.columns.tolist())

    # Common config columns
    merge_cols = ['hidden_dim', 'num_layers', 'bidirectional', 'lr', 'dropout', 'window_size']
    
    # Verify these columns exist in both
    for col in merge_cols:
        if col not in df_bce.columns:
            print(f"Warning: {col} not in BCE")
        if col not in df_focal.columns:
            print(f"Warning: {col} not in Focal")

    # Merge
    # We want to match each Focal experiment to its corresponding BCE baseline
    df_merged = pd.merge(
        df_focal,
        df_bce[merge_cols + ['micro_f1', 'macro_f1']],
        on=merge_cols,
        how='left',
        suffixes=('_focal', '_bce')
    )

    # Check for unmatched rows
    if df_merged['micro_f1_bce'].isnull().any():
        print("Warning: Some Focal experiments did not match a BCE baseline!")
        print(df_merged[df_merged['micro_f1_bce'].isnull()][merge_cols])

    # Calculate Deltas
    df_merged['delta_micro_f1'] = df_merged['micro_f1_focal'] - df_merged['micro_f1_bce']
    df_merged['delta_macro_f1'] = df_merged['macro_f1_focal'] - df_merged['macro_f1_bce']

    print(f"\nTotal Focal experiments: {len(df_focal)}")
    print(f"Merged rows: {len(df_merged)}")

    # Statistics
    print("\n--- Statistics ---")
    print(f"Mean Delta Micro F1: {df_merged['delta_micro_f1'].mean():.4f}")
    print(f"Median Delta Micro F1: {df_merged['delta_micro_f1'].median():.4f}")
    print(f"Max Delta Micro F1: {df_merged['delta_micro_f1'].max():.4f}")
    
    print("\n--- Top 5 Improvements ---")
    print(df_merged.sort_values('delta_micro_f1', ascending=False)[
        ['focal_alpha', 'focal_gamma'] + merge_cols + ['micro_f1_focal', 'micro_f1_bce', 'delta_micro_f1']
    ].head(5))

    # Save heatmap plot
    try:
        pivot_table = df_merged.pivot_table(
            values='delta_micro_f1', 
            index='focal_alpha', 
            columns='focal_gamma', 
            aggfunc='mean'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap='RdBu_r', center=0, fmt=".3f")
        plt.title('Average Delta Micro F1 by Focal Alpha and Gamma')
        plt.savefig(os.path.join(base_dir, 'focal_vs_bce_heatmap.png'))
        print("\nSaved heatmap to results/lstm/focal_vs_bce_heatmap.png")
    except Exception as e:
        print(f"Could not create heatmap: {e}")

if __name__ == "__main__":
    main()

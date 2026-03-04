"""
OEMLHAT evaluation on AI4I dataset with metrics logged every 500 steps.
Mirrors the evaluation pattern of lstm_ai4i_adaptive_v2.py but using
the OEMLHAT ensemble model instead of LSTM.

Must be run from OEMLHAT4PdM/src/ directory (or with sys.path adjusted).
"""

import sys
import os
import time
import numbers
import pandas as pd
from datetime import datetime

# Import evaluate and river metrics from project root FIRST
import evaluate
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from river import preprocessing
from custommetrics.multioutput import HammingLoss

# THEN add OEMLHAT source to path for oemlhat-specific imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OEMLHAT4PdM', 'src'))

from oemlhat import OEMLHAT
from datasets.multioutput import Ai4i


def safe_get(m):
    """Safely get metric value, returning 0.0 if metric has no data yet."""
    try:
        return m.get()
    except Exception:
        return 0.0


def run_oemlhat_experiment():
    """Run OEMLHAT on AI4I with metrics every 500 steps."""

    start_time = time.time()

    # Initialize stream
    stream = Ai4i()

    # Initialize OEMLHAT with default parameters (as in tutorial)
    model = OEMLHAT(grouping_features=['Type_H', 'Type_L', 'Type_M'])

    # Preprocessing pipeline (same as tutorial)
    pp = SelectType(numbers.Number)
    pp += SelectType(str) | preprocessing.OneHotEncoder()
    pipeline = pp | model

    # Metrics
    metric = Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1()), HammingLoss()])

    task_id = "OEMLHAT_default"
    print(f"\n{'='*80}")
    print(f"OEMLHAT Evaluation on AI4I Dataset")
    print(f"{'='*80}\n")
    print(f"▶ STARTING: {task_id}")

    # Evaluate with iter_progressive_val_score every 500 steps
    checkpoints_data = []
    for checkpoint in evaluate.iter_progressive_val_score(
        dataset=stream,
        model=pipeline,
        metric=metric,
        step=500,
        measure_time=True,
        measure_memory=True
    ):
        step_n = checkpoint['Step']
        elapsed = checkpoint['Time'].total_seconds()
        exact_match_val = safe_get(metric[0]) * 100
        macro_f1_val = safe_get(metric[1]) * 100
        micro_f1_val = safe_get(metric[2]) * 100
        hamming_val = safe_get(metric[3])
        memory = checkpoint.get('Memory', 'N/A')

        print(f"  [{step_n:,d}] ExactMatch={exact_match_val:.2f}% | MacroF1={macro_f1_val:.2f}% | MicroF1={micro_f1_val:.2f}% | HammingLoss={hamming_val:.4f} | Time={elapsed:.1f}s | Mem={memory}")

        checkpoint_row = {
            "Step": step_n,
            "Elapsed_s": elapsed,
            "ExactMatch": exact_match_val,
            "MacroF1": macro_f1_val,
            "MicroF1": micro_f1_val,
            "HammingLoss": hamming_val,
        }
        checkpoints_data.append(checkpoint_row)

    duration = time.time() - start_time
    micro_f1 = safe_get(metric[2]) * 100

    # Save checkpoint history to CSV
    os.makedirs("results/oemlhat_ai4i", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f"results/oemlhat_ai4i/metrics_every500_{task_id}_{timestamp}.csv"
    pd.DataFrame(checkpoints_data).to_csv(checkpoint_filename, index=False)
    print(f"\n📄 Metrics history saved to: {checkpoint_filename}")

    print(f"✅ FINISHED: {task_id} | MicroF1={micro_f1:.2f}% | {duration:.1f}s")

    # Final summary
    result = {
        "Model": "OEMLHAT",
        "ExactMatch": safe_get(metric[0]) * 100,
        "MacroF1": safe_get(metric[1]) * 100,
        "MicroF1": micro_f1,
        "HammingLoss": safe_get(metric[3]),
        "Duration": duration,
    }

    # Save final result
    df = pd.DataFrame([result])
    result_filename = f"results/oemlhat_ai4i/result_{task_id}_{timestamp}.csv"
    df.to_csv(result_filename, index=False)

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\n\nSaved to: {result_filename}")

    return result


if __name__ == "__main__":
    run_oemlhat_experiment()

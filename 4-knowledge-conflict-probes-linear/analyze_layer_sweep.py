"""
Analyze layer sweep results for Qwen2.5-VL-7B-Instruct.
Collects eval_metrics.jsonl from each layer and plots layer-wise curves.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROBE_DIR = Path("./value_head_probes")
OUTPUT_DIR = Path("./layer_sweep_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

results = []

for layer in range(28):
    probe_id = f"Qwen2.5-VL-7B-Instruct-linear-{layer}"
    metrics_path = PROBE_DIR / probe_id / "evaluation_results" / "eval_metrics.jsonl"

    if not metrics_path.exists():
        print(f"Layer {layer}: no metrics found, skipping")
        continue

    with open(metrics_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    if not lines:
        continue

    # Take the last entry (end of training)
    last = lines[-1]
    last["layer"] = layer
    results.append(last)
    print(f"Layer {layer}: loaded {len(lines)} eval entries")

if not results:
    print("No results found. Make sure eval_metrics.jsonl files exist.")
    exit(1)

df = pd.DataFrame(results).sort_values("layer")
print("\n=== Available columns ===")
print(df.columns.tolist())
print("\n=== Results ===")
print(df.to_string(index=False))

df.to_csv(OUTPUT_DIR / "layer_sweep_results.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR}/layer_sweep_results.csv")

# Plot
metric_groups = {
    "Sample-level F1": [c for c in df.columns if "sample" in c.lower() and "f1" in c.lower()],
    "Span-level F1":   [c for c in df.columns if "span" in c.lower() and "f1" in c.lower()],
    "Token-level F1":  [c for c in df.columns if "token" in c.lower() and "f1" in c.lower()],
    "AUC-ROC":         [c for c in df.columns if "auc" in c.lower() or "roc" in c.lower()],
}

for group_name, cols in metric_groups.items():
    if not cols:
        continue
    plt.figure(figsize=(10, 5))
    for col in cols:
        if col in df.columns:
            plt.plot(df["layer"], df[col], marker="o", label=col)
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(f"Qwen2.5-VL-7B-Instruct — {group_name} by Layer")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    fname = group_name.lower().replace(" ", "_").replace("-", "_") + ".png"
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved plot: {OUTPUT_DIR}/{fname}")

print("\nDone.")

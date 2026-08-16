#!/usr/bin/env python3
"""Plot loss and error trends from metrics CSV."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Read metrics
df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "test_out" / "metrics.csv"))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Plot 1: Loss components ---
ax1.plot(df['iter'], df['loss_total'], label='Total', color='#1f77b4', linewidth=2)
ax1.plot(df['iter'], df['l1'], label='L1', color='#ff7f0e', linewidth=1.5, alpha=0.8)
ax1.plot(df['iter'], df['lg'], label='Lg (global)', color='#2ca02c', linewidth=1.5, alpha=0.8)
ax1.plot(df['iter'], df['ll'], label='Ll (local)', color='#d62728', linewidth=1.5, alpha=0.8)
ax1.plot(df['iter'], df['lm'], label='Lm (mask)', color='#9467bd', linewidth=1.5, alpha=0.8)

ax1.set_ylabel('Loss')
ax1.set_title('Loss Components Over Training')
ax1.legend(loc='upper right', ncol=3, fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# --- Plot 2: Error metrics ---
ax2.plot(df['iter'], df['absrel'], label='AbsRel', color='#1f77b4', linewidth=2)
ax2.plot(df['iter'], df['mae'] / 1000, label='MAE (m)', color='#ff7f0e', linewidth=1.5, alpha=0.8)
ax2.plot(df['iter'], df['rmse'] / 1000, label='RMSE (m)', color='#2ca02c', linewidth=1.5, alpha=0.8)

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Error')
ax2.set_title('Depth Error Metrics Over Training')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent.parent / "test_out" / "metrics_plot.png"), dpi=150, bbox_inches='tight')
print("Saved plot to test_out/metrics_plot.png")
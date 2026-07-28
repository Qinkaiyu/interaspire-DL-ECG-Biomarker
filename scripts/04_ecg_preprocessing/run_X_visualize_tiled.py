#!/usr/bin/env python3
"""
Step X: Visualize 100 tiled ECG files — before vs after, side by side.
Only picks files that were actually tiled (not already full).
"""
import numpy as np
import scipy.io as sio
import os, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR    = Path("data/ecg_matrices/preprocessed")
TILED_DIR = Path("data/ecg_matrices/preprocessed_tiled")
OUT_DIR   = Path("results/ecg_tiled_visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Find files that were tiled (not already full) ──────────────────────────────
log("Scanning for tiled files …")
files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.mat')])

tiled_files = []
for fname in files:
    orig = sio.loadmat(IN_DIR / fname)['feats']
    # Check if any lead has zeros (i.e. was tiled)
    has_partial = False
    for i in range(12):
        nz = np.count_nonzero(orig[i])
        if 0 < nz < 5000:
            has_partial = True
            break
    if has_partial:
        tiled_files.append(fname)

log(f"  Found {len(tiled_files)} tiled files, picking 100")

# Pick 100 spread across the list
if len(tiled_files) > 100:
    idx = np.linspace(0, len(tiled_files)-1, 100, dtype=int)
    selected = [tiled_files[i] for i in idx]
else:
    selected = tiled_files

# ── Plot each file ─────────────────────────────────────────────────────────────
log(f"Generating {len(selected)} visualizations …")

for plot_i, fname in enumerate(selected):
    orig  = sio.loadmat(IN_DIR / fname)['feats']       # (12, 5000)
    tiled = sio.loadmat(TILED_DIR / fname)['feats']     # (12, 5000)

    fig, axes = plt.subplots(12, 2, figsize=(16, 20), sharex='col')
    fig.suptitle(f"{fname}  —  Original (left) vs Tiled (right)", fontsize=13, y=0.995)

    for i in range(12):
        # Original
        ax_l = axes[i, 0]
        ax_l.plot(orig[i], linewidth=0.4, color='#1565C0')
        orig_nz = np.count_nonzero(orig[i])
        ax_l.set_ylabel(LEAD_NAMES[i], fontsize=9, rotation=0, labelpad=25, va='center')
        ax_l.tick_params(labelsize=6)
        if orig_nz == 0:
            ax_l.text(2500, 0, 'ALL ZERO', ha='center', va='center',
                     fontsize=8, color='red', fontweight='bold')
        else:
            ax_l.set_title(f"{orig_nz}/5000", fontsize=7, pad=1) if i == 0 else None

        # Tiled
        ax_r = axes[i, 1]
        tiled_nz = np.count_nonzero(tiled[i])
        if orig_nz == 0:
            ax_r.plot(tiled[i], linewidth=0.4, color='gray')
            ax_r.text(2500, 0, 'UNFILLED (zero)', ha='center', va='center',
                     fontsize=8, color='red', fontweight='bold')
        elif orig_nz < 5000:
            # Show tiled with color indicating repeats
            ax_r.plot(tiled[i], linewidth=0.4, color='#E65100')
            # Mark original segment boundaries
            is_nz = (orig[i] != 0)
            nz_idx = np.where(is_nz)[0]
            seg_start, seg_end = nz_idx[0], nz_idx[-1] + 1
            seg_len = seg_end - seg_start
            # Draw vertical lines at repeat boundaries
            for k in range(1, 5000 // seg_len + 1):
                bnd = k * seg_len
                if bnd < 5000:
                    ax_r.axvline(bnd, color='gray', linewidth=0.3, linestyle='--', alpha=0.5)
        else:
            ax_r.plot(tiled[i], linewidth=0.4, color='#2E7D32')

        ax_r.tick_params(labelsize=6)

        # Match y-limits
        all_vals = np.concatenate([orig[i], tiled[i]])
        nz_vals = all_vals[all_vals != 0]
        if len(nz_vals) > 0:
            ymin, ymax = nz_vals.min(), nz_vals.max()
            margin = (ymax - ymin) * 0.1 + 0.001
            ax_l.set_ylim(ymin - margin, ymax + margin)
            ax_r.set_ylim(ymin - margin, ymax + margin)

    axes[0, 0].set_title("Original", fontsize=10, fontweight='bold')
    axes[0, 1].set_title("Tiled", fontsize=10, fontweight='bold')
    axes[11, 0].set_xlabel("Sample (0–5000)", fontsize=8)
    axes[11, 1].set_xlabel("Sample (0–5000)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_name = f"{plot_i+1:03d}_{fname.replace('.mat','')}.png"
    plt.savefig(OUT_DIR / out_name, dpi=100, bbox_inches='tight')
    plt.close()

    if (plot_i + 1) % 20 == 0:
        log(f"  {plot_i+1}/{len(selected)} done")

log(f"\n✓ Saved {len(selected)} visualizations → {OUT_DIR}")

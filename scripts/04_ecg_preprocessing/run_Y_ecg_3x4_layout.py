#!/usr/bin/env python3
"""
Step Y: Standardize all ECG signals to 3x4 layout.

3x4 layout (standard 12-lead ECG paper):
  Columns 0-1249:    Leads I, II, III
  Columns 1250-2499: Leads aVR, aVL, aVF
  Columns 2500-3749: Leads V1, V2, V3
  Columns 3750-4999: Leads V4, V5, V6

Each lead gets exactly 1250 samples of signal at its designated slot,
zeros elsewhere. If the original signal is longer than 1250, truncate.

Input:  data/ecg_matrices/preprocessed/
Output: data/ecg_matrices/preprocessed_3x4/
Viz:    results/ecg_3x4_visualizations/
"""
import numpy as np
import scipy.io as sio
import os, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR  = Path("data/ecg_matrices/preprocessed")
OUT_DIR = Path("data/ecg_matrices/preprocessed_3x4")
VIZ_DIR = Path("results/ecg_3x4_visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN = 5000
SLOT_LEN = 1250
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# 3x4 layout: which slot (0-3) each lead belongs to
LEAD_SLOT = {
    0: 0, 1: 0, 2: 0,     # I, II, III       → [0, 1250)
    3: 1, 4: 1, 5: 1,     # aVR, aVL, aVF    → [1250, 2500)
    6: 2, 7: 2, 8: 2,     # V1, V2, V3       → [2500, 3750)
    9: 3, 10: 3, 11: 3,   # V4, V5, V6       → [3750, 5000)
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_signal_segment(lead_data):
    """Find the largest contiguous non-zero segment."""
    is_nonzero = (lead_data != 0)
    if not is_nonzero.any():
        return None, None
    if is_nonzero.all():
        return 0, len(lead_data)
    diff = np.diff(is_nonzero.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if is_nonzero[0]:
        starts = np.concatenate([[0], starts])
    if is_nonzero[-1]:
        ends = np.concatenate([ends, [len(lead_data)]])
    lengths = ends - starts
    best = np.argmax(lengths)
    return int(starts[best]), int(ends[best])


def convert_to_3x4(ecg):
    """Convert any layout to standardized 3x4 layout."""
    out = np.zeros((12, TARGET_LEN), dtype=ecg.dtype)
    lead_info = []

    for i in range(12):
        slot = LEAD_SLOT[i]
        slot_start = slot * SLOT_LEN
        slot_end = slot_start + SLOT_LEN

        s, e = find_signal_segment(ecg[i])
        if s is None:
            lead_info.append(('zero', 0))
            continue

        seg = ecg[i, s:e]
        seg_len = len(seg)

        # Take first 1250 samples (or pad if shorter)
        if seg_len >= SLOT_LEN:
            out[i, slot_start:slot_end] = seg[:SLOT_LEN]
            lead_info.append(('truncated' if seg_len > SLOT_LEN else 'exact', seg_len))
        else:
            out[i, slot_start:slot_start + seg_len] = seg
            lead_info.append(('short', seg_len))

    return out, lead_info


# ── Process all files ──────────────────────────────────────────────────────────
files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.mat')])
log(f"Processing {len(files)} .mat files → 3x4 layout …")

stats = {"already_3x4": 0, "converted": 0, "has_zero": 0, "all_zero": 0}

for idx, fname in enumerate(files):
    d = sio.loadmat(IN_DIR / fname)
    ecg = d['feats']

    ecg_3x4, lead_info = convert_to_3x4(ecg)

    # Save
    out_d = {k: v for k, v in d.items() if not k.startswith('__')}
    out_d['feats'] = ecg_3x4
    sio.savemat(OUT_DIR / fname, out_d, do_compression=True)

    n_zero = sum(1 for status, _ in lead_info if status == 'zero')
    if n_zero == 12:
        stats["all_zero"] += 1
    elif n_zero > 0:
        stats["has_zero"] += 1
        stats["converted"] += 1
    else:
        stats["converted"] += 1

    if (idx + 1) % 500 == 0:
        log(f"  {idx+1}/{len(files)} done")

log(f"\nDone. Processed {len(files)} files → {OUT_DIR}")
log(f"  Converted:             {stats['converted']}")
log(f"  Has zero leads:        {stats['has_zero']}")
log(f"  All-zero (empty):      {stats['all_zero']}")

# ── Visualize 100 files ───────────────────────────────────────────────────────
log("\nGenerating 100 visualizations …")

# Pick 100 spread across all files
viz_idx = np.linspace(0, len(files)-1, 100, dtype=int)
viz_files = [files[i] for i in viz_idx]

SLOT_COLORS = ['#1565C0', '#2E7D32', '#E65100', '#AD1457']  # blue, green, orange, pink

for plot_i, fname in enumerate(viz_files):
    orig = sio.loadmat(IN_DIR / fname)['feats']
    conv = sio.loadmat(OUT_DIR / fname)['feats']

    fig, axes = plt.subplots(12, 2, figsize=(16, 20), sharex='col')
    fig.suptitle(f"{fname}  —  Original (left) vs 3×4 Layout (right)", fontsize=13, y=0.995)

    for i in range(12):
        slot = LEAD_SLOT[i]
        slot_start = slot * SLOT_LEN
        slot_end = slot_start + SLOT_LEN
        color = SLOT_COLORS[slot]

        # Original
        ax_l = axes[i, 0]
        ax_l.plot(orig[i], linewidth=0.4, color='#1565C0')
        ax_l.set_ylabel(LEAD_NAMES[i], fontsize=9, rotation=0, labelpad=25, va='center')
        ax_l.tick_params(labelsize=6)
        orig_nz = np.count_nonzero(orig[i])
        if orig_nz == 0:
            ax_l.text(2500, 0, 'ALL ZERO', ha='center', va='center',
                     fontsize=8, color='red', fontweight='bold')

        # 3x4
        ax_r = axes[i, 1]
        conv_nz = np.count_nonzero(conv[i])
        if conv_nz == 0:
            ax_r.text(2500, 0, 'ZERO', ha='center', va='center',
                     fontsize=8, color='red', fontweight='bold')
        else:
            ax_r.plot(conv[i], linewidth=0.4, color=color)
            # Highlight the active slot
            ax_r.axvspan(slot_start, slot_end, alpha=0.08, color=color)
            # Mark slot boundaries
            for bnd in [0, 1250, 2500, 3750, 5000]:
                ax_r.axvline(bnd, color='gray', linewidth=0.3, linestyle=':', alpha=0.4)

        ax_r.tick_params(labelsize=6)

        # Match y-limits
        all_vals = np.concatenate([orig[i], conv[i]])
        nz_vals = all_vals[all_vals != 0]
        if len(nz_vals) > 0:
            ymin, ymax = nz_vals.min(), nz_vals.max()
            margin = (ymax - ymin) * 0.1 + 0.001
            ax_l.set_ylim(ymin - margin, ymax + margin)
            ax_r.set_ylim(ymin - margin, ymax + margin)

    axes[0, 0].set_title("Original", fontsize=10, fontweight='bold')
    axes[0, 1].set_title("3×4 Layout (I,II,III | aVR,aVL,aVF | V1-V3 | V4-V6)",
                         fontsize=9, fontweight='bold')
    axes[11, 0].set_xlabel("Sample (0–5000)", fontsize=8)
    axes[11, 1].set_xlabel("Sample (0–5000)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_name = f"{plot_i+1:03d}_{fname.replace('.mat','')}.png"
    plt.savefig(VIZ_DIR / out_name, dpi=100, bbox_inches='tight')
    plt.close()

    if (plot_i + 1) % 20 == 0:
        log(f"  {plot_i+1}/100 done")

log(f"\n✓ Saved 100 visualizations → {VIZ_DIR}")
log("✓ Step Y done.")

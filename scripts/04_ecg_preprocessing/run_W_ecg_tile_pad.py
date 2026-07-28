#!/usr/bin/env python3
"""
Step W: Tile-pad ECG signals to fill (12, 5000).

For each lead in each .mat file:
  1. Find the contiguous non-zero signal segment
  2. Extract it
  3. Tile (repeat) to fill the full 5000 samples
  4. If the lead is all-zero, leave it as-is (cannot recover)

Input:  data/ecg_matrices/preprocessed/
Output: data/ecg_matrices/preprocessed_tiled/
"""
import numpy as np
import scipy.io as sio
import os, time
from pathlib import Path

IN_DIR  = Path("data/ecg_matrices/preprocessed")
OUT_DIR = Path("data/ecg_matrices/preprocessed_tiled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN = 5000

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def find_signal_segment(lead_data):
    """Find the largest contiguous non-zero segment in a 1D array."""
    is_nonzero = (lead_data != 0)
    if not is_nonzero.any():
        return None, None  # all zero
    if is_nonzero.all():
        return 0, len(lead_data)  # full signal

    # Find all contiguous non-zero runs
    diff = np.diff(is_nonzero.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if is_nonzero[0]:
        starts = np.concatenate([[0], starts])
    if is_nonzero[-1]:
        ends = np.concatenate([ends, [len(lead_data)]])

    # Pick the longest run
    lengths = ends - starts
    best = np.argmax(lengths)
    return int(starts[best]), int(ends[best])


def tile_pad_ecg(ecg):
    """Tile-pad each lead to fill TARGET_LEN. Returns (12, 5000) array."""
    out = np.zeros_like(ecg)
    n_tiled = 0
    n_full = 0
    n_zero = 0

    for i in range(ecg.shape[0]):
        s, e = find_signal_segment(ecg[i])
        if s is None:
            # All-zero lead — cannot recover
            n_zero += 1
            continue

        seg = ecg[i, s:e]
        seg_len = len(seg)

        if seg_len == TARGET_LEN:
            out[i] = seg
            n_full += 1
        else:
            # Tile to fill TARGET_LEN
            n_repeats = (TARGET_LEN // seg_len) + 1
            tiled = np.tile(seg, n_repeats)[:TARGET_LEN]
            out[i] = tiled
            n_tiled += 1

    return out, n_full, n_tiled, n_zero


# ── Process all files ──────────────────────────────────────────────────────────
files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.mat')])
log(f"Processing {len(files)} .mat files …")

stats = {"full": 0, "tiled": 0, "has_zero_leads": 0, "all_zero": 0}

for idx, fname in enumerate(files):
    d = sio.loadmat(IN_DIR / fname)
    ecg = d['feats']  # (12, 5000)

    ecg_out, n_full, n_tiled, n_zero = tile_pad_ecg(ecg)

    # Save with same structure
    out_d = {}
    for k, v in d.items():
        if k.startswith('__'):
            continue
        out_d[k] = v
    out_d['feats'] = ecg_out

    sio.savemat(OUT_DIR / fname, out_d, do_compression=True)

    if n_full == 12:
        stats["full"] += 1
    elif n_zero == 12:
        stats["all_zero"] += 1
    else:
        if n_tiled > 0:
            stats["tiled"] += 1
        if n_zero > 0:
            stats["has_zero_leads"] += 1

    if (idx + 1) % 500 == 0:
        log(f"  {idx+1}/{len(files)} done")

log(f"\nDone. Processed {len(files)} files → {OUT_DIR}")
log(f"  Already full (12/12 leads):     {stats['full']}")
log(f"  Tiled (at least 1 lead padded): {stats['tiled']}")
log(f"  Has zero leads (unrecoverable): {stats['has_zero_leads']}")
log(f"  All-zero (empty file):          {stats['all_zero']}")

# ── Verify a few tiled files ──────────────────────────────────────────────────
log("\nVerification (before → after):")
verify_files = ['PL40_001.mat', 'AE20_018.mat', 'EG60_018.mat']
for fname in verify_files:
    if not (IN_DIR / fname).exists():
        continue
    orig = sio.loadmat(IN_DIR / fname)['feats']
    tiled = sio.loadmat(OUT_DIR / fname)['feats']

    log(f"\n  {fname}:")
    for i in range(12):
        orig_nz = np.count_nonzero(orig[i])
        tiled_nz = np.count_nonzero(tiled[i])
        log(f"    Lead {i:2d}: {orig_nz:5d}/5000 → {tiled_nz:5d}/5000")

log("\n✓ Step W done.")

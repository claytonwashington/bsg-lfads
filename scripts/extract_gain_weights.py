#!/usr/bin/env python3
"""Extract gains and I_e weights from the SOC model checkpoint.

Run with: conda run -n lfads-torch-cuda12 python scripts/extract_gain_weights.py

Saves controller_to_g and controller_to_Ie weight/bias as .npz so that
gains can be reconstructed from gen_inputs in the stkit-nwb environment.
"""

import sys
from pathlib import Path

BSG_ROOT = str(Path(__file__).resolve().parent.parent)
if BSG_ROOT not in sys.path:
    sys.path.insert(0, BSG_ROOT)

import numpy as np
import torch

CKPT_PATH = Path("/home/cbwash2/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp/lightning_checkpoints/536-537.ckpt")
OUT_NPZ = Path("/home/cbwash2/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp/gain_projection_weights.npz")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
sd = ckpt["state_dict"]

# Extract the two linear projection layers
g_weight = sd["soc_decoder.controller_to_g.weight"].numpy()   # (200, 16)
g_bias = sd["soc_decoder.controller_to_g.bias"].numpy()       # (200,)
ie_weight = sd["soc_decoder.controller_to_Ie.weight"].numpy() # (200, 16)
ie_bias = sd["soc_decoder.controller_to_Ie.bias"].numpy()     # (200,)

np.savez(OUT_NPZ, g_weight=g_weight, g_bias=g_bias, ie_weight=ie_weight, ie_bias=ie_bias)
print(f"Saved gain projection weights to {OUT_NPZ}")
print(f"  g_weight: {g_weight.shape}, g_bias: {g_bias.shape}")
print(f"  ie_weight: {ie_weight.shape}, ie_bias: {ie_bias.shape}")

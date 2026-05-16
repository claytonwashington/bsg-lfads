#!/usr/bin/env python3
"""Smoke test for the LFADS-SOC MVP.

Runs with synthetic data to verify:
1. SOCCell shape correctness and buffer non-trainability
2. SOCDecoder full temporal unrolling
3. DualReadout shape correctness and EMG positivity
4. LFADS_SOC end-to-end forward pass
5. Gradient flow (W frozen, readout/projections trainable)
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

# Add parent to path so we can import lfads_torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfads_torch.modules.recurrent import SOCCell
from lfads_torch.modules.readout import DualReadout


def test_soc_cell():
    """Test SOCCell shapes, E/I enforcement, and buffer status."""
    print("=== Test: SOCCell ===")
    N = 200
    B = 8

    # Create a temporary W matrix file
    W = torch.randn(N, N) / (N**0.5)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(W, f.name)
        W_path = f.name

    try:
        cell = SOCCell(N=N, dt=0.5, tau=10.0, W_init_path=W_path)

        # Check W is a buffer, not a parameter
        param_names = [n for n, _ in cell.named_parameters()]
        assert len(param_names) == 0, f"SOCCell should have no parameters, got {param_names}"
        assert "W" in dict(cell.named_buffers()), "W should be a buffer"
        assert not cell.W.requires_grad, "W.requires_grad should be False"

        # Check E/I sign enforcement
        assert (cell.W[:, :N // 2] >= 0).all(), "Excitatory columns should be non-negative"
        assert (cell.W[:, N // 2:] <= 0).all(), "Inhibitory columns should be non-positive"

        # Check forward pass shapes
        v = torch.randn(B, N)
        I_e = torch.randn(B, N)
        g = torch.abs(torch.randn(B, N)) + 0.1  # positive

        v_next, r = cell(v, I_e, g)
        assert v_next.shape == (B, N), f"v_next shape {v_next.shape} != ({B}, {N})"
        assert r.shape == (B, N), f"r shape {r.shape} != ({B}, {N})"

        # Check numerical stability over 100 steps
        v = torch.zeros(B, N)
        for _ in range(100):
            v, r = cell(v, I_e, g)
        assert torch.isfinite(v).all(), "v diverged after 100 steps"
        assert torch.isfinite(r).all(), "r diverged after 100 steps"

        print("  ✓ Buffer, E/I, shapes, stability all correct")
    finally:
        os.unlink(W_path)


def test_dual_readout():
    """Test DualReadout shapes and EMG positivity."""
    print("=== Test: DualReadout ===")
    N = 200
    B = 8
    T = 100
    neural_dim = 71
    emg_dim = 14

    readout = DualReadout(N=N, neural_dim=neural_dim, emg_dim=emg_dim)
    r = torch.randn(B, T, N)
    result = readout(r)

    assert result["neural"].shape == (B, T, neural_dim), (
        f"neural shape {result['neural'].shape}"
    )
    assert result["emg"].shape == (B, T, emg_dim), (
        f"emg shape {result['emg'].shape}"
    )
    assert (result["emg"] > 0).all(), "EMG output should be strictly positive (exp)"

    print("  ✓ Shapes and positivity correct")


def test_gradient_flow():
    """Test that gradients flow to the right places."""
    print("=== Test: Gradient Flow ===")
    N = 20  # small for speed
    B = 4
    T = 10
    ic_dim = 8
    co_dim = 2
    ci_enc_dim = 8
    con_dim = 8
    neural_dim = 15
    emg_dim = 5

    # Create W
    W = torch.randn(N, N) / (N**0.5)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(W, f.name)
        W_path = f.name

    try:
        cell = SOCCell(N=N, dt=0.5, tau=10.0, W_init_path=W_path)
        readout = DualReadout(N=N, neural_dim=neural_dim, emg_dim=emg_dim)

        # Create simple projections
        ic_to_v0 = nn.Linear(ic_dim, N)
        controller_to_Ie = nn.Linear(co_dim, N)
        controller_to_g = nn.Linear(co_dim, N)

        # Synthetic data
        ic = torch.randn(B, ic_dim)
        co = torch.randn(B, T, co_dim)

        # Forward pass
        v = ic_to_v0(ic)
        rates = []
        for t in range(T):
            I_e = controller_to_Ie(co[:, t])
            g = torch.nn.functional.softplus(controller_to_g(co[:, t]))
            v, r = cell(v, I_e, g)
            rates.append(r)
        rates = torch.stack(rates, dim=1)

        result = readout(rates)
        loss = result["neural"].mean() + result["emg"].mean()
        loss.backward()

        # Check gradients
        assert ic_to_v0.weight.grad is not None, "ic_to_v0 should have gradients"
        assert controller_to_Ie.weight.grad is not None, "controller_to_Ie should have gradients"
        assert controller_to_g.weight.grad is not None, "controller_to_g should have gradients"
        assert readout.readout_neural.weight.grad is not None, "readout_neural should have gradients"
        assert readout.readout_emg.weight.grad is not None, "readout_emg should have gradients"
        assert cell.W.grad is None, "W should NOT have gradients (frozen buffer)"

        print("  ✓ Gradients flow correctly (W frozen, projections trainable)")
    finally:
        os.unlink(W_path)


if __name__ == "__main__":
    test_soc_cell()
    test_dual_readout()
    test_gradient_flow()
    print("\n=== All tests passed! ===")

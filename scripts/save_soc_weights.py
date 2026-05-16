#!/usr/bin/env python3
"""Generate and save a Stability-Optimized Circuit (SOC) weight matrix.

Usage:
    python scripts/save_soc_weights.py \\
        --N 200 --p 0.1 --R 10 --gamma 3 --frac_inh 0.5 \\
        --lr 10 --desired_SA 0.15 \\
        --output weights/W_soc_200.pt

Creates a .pt file containing a torch.Tensor of shape (N, N).
"""

import argparse

import numpy as np
import scipy.linalg as la
import torch


def initial_net(N, p, R, gamma, frac_inh):
    """Generate the initial weight matrix with excitatory and inhibitory connections."""
    start_inh = round(N * (1 - frac_inh))
    NN = round(p * N * (N - 1))
    fill = np.hstack([np.ones(NN), np.zeros(N * (N - 1) - NN)])
    np.random.shuffle(fill)
    fill = fill.reshape(N, N - 1)

    W1 = np.zeros((N, N))
    W1[:-1, 1:] = fill[:-1, :]

    W2 = np.zeros((N, N))
    W2[1:, :-1] = fill[1:, :]

    W = np.triu(W1, 1) + np.tril(W2, -1)
    w0 = np.sqrt(2) * R / np.sqrt(p * (1 - p) * (1 + gamma**2))
    W = W * (w0 / np.sqrt(N))
    W[:, start_inh:] *= -gamma
    return W


def stability_opt_step(Wi, rate, gamma, frac_inh):
    """Perform one iteration of gradient descent for stability optimization."""
    N = Wi.shape[0]
    end_exc = start_inh = round(N * (1 - frac_inh))

    Emax = max(np.real(la.eigvals(Wi)))

    s = max(Emax * 1.5, Emax + 0.2)
    A = Wi - s * np.eye(N)
    X = 2 * np.eye(N)

    Q = la.solve_continuous_lyapunov(A.T, X)
    P = la.solve_continuous_lyapunov(A, X)

    grad = Q @ P / np.trace(Q @ P)

    Wo = Wi.copy()
    Wo[:, start_inh:] -= rate * grad[:, start_inh:]
    Wo[Wo > 0] = 0
    Wo[:, :end_exc] = Wi[:, :end_exc]

    # E/I balancing
    meanEE = np.mean(Wo[:end_exc, :end_exc])
    meanEI = np.mean(Wo[start_inh:, :end_exc])
    meanIE = np.mean(Wo[:end_exc, start_inh:])
    meanII = np.mean(Wo[start_inh:, start_inh:])

    Wo[:end_exc, start_inh:] = -gamma * (meanEE / meanIE) * Wo[:end_exc, start_inh:]
    Wo[start_inh:, start_inh:] = -gamma * (meanEI / meanII) * Wo[start_inh:, start_inh:]

    # Inhibitory pruning (40% non-zero)
    inh_weights = Wo[:, start_inh:].flatten()
    sorted_indices = np.argsort(inh_weights)
    thres = round(0.4 * len(inh_weights))
    inh_weights[sorted_indices[thres:]] = 0
    Wo[:, start_inh:] = inh_weights.reshape(N, N - end_exc)

    np.fill_diagonal(Wo, 0)
    return Wo


def generate_soc_matrix(N, p, R, gamma, frac_inh, lr, desired_SA, seed=None):
    """Generate a stability-optimized weight matrix.

    Returns
    -------
    W : np.ndarray, shape (N, N)
        The SOC weight matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    W = initial_net(N, p, R, gamma, frac_inh)

    i = 0
    sa = max(np.real(la.eigvals(W)))
    print(f"Initial spectral abscissa: {sa:.4f}")

    while sa > desired_SA:
        i += 1
        W = stability_opt_step(W, lr, gamma, frac_inh)
        sa = max(np.real(la.eigvals(W)))
        if i % 20 == 0:
            print(f"  Iteration {i}, spectral abscissa: {sa:.4f}")

    print(f"Converged at iteration {i}, spectral abscissa: {sa:.4f}")
    return W


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save a SOC weight matrix."
    )
    parser.add_argument("--N", type=int, default=200, help="Number of neurons")
    parser.add_argument("--p", type=float, default=0.1, help="Connection probability")
    parser.add_argument("--R", type=float, default=10.0, help="Spectral radius scale")
    parser.add_argument("--gamma", type=float, default=3.0, help="E/I ratio")
    parser.add_argument("--frac_inh", type=float, default=0.5, help="Fraction inhibitory")
    parser.add_argument("--lr", type=float, default=10.0, help="SOC optimization learning rate")
    parser.add_argument("--desired_SA", type=float, default=0.15, help="Target spectral abscissa")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="weights/W_soc_200.pt",
        help="Output .pt file path",
    )
    args = parser.parse_args()

    W = generate_soc_matrix(
        N=args.N,
        p=args.p,
        R=args.R,
        gamma=args.gamma,
        frac_inh=args.frac_inh,
        lr=args.lr,
        desired_SA=args.desired_SA,
        seed=args.seed,
    )

    # Save as torch tensor
    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    W_tensor = torch.tensor(W, dtype=torch.float32)
    torch.save(W_tensor, args.output)
    print(f"Saved W matrix ({W_tensor.shape}) to {args.output}")


if __name__ == "__main__":
    main()

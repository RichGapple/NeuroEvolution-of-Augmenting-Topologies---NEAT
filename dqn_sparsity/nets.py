"""A small MLP with hand-written backprop and Adam, in numpy only.

Why not PyTorch
---------------
The NEAT arm depends on numpy/scipy/pandas/matplotlib and nothing else.  Adding
a deep-learning framework for a 8-64-64-3 network would be the largest
dependency in the project by two orders of magnitude, and would make the
"DQN needs heavier infrastructure" claim in the paper harder to state cleanly
(you would be measuring PyTorch's startup cost, not the algorithm's).

The risk of a hand-written learner is that a reviewer suspects the baseline is
simply broken.  Two things guard against that:
  * `tests/test_dqn.py` runs a finite-difference gradient check on every layer;
  * `scripts/11_dqn_calibrate.py` gates the sweep on DQN actually solving the
    dense condition, exactly as the NEAT arm gates on its own calibration.

If you would rather use PyTorch, replace only this file: `agent.py` touches the
network solely through `predict`, `train_step`, `get_weights` and `set_weights`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _he(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in),
                      size=(fan_in, fan_out)).astype(np.float32)


class MLP:
    """Fully connected net.  Layers: in -> hidden... -> out (linear output)."""

    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int,
                 activation: str = "relu", seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        dims = [in_dim] + list(hidden) + [out_dim]
        self.W = [_he(rng, dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1], dtype=np.float32)
                  for i in range(len(dims) - 1)]
        self.activation = activation
        self.n_layers = len(self.W)

    # -- forward -------------------------------------------------------------- #
    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        return np.tanh(z)

    def _dact(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0.0).astype(z.dtype)
        return 1.0 - a * a

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, list]:
        """Returns (output, cache).  x is (batch, in_dim)."""
        cache = []
        a = x
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            if i < self.n_layers - 1:
                a_next = self._act(z)
            else:
                a_next = z                      # linear head: Q-values
            cache.append((a, z, a_next))
            a = a_next
        return a, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.atleast_2d(x))[0]

    # -- backward ------------------------------------------------------------- #
    def backward(self, cache: list, dout: np.ndarray
                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """dout is dLoss/dOutput, shape (batch, out_dim)."""
        gW = [None] * self.n_layers
        gb = [None] * self.n_layers
        delta = dout
        for i in range(self.n_layers - 1, -1, -1):
            a_prev, z, a_next = cache[i]
            gW[i] = a_prev.T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                da = delta @ self.W[i].T
                _, z_prev, a_prev_out = cache[i - 1]
                delta = da * self._dact(z_prev, a_prev_out)
        return gW, gb

    # -- weights -------------------------------------------------------------- #
    def get_weights(self) -> list:
        return [w.copy() for w in self.W] + [b.copy() for b in self.b]

    def set_weights(self, ws: list) -> None:
        n = self.n_layers
        self.W = [w.copy() for w in ws[:n]]
        self.b = [b.copy() for b in ws[n:]]

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


class Adam:
    """Adam with bias correction and global-norm gradient clipping."""

    def __init__(self, net: MLP, lr: float, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 clip: float = 0.0) -> None:
        self.net = net
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, beta1, beta2, eps, clip
        self.mW = [np.zeros_like(w) for w in net.W]
        self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]
        self.vb = [np.zeros_like(b) for b in net.b]
        self.t = 0

    def step(self, gW: list, gb: list) -> float:
        self.t += 1
        if self.clip > 0:
            sq = sum(float((g ** 2).sum()) for g in gW)
            sq += sum(float((g ** 2).sum()) for g in gb)
            norm = np.sqrt(sq)
            if norm > self.clip:
                scale = self.clip / (norm + 1e-12)
                gW = [g * scale for g in gW]
                gb = [g * scale for g in gb]
        else:
            norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in gW)))

        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i in range(len(gW)):
            self.mW[i] = self.b1 * self.mW[i] + (1 - self.b1) * gW[i]
            self.vW[i] = self.b2 * self.vW[i] + (1 - self.b2) * gW[i] ** 2
            self.net.W[i] -= (self.lr * (self.mW[i] / bc1)
                              / (np.sqrt(self.vW[i] / bc2) + self.eps)).astype(np.float32)

            self.mb[i] = self.b1 * self.mb[i] + (1 - self.b1) * gb[i]
            self.vb[i] = self.b2 * self.vb[i] + (1 - self.b2) * gb[i] ** 2
            self.net.b[i] -= (self.lr * (self.mb[i] / bc1)
                              / (np.sqrt(self.vb[i] / bc2) + self.eps)).astype(np.float32)
        return float(norm)


# --------------------------------------------------------------------------- #
def huber_grad(pred: np.ndarray, target: np.ndarray, delta: float
               ) -> Tuple[np.ndarray, float]:
    """Returns (dLoss/dPred, mean loss).  delta <= 0 gives plain MSE."""
    err = pred - target
    if delta <= 0:
        return err.astype(pred.dtype), float(0.5 * np.mean(err ** 2))
    absr = np.abs(err)
    quad = absr <= delta
    grad = np.where(quad, err, delta * np.sign(err))
    loss = np.where(quad, 0.5 * err ** 2, delta * (absr - 0.5 * delta))
    return grad.astype(np.float32), float(np.mean(loss))

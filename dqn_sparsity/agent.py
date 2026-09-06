"""DQN: replay buffer, target network, epsilon-greedy, optional Double DQN.

Everything here is textbook (Mnih et al. 2015; van Hasselt et al. 2016).  The
only study-specific parts are the diagnostics at the bottom, which measure the
quantities the reward-sparsity hypothesis is actually about:

  * `zero_reward_fraction` -- what share of transitions carry no signal at all.
    This is the DQN analogue of NEAT's "distinct fitness values per generation":
    it is the manipulation check for this arm.
  * `distinct_rewards` -- how many different reward values the buffer holds.
  * `q_spread` -- max(Q) - min(Q) over the greedy policy's states, i.e. how
    strongly the learned value function discriminates between actions.  When
    this collapses, the agent has stopped being able to tell actions apart, the
    same failure mode NEAT hits when selection differential goes to zero.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import DQNConfig
from .nets import MLP, Adam, huber_grad


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done)."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int) -> Tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    # -- sparsity diagnostics -------------------------------------------------- #
    def zero_reward_fraction(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(np.mean(self.r[:self.n] == 0.0))

    def distinct_rewards(self) -> int:
        if self.n == 0:
            return 0
        return int(len(np.unique(np.round(self.r[:self.n], 9))))

    def reward_entropy(self) -> float:
        """Normalised Shannon entropy of the reward histogram, in [0, 1].

        The DQN counterpart of the NEAT arm's selection entropy.  1.0 means the
        buffer's rewards are maximally varied; 0.0 means they are all identical
        and no gradient can distinguish any transition from any other.
        """
        if self.n == 0:
            return float("nan")
        vals, counts = np.unique(np.round(self.r[:self.n], 9), return_counts=True)
        if len(vals) <= 1:
            return 0.0
        p = counts / counts.sum()
        h = float(-(p * np.log(p)).sum())
        return h / float(np.log(len(vals))) if len(vals) > 1 else 0.0


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig,
                 seed: int = 0) -> None:
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self.q = MLP(obs_dim, cfg.hidden, n_actions, cfg.activation, seed=seed)
        self.target = MLP(obs_dim, cfg.hidden, n_actions, cfg.activation,
                          seed=seed)
        self.target.set_weights(self.q.get_weights())
        self.opt = Adam(self.q, cfg.lr, cfg.adam_beta1, cfg.adam_beta2,
                        cfg.adam_eps, cfg.grad_clip)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, seed=seed + 1)

        self.step_count = 0
        self.updates = 0
        self._loss_acc = 0.0
        self._loss_n = 0
        self._gradnorm_acc = 0.0

    # -- acting --------------------------------------------------------------- #
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy:
            eps = self.cfg.eps_at(self.step_count)
            if self.step_count < self.cfg.warmup_steps or self.rng.random() < eps:
                return int(self.rng.integers(0, self.n_actions))
        qs = self.q.predict(obs)[0]
        return int(np.argmax(qs))

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.q.predict(obs)[0]

    # -- learning ------------------------------------------------------------- #
    def observe(self, s, a, r, s2, done) -> None:
        self.buffer.add(s, a, r, s2, done)
        self.step_count += 1
        if (self.step_count >= self.cfg.warmup_steps
                and self.buffer.n >= self.cfg.batch_size
                and self.step_count % self.cfg.train_every == 0):
            self._train_step()
        if self.step_count % self.cfg.target_update_every == 0:
            self.target.set_weights(self.q.get_weights())

    def _train_step(self) -> None:
        cfg = self.cfg
        s, a, r, s2, d = self.buffer.sample(cfg.batch_size)

        q_next_target = self.target.predict(s2)
        if cfg.double_dqn:
            a_star = np.argmax(self.q.predict(s2), axis=1)
            boot = q_next_target[np.arange(len(a_star)), a_star]
        else:
            boot = q_next_target.max(axis=1)
        y = r + cfg.gamma * (1.0 - d) * boot

        pred_all, cache = self.q.forward(s)
        idx = np.arange(len(a))
        pred = pred_all[idx, a]

        g, loss = huber_grad(pred, y.astype(np.float32), cfg.huber_delta)
        dout = np.zeros_like(pred_all)
        dout[idx, a] = g / len(a)

        gW, gb = self.q.backward(cache, dout)
        norm = self.opt.step(gW, gb)

        self._loss_acc += loss
        self._gradnorm_acc += norm
        self._loss_n += 1
        self.updates += 1

    # -- diagnostics ---------------------------------------------------------- #
    def drain_train_stats(self) -> dict:
        n = max(1, self._loss_n)
        out = {"td_loss": self._loss_acc / n,
               "grad_norm": self._gradnorm_acc / n,
               "n_updates_block": self._loss_n}
        self._loss_acc = 0.0
        self._gradnorm_acc = 0.0
        self._loss_n = 0
        return out

    def q_spread_on(self, states: np.ndarray) -> float:
        """Mean (max Q - min Q) across a batch of states."""
        if len(states) == 0:
            return float("nan")
        qs = self.q.predict(np.asarray(states, dtype=np.float32))
        return float(np.mean(qs.max(axis=1) - qs.min(axis=1)))

    def get_weights(self):
        return self.q.get_weights()

    def set_weights(self, ws) -> None:
        self.q.set_weights(ws)
        self.target.set_weights(ws)

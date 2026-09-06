"""The two mazes. Shared by both arms so they cannot drift apart.

MAZE_A is your original layout. Grid analysis shows it is **deceptive**: a policy
that always reduces distance-to-goal walks into the cul-de-sac at (58, 90) and
stops there at progress 0.717. Since the intermediate credit term is a monotone
function of distance-to-goal, that cul-de-sac is the ceiling for any learner
following the reward gradient alone. Measured DQN plateau: 0.697 median across
8 seeds, which is the analytic optimum to three decimals.

MAZE_B is MAZE_A with the second bar shortened from y<100 to y<78. That is the
only difference. Greedy descent now reaches the goal, so the maze is
non-deceptive, and the shortest path drops from 216 to 160 units (80 of 150
steps, comfortable slack).

Why both
--------
On MAZE_A, reward *resolution* (eta) and reward *deceptiveness* are confounded:
DQN fails at every eta, and you cannot tell which property it is losing to.
Running the same eta sweep on both mazes separates them. One property differs;
everything else -- start, goal, sensors, dynamics, step budget, reward model --
is identical.

Changing the obstacles changes `EnvConfig`, which changes the config hash, so
runs on the two mazes can never be accidentally pooled. That is the guard rail
working, not a nuisance.
"""

from __future__ import annotations

import copy
from typing import List, Tuple

Obstacles = List[Tuple[float, float, float, float]]

# (x0, y0, x1, y1)
MAZE_A: Obstacles = [
    (30.0,  0.0, 40.0,  62.0),     # forces a northward detour
    (60.0, 38.0, 70.0, 100.0),     # forces a southward detour -> contradiction
    ( 0.0, 74.0, 22.0,  84.0),
    (78.0, 12.0, 100.0, 22.0),
]

MAZE_B: Obstacles = [
    (30.0,  0.0, 40.0,  62.0),
    (60.0, 38.0, 70.0,  78.0),     # <- shortened; the only change
    ( 0.0, 74.0, 22.0,  84.0),
    (78.0, 12.0, 100.0, 22.0),
]

MAZES = {"A": MAZE_A, "B": MAZE_B}

DESCRIPTIONS = {
    "A": "deceptive (greedy descent traps at (58,90), progress 0.717)",
    "B": "non-deceptive (greedy descent reaches the goal)",
}


def apply_maze(env_cfg, maze: str):
    """Return a copy of `env_cfg` with the named obstacle set installed."""
    if maze not in MAZES:
        raise ValueError(f"unknown maze {maze!r}; choose from {sorted(MAZES)}")
    cfg = copy.deepcopy(env_cfg)
    cfg.obstacles = [tuple(o) for o in MAZES[maze]]
    return cfg


def identify(env_cfg) -> str:
    """Which maze is this? Returns 'A', 'B' or 'custom'."""
    obs = [tuple(float(v) for v in o) for o in env_cfg.obstacles]
    for name, ref in MAZES.items():
        if obs == [tuple(float(v) for v in o) for o in ref]:
            return name
    return "custom"

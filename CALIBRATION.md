# Calibration Record

## Status

CALIBRATED — FROZEN

## Configuration

Config hash: 939192a95e0e

Population: 100
Generations: 100
Environment max_steps: 200

## Calibration

Endpoint conditions:

- Dense reward: eta = 0
- Sparse reward: eta = 1

Seeds:

- 2000
- 2001
- 2002
- 2003
- 2004
- 2005

## Results

### C1 — Dense endpoint solvable but not trivial

PASS

Dense solve rate: 67%

Target range: 30–95%

### C2 — Not solved at initialization

PASS

Median first solution generation: 39.0

### C3 — Topology has range under dense reward

PASS

Median change:

- Nodes: +0.87
- Connections: +2.29

### C4 — Sparsity reduces fitness differentiation

PASS

Distinct fitness values per generation:

- Dense: 54.6
- Sparse: 1.0

## Configuration Hash

939192a95e0e

This configuration is frozen for the pilot and subsequent
experimental preparation unless a documented calibration failure
requires reopening calibration.
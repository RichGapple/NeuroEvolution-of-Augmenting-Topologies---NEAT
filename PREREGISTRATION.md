# Pre-registration

Fill this in **before** running `04_run_sweep.py`, commit it, and do not edit it
afterwards. The roadmap's discipline checklist asks that topology metrics be
defined before inspecting results; this is where that commitment is recorded.

The code logs ~40 quantities per generation. That is deliberate — you want the
trajectory, not just the endpoint — but it also means an unconstrained analysis
has enough freedom to find something in almost any dataset. Declaring one
primary metric per hypothesis in advance is what keeps the rest exploratory
rather than silently confirmatory.

---

## 1. Configuration

| item | value |
|---|---|
| config hash (from `04_run_sweep.py` startup line) | |
| sparsity mode | `quantized` |
| η grid | 0, 0.125, …, 1.0 |
| seeds per condition | (from `03_pilot.py`, not assumed) |
| generations | |
| population size | |
| date frozen | |

Calibration (`00_calibrate.py`) passed on: ____________
Dense-endpoint solve rate at calibration: ______

---

## 2. Primary metrics — one per hypothesis

These carry the confirmatory claims. Everything else in `tests.csv` is
exploratory and must be described as such.

| hypothesis | primary metric | direction predicted | why this one |
|---|---|---|---|
| H1 selection | `mean_selection_differential` | decreases with η | realised selection pressure; the quantity that actually enters the causal chain, unlike distinct-value counts which are partly a property of the quantiser |
| H2 innovation | `mean_tir` | (state a direction or "no directional prediction") | retention of new structure, not merely its introduction |
| H3 complexity | `final_mean_connections` | | connections change before node counts do under NEAT's complexification |
| H4 functionality | `functional_conn_fraction` | | |
| H5 nonlinearity | ΔAIC on the H3 primary | piecewise preferred if ΔAIC > 2 | |

Secondary metrics (reported with effect sizes, not used for confirmatory
claims): _______________________________________________

---

## 3. Analysis plan

- α = 0.05, two-sided.
- Omnibus: Kruskal–Wallis across all η. Post-hoc: Dunn, Holm-adjusted, only if
  the omnibus is significant.
- Multiplicity: the five primary metrics are corrected as a family
  (Holm across the five omnibus p-values). Secondary metrics are not corrected
  and are labelled exploratory.
- Every reported effect carries Cliff's δ and a bootstrap CI. A non-significant
  result with a wide CI is reported as underpowered, not as "no effect".
- `gen_first_solution` is conditional on solving and is never reported without
  the solve rate beside it.

Deviations from this plan, if any, get listed in the manuscript with reasons:

_______________________________________________

---

## 4. Robustness conditions declared in advance

The headline result will be re-run under `--mode gated` and `--mode checkpoint`.
Commitment: if the direction of the H3 primary metric **flips** between modes,
the paper reports the finding as specific to the reward operationalisation
rather than as a property of reward sparsity.

---

## 5. Stopping rule

No peeking-and-extending. The seed count comes from `03_pilot.py` and is fixed
above. If the sweep turns out underpowered, the extra seeds are added for
**all** conditions and the whole analysis is re-run, and that is stated in the
manuscript.

---

## 6. Outcomes that count as success

Per the roadmap, all four of these are publishable and none is a failure:

- a measurable relationship between η and topology/dynamics
- a clear quantitative relationship, e.g. a threshold
- a mechanism linking reward information → fitness differentiation → selection
  → innovation → topology → function
- **a well-controlled null**: η has little effect on topology

The null only counts if calibration check C3 passed — i.e. topology had room to
move under dense reward. Record that here: ______

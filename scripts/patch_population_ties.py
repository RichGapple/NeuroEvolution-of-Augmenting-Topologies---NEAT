#!/usr/bin/env python3
"""Add selection-tie instrumentation to `neat_sparsity/population.py`.

What this measures and why
--------------------------
The paper claims NEAT tolerates coarse reward because parent selection is
*truncation*: sort a species by fitness, keep the top `survival_threshold`
fraction. Truncation reads only the ORDERING, never the fitness magnitudes, so
quantising the reward is harmless right up to the point where it creates enough
ties that the cut line falls inside a tied group. At that point "top 30%" is
decided by genome key rather than by merit, and selection is partly random.

That is currently an argument, not a measurement. This patch makes it a
measurement, by recording per generation:

    arbitrary_selection_fraction
        Fraction of the population whose survive/die outcome was decided by
        key order rather than by fitness, because the truncation cut fell
        strictly inside a group of tied individuals.

    cut_ambiguous_species_fraction
        Fraction of species whose cut was ambiguous at all.

The prediction is sharp and falsifiable: `arbitrary_selection_fraction` should
rise with eta, and the performance cliff should coincide with it crossing a
threshold. If performance collapses while this stays near zero, the
rank-invariance account is wrong and the paper must say so.

Safety
------
The patch is measurement-only: it reads `members`, computes two counters, and
touches nothing that affects reproduction. `--verify` proves this by running
the same seed before and after and asserting the generation records are
identical on every pre-existing field.

Usage
-----
    python scripts/patch_population_ties.py --verify     # check, do not write
    python scripts/patch_population_ties.py --apply      # patch in place
    python scripts/patch_population_ties.py --revert     # restore the backup
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "neat_sparsity", "population.py")
BACKUP = TARGET + ".pre_ties.bak"

ANCHOR = """        for s in self.species:
            members = sorted(s.members, key=lambda g: (-g.fitness, g.key))
            n_survivors = max(1, int(round(len(members) * cfg.survival_threshold)))
            pool = members[:n_survivors]
"""

REPLACEMENT = '''        for s in self.species:
            members = sorted(s.members, key=lambda g: (-g.fitness, g.key))
            n_survivors = max(1, int(round(len(members) * cfg.survival_threshold)))
            pool = members[:n_survivors]

            # -- measurement only; does not affect reproduction --------------- #
            # Truncation selection reads ranks, not magnitudes. It only becomes
            # arbitrary when the cut falls strictly inside a tied group: then
            # who survives is decided by genome key, not by fitness. Count how
            # many individuals had their fate decided that way.
            n_species_considered += 1
            if 0 < n_survivors < len(members):
                boundary = members[n_survivors - 1].fitness
                if members[n_survivors].fitness == boundary:
                    tied = [m for m in members if m.fitness == boundary]
                    n_arbitrary += len(tied)
                    n_ambiguous_species += 1
            # ------------------------------------------------------------------ #
'''

INIT_ANCHOR = """        pop_mean = sum(g.fitness for g in self.genomes) / len(self.genomes)
        parent_fitnesses: List[float] = []
"""

INIT_REPLACEMENT = """        pop_mean = sum(g.fitness for g in self.genomes) / len(self.genomes)
        parent_fitnesses: List[float] = []

        # selection-tie instrumentation (measurement only)
        n_arbitrary = 0
        n_ambiguous_species = 0
        n_species_considered = 0
"""

STORE_ANCHOR = """        self.genomes = new_genomes
        self.generation += 1
"""

STORE_REPLACEMENT = """        self.genomes = new_genomes
        self.generation += 1

        # Exposed for the generation record. See scripts/patch_population_ties.py
        # for what these mean and why they are the paper's mechanism test.
        self.arbitrary_selection_fraction = n_arbitrary / max(1, cfg.pop_size)
        self.cut_ambiguous_species_fraction = (
            n_ambiguous_species / max(1, n_species_considered))
"""

INIT_ATTRS_ANCHOR = """        self._structural_events = 0
"""

INIT_ATTRS_REPLACEMENT = """        self._structural_events = 0

        # selection-tie instrumentation (measurement only)
        self.arbitrary_selection_fraction = 0.0
        self.cut_ambiguous_species_fraction = 0.0
"""

RECORD_ANCHOR = """            reached_goal_count=n_reached,
        )
        records.append(rec)
"""

RECORD_REPLACEMENT = """            reached_goal_count=n_reached,
        )
        # mechanism metrics from the previous generation's reproduction step
        rec["arbitrary_selection_fraction"] = pop.arbitrary_selection_fraction
        rec["cut_ambiguous_species_fraction"] = pop.cut_ambiguous_species_fraction
        records.append(rec)
"""

PATCHES = [
    (INIT_ATTRS_ANCHOR, INIT_ATTRS_REPLACEMENT, "Population.__init__ defaults"),
    (INIT_ANCHOR, INIT_REPLACEMENT, "reproduce() counters"),
    (ANCHOR, REPLACEMENT, "selection-cut tie detection"),
    (STORE_ANCHOR, STORE_REPLACEMENT, "expose the fractions"),
    (RECORD_ANCHOR, RECORD_REPLACEMENT, "write into the generation record"),
]


def is_patched(src: str) -> bool:
    return "arbitrary_selection_fraction" in src


def apply_patches(src: str) -> str:
    for anchor, replacement, name in PATCHES:
        if anchor not in src:
            raise SystemExit(
                f"could not find the anchor for '{name}'.\n"
                f"population.py has diverged from the expected version; "
                f"patch by hand or send me the current file.")
        src = src.replace(anchor, replacement, 1)
    return src


# --------------------------------------------------------------------------- #
def verify(patched_src: str) -> bool:
    """Run the same seed with and without the patch; assert identical results.

    This is the check that matters. Instrumentation that changes behaviour is
    worse than no instrumentation, because it silently invalidates every run
    made after it was added.
    """
    import importlib
    import importlib.util
    import tempfile
    import types

    print("verifying the patch changes nothing...")

    original = open(TARGET, encoding="utf-8").read()

    def run_once(source: str, tag: str):
        # Load the module under a private name so both versions can coexist.
        mod_name = f"_pop_{tag}"
        path = os.path.join(tempfile.mkdtemp(), "population.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        import neat_sparsity
        spec = importlib.util.spec_from_file_location(
            f"neat_sparsity.{mod_name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"neat_sparsity.{mod_name}"] = mod
        # rewrite relative imports to absolute so the temp module loads
        mod.__package__ = "neat_sparsity"
        spec.loader.exec_module(mod)

        from neat_sparsity.config import core_config
        from neat_sparsity.env import NavEnv
        from neat_sparsity.network import build_network
        from neat_sparsity.reward import RewardModel, true_objective

        cfg = core_config()
        cfg.generations = 12
        cfg.neat.pop_size = 40
        env = NavEnv(cfg.env)
        model = RewardModel(cfg.sparsity, 0.5)

        def evaluate(genomes):
            fits, trues, n = [], [], 0
            for g in genomes:
                net = build_network(g, cfg.neat)
                traj = env.rollout(net)
                fits.append(model.fitness(traj))
                trues.append(true_objective(traj, cfg.sparsity))
                n += int(traj.reached_goal)
            return fits, trues, n

        recs, best, pop = mod.run_evolution(
            cfg.neat, seed=7, evaluate=evaluate, generations=cfg.generations)
        return recs

    a = run_once(original, "orig")
    b = run_once(patched_src, "patched")

    if len(a) != len(b):
        print(f"  FAIL: different number of generations ({len(a)} vs {len(b)})")
        return False

    new_fields = {"arbitrary_selection_fraction", "cut_ambiguous_species_fraction"}
    bad = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in ra:
            if k in new_fields:
                continue
            va, vb = ra[k], rb.get(k)
            same = (va == vb) or (isinstance(va, float) and isinstance(vb, float)
                                  and (va != va) and (vb != vb))
            if not same:
                bad.append(f"gen {i} field {k!r}: {va} != {vb}")
    if bad:
        print(f"  FAIL: {len(bad)} differing fields, first few:")
        for line in bad[:5]:
            print(f"    {line}")
        return False

    present = all(k in b[-1] for k in new_fields)
    print(f"  PASS: {len(a)} generations identical on every pre-existing field")
    print(f"  new fields present: {present}")
    if present:
        vals = [r["arbitrary_selection_fraction"] for r in b]
        print(f"  arbitrary_selection_fraction over the trial: "
              f"min {min(vals):.3f}, max {max(vals):.3f}")
    return present


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true",
                   help="check the patch changes nothing; do not write")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        if not os.path.exists(BACKUP):
            raise SystemExit(f"no backup at {BACKUP}")
        shutil.copy2(BACKUP, TARGET)
        print(f"restored {TARGET} from backup")
        return

    src = open(TARGET, encoding="utf-8").read()
    if is_patched(src):
        print("population.py is already patched.")
        if args.apply:
            return
        src_patched = src
    else:
        src_patched = apply_patches(src)
        print("patch applies cleanly (5 hunks)")

    if args.verify:
        ok = verify(src_patched)
        sys.exit(0 if ok else 1)

    if args.apply:
        if not os.path.exists(BACKUP):
            shutil.copy2(TARGET, BACKUP)
            print(f"backup -> {BACKUP}")
        with open(TARGET, "w", encoding="utf-8") as fh:
            fh.write(src_patched)
        print(f"patched {TARGET}")
        print("\nNOTE: this does not change the config, so the config hash is")
        print("unchanged and new runs remain poolable with existing ones.")
        print("Re-run tests/test_smoke.py to confirm nothing broke.")


if __name__ == "__main__":
    main()

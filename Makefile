PY      ?= python3
WORKERS ?= 8
OUT     ?= results/core_sweep
SEEDS   ?= 30

.PHONY: test check baseline calibrate pilot sweep ablation analyze methods robust all clean

test:      ; $(PY) tests/test_smoke.py
check:     ; $(PY) scripts/02_sparsity_check.py
baseline:  ; $(PY) scripts/01_baseline.py --check-reproducibility
calibrate: ; $(PY) scripts/00_calibrate.py --workers $(WORKERS)
pilot:     ; $(PY) scripts/03_pilot.py --workers $(WORKERS)
sweep:     ; $(PY) scripts/04_run_sweep.py --out $(OUT) --workers $(WORKERS) --seeds $(SEEDS)
ablation:  ; $(PY) scripts/05_ablation.py --out $(OUT)
analyze:   ; $(PY) scripts/06_analyze.py --out $(OUT)
methods:   ; $(PY) scripts/09_methods.py --out $(OUT)

robust:
	$(PY) scripts/04_run_sweep.py --mode gated      --out results/robust_gated      --workers $(WORKERS) --seeds $(SEEDS)
	$(PY) scripts/04_run_sweep.py --mode checkpoint --out results/robust_checkpoint --workers $(WORKERS) --seeds $(SEEDS)
	$(PY) scripts/08_compare.py --a $(OUT) --b results/robust_gated      --label-a quantized --label-b gated
	$(PY) scripts/08_compare.py --a $(OUT) --b results/robust_checkpoint --label-a quantized --label-b checkpoint

all: test check baseline calibrate sweep ablation analyze methods

clean: ; rm -rf results

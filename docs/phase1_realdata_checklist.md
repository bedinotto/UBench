# Phase-1 real-data validation checklist (GPU box)

**Why this exists.** Phase 1 is *smoke-green*: `./run.sh` runs end-to-end,
honestly, on the synthetic CPU fixture (`codes/tests/`), and all 41 tests pass
in CI. It is **not** real-data-green. The outstanding half of Phase-1's
Definition of Done — a full run on the real corpus (≈10 subjects, K=5) with the
resume drill exercised — can only be closed on a machine with the Git-LFS data
and an NVIDIA GPU. That is this checklist. Run it on the GPU box, capture the
evidence, and record any deviation as a new ledger row.

> **Provisional numbers.** Any timing / VRAM figures produced here are
> *provisional*. Per-epoch timing was removed as dishonest (UB-09, done);
> cross-model VRAM is still compared at unequal batch sizes until the fixed-batch
> probe lands (UB-10 / T2.3). Do **not** quote latency or memory as results yet.
> mIoU / Dice / loss are now on the single hard-metric authority (UB-11, done)
> and are comparable across the trainer and the benchmark.

---

## 0. Preconditions

- [ ] NVIDIA GPU with **≥ 6 GB** VRAM (GTX 1660 Ti baseline tier; AMP is
      auto-disabled on GTX/CPU — do not force it on).
- [ ] Git **LFS** installed (`git lfs version`).
- [ ] Python 3.10 or 3.11 (GPU training boxes stay on 3.10/3.11 until T3.5
      verifies the CUDA wheels on 3.13).

## 1. Fresh clone + data

```bash
git clone <repo-url> ubench-realrun && cd ubench-realrun
git lfs pull                     # pulls requirements/*.zip (the real corpus)
```

- [ ] `git lfs pull` completed; `ls -la requirements/*.zip` shows non-pointer
      files (sizes in MB, not ~130-byte pointers).

## 2. Environment (GPU path, §3.1)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python codes/setup.py                       # validates env + installs CUDA torch
pip install -r requirements/requirements.txt
```

- [ ] `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
      prints a CUDA build and `True`.

> ⚠ **UB-20b caveats (until T3.5).** `codes/setup.py` installs torch with
> `--force-reinstall --no-deps` against a pinned `cu121` index and dependencies
> are unpinned (`>=`). If the install resolves a broken combination, capture the
> exact versions (`pip freeze > freeze.txt`) and file it against UB-20 — do not
> hand-patch and continue silently.

## 3. Dry run (single model, 2 epochs)

```bash
python codes/main_pipeline.py --models unet --epochs 2
```

- [ ] Preprocessing runs automatically (no manual `preprocess_data.py`);
      `data/processed/metadata.csv` appears.
- [ ] `--epochs 2` is honored — training stops at epoch 2, not 100 (UB-13).
- [ ] Exit code `0`; no `TRAINING FAILURE` summary (UB-07).

## 4. Full run

```bash
./run.sh
```

- [ ] Fold count reported as **K = min(5, n_subjects)** with a clear message if
      the corpus has < 5 subjects (leave-subjects-out CV, UB-03/04).
- [ ] All three models train (`unet`, `transunet`, `swin`) across all folds.
- [ ] SUCCESS banner prints **only** on zero recorded failures; a failed
      model/fold makes the run exit non-zero with a traceback in
      `error_log_*.txt` (UB-07).

## 5. Resume drill (the Phase-1 headline behavior — UB-06)

Interrupt a full run mid-training, then resume it:

```bash
./run.sh                         # let a couple of folds/epochs complete, then Ctrl-C
#   note the run id printed near the top, e.g. 20260717_1330
python codes/main_pipeline.py --resume <run_id>   # continues; or ./run.sh ... --resume <run_id>
```

- [ ] `--resume <run_id>` reuses `outputs/<run_id>` + `logs/<run_id>` — **no new
      timestamped dir is minted** for the resumed run.
- [ ] Training continues from the last epoch checkpoint (metric history length
      grows; it does not restart from epoch 1).
- [ ] An unknown run id fails with an actionable `FileNotFoundError` that lists
      the available run ids.

## 6. Expected artifacts

- [ ] Exactly **one** `logs/<ts>/` directory per `./run.sh` invocation (shell and
      Python share one `UBENCH_RUN_ID` — UB-14); no duplicate timestamp dir.
- [ ] `outputs/latest` symlink resolves to the newest `outputs/<run_id>/`.
- [ ] `outputs/<run_id>/benchmark_comparison.csv` has **3 rows**:
      `U-Net`, `TransUNet`, `Swin-UNet++`.
- [ ] `logs/<run_id>/benchmark_report.txt` states the timing conditions
      (warm-up count, batch size, dtype, device, torch version) and labels
      metrics as hard/macro (UB-09/UB-11).

## 7. Capture protocol

For the record, save alongside this checklist:

- [ ] `pip freeze` output, GPU name (`nvidia-smi -L`), `torch.__version__`.
- [ ] The full console log (`logs/<run_id>/console.log`) and
      `logs/<run_id>/pipeline.log`.
- [ ] `benchmark_comparison.csv` and `benchmark_report.txt`.

**Any deviation from an expected result becomes a new `UB-##` ledger row**
(CLAUDE.md §5) with the logs attached — not a silent workaround (R4). A green
run here closes the real-data half of Phase-1's DoD; the numbers stay labeled
*provisional* until UB-10 / T2.3 makes VRAM and latency trustworthy.

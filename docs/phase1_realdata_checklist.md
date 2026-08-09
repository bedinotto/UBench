# Real-data validation checklist (GPU box)

**Why this exists.** The test suite and CI are CPU-only, on the synthetic
fixture in `codes/tests/`. Green there means the pipeline *runs honestly*; it
does not mean the real corpus has been through it from a clean checkout. This
checklist is the protocol for that, and it is the last open item of the
Definition of Done (CLAUDE.md §10). Run it on a machine with the Git-LFS data
and an NVIDIA GPU, capture the evidence, and record any deviation as a new
ledger row (§5) rather than a silent workaround (R4).

## Status: what the reported run already closed

Run **`2026-07-25_10-55-17`** (15 model-folds, 186.9 h of GPU) is a real
full-corpus execution, and its artifacts are preserved in the repo. Verified
present on disk:

| Item | Evidence |
|---|---|
| Full run, 3 architectures × 5 folds | `outputs/2026-07-25_10-55-17/models/`, `logs/.../{unet,transunet,swin_unet_plus_plus}_fold_{1..5}_metrics.json` |
| 3-row benchmark comparison | `outputs/2026-07-25_10-55-17/benchmark_comparison.csv` |
| Provenance recorded | `logs/2026-07-25_10-55-17/run_metadata.json` |
| Capture protocol (§7) | `console.log`, `pipeline.log`, `benchmark_report.txt`, `hardware_profile.json` |
| CUDA lock generated and committed (§2) | `requirements/requirements.cuda.lock` |

**What that run does *not* close**, and why this file survives:

- It was **not** launched from a fresh clone. Its own metadata records
  `git_dirty: true`, so the nominal commit does not fully describe the code
  that ran. §1 and §3 below remain unexercised.
- The **resume drill** (§5) was never run against the real corpus. It is
  verified only by `codes/tests/test_resume.py` on the synthetic fixture.
- It used the shipped default `test_subjects` (empty), so the report is
  **CV-only**. See the held-out note below before producing a citable run.

> **Held-out test subjects (M1).** A citable run **must** set
> `training.test_subjects` in `codes/config.yaml` (or the `TEST_SUBJECTS` env
> var) — reserve 1–2 subjects (e.g. `["S9", "S10"]`) held out from all CV folds
> and scored separately. With the default (empty) there is **no TEST section**.
> Test subjects are still preprocessed; they are only filtered out of the CV
> pool. A holdout that leaves fewer than 2 CV subjects, or names a subject not
> in the data, is a hard error.

> **On quoting numbers.** Latency and peak VRAM *are* now trustworthy, under
> the conditions the report discloses: synchronized timing with warm-up
> discarded (UB-09), a fixed-batch probe (UB-10) isolated from other resident
> models (UB-28), and one hard-metric authority shared by trainer and benchmark
> (UB-11). Quote them **with** those conditions attached (M9), never bare.

---

## 0. Preconditions

- [ ] NVIDIA GPU with **≥ 6 GB** VRAM (GTX 1660 Ti baseline tier; AMP is
      auto-disabled on GTX/CPU — do not force it on).
- [ ] Git **LFS** installed (`git lfs version`).
- [ ] Python 3.10 or 3.11. CUDA torch wheels are not published for 3.13+, and
      `codes/setup.py` hard-errors the CUDA path there.

## 1. Fresh clone + data

```bash
git clone <repo-url> ubench-realrun && cd ubench-realrun
git lfs pull                     # pulls requirements/*.zip (the real corpus)
```

- [ ] `git lfs pull` completed; `ls -la requirements/*.zip` shows non-pointer
      files (sizes in MB, not ~130-byte pointers).
- [ ] `git status --porcelain` is **empty** before training starts, so the run's
      `git_dirty` is `false` and the commit describes the code that ran.

## 2. Environment (GPU path, §3.1)

`requirements/requirements.cuda.lock` is committed. Install from it:

```bash
uv venv --python 3.11 && source .venv/bin/activate
python -m codes.setup            # auto-detects the driver → cu121/cu118, syncs the lock
```

`setup.py` regenerates the lock only if the file is absent. If this box's driver
needs a different tag than auto-detect picks, override with
`UBENCH_TORCH_BACKEND=cu121|cu118`; if the lock itself is stale for this box,
recompile and commit it:

```bash
uv pip compile pyproject.toml --extra dev --python-version 3.10 \
    --torch-backend=cu121 -o requirements/requirements.cuda.lock
```

- [ ] `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
      prints a CUDA build and `True`.
- [ ] `logs/<run>/run_metadata.json` `lockfile_hash` == `sha256sum` of the lock
      actually installed from.

## 3. Dry run (single model, 2 epochs)

> **Rebuild processed data first (T3.4).** Processed data stores **Celsius**
> plus a `data/processed/preprocess_manifest.json` schema version. Any
> `data/processed/` created before T3.4 has no manifest and **hard-errors at
> load**, by design — legacy [0,1] data would be misread as °C.

```bash
python -m codes.main_pipeline --models unet --epochs 2 --force-preprocess
```

- [ ] Preprocessing runs automatically (no manual `preprocess_data.py`);
      `data/processed/metadata.csv` **and** `preprocess_manifest.json` appear.
- [ ] `--epochs 2` is honored — training stops at epoch 2, not 100 (UB-13).
- [ ] Exit code `0`; no `TRAINING FAILURE` summary (UB-07).

## 4. Full run

```bash
./run.sh
```

Five models are registered. `./run.sh` with no `--models` trains the
from-scratch trio, which is what the reported run did. To compare pretrained
against scratch (M5), name them explicitly:

```bash
./run.sh --models unet transunet swin swin_pretrained transunet_pretrained
```

- [ ] Fold count reported as **K = min(5, n_subjects)**, with a clear message if
      the corpus has fewer than 5 subjects (leave-subjects-out CV, UB-03/04).
- [ ] Every requested model trains across all folds.
- [ ] SUCCESS banner prints **only** on zero recorded failures; a failed
      model/fold exits non-zero with a traceback in `error_log_*.txt` (UB-07).
- [ ] Pretrained keys, if requested, downloaded their weights (they are gated by
      `pretrained` / `UBENCH_PRETRAINED`; tests and the CPU smoke never fetch).

## 5. Resume drill (UB-06) — still open

Interrupt a full run mid-training, then resume it:

```bash
./run.sh                         # let a couple of folds/epochs complete, then Ctrl-C
#   note the run id printed near the top
python -m codes.main_pipeline --resume <run_id>
```

- [ ] `--resume <run_id>` reuses `outputs/<run_id>` + `logs/<run_id>` — **no new
      timestamped dir is minted**.
- [ ] Training continues from the last epoch checkpoint (metric history length
      grows; it does not restart from epoch 1).
- [ ] An unknown run id fails with an actionable `FileNotFoundError` listing the
      available run ids.
- [ ] Resuming a run whose recipe no longer matches the config **hard-errors**
      instead of loading an AdamW state into Adam (UB-18/T3.3).

## 6. Expected artifacts

- [ ] Exactly **one** `logs/<ts>/` directory per `./run.sh` invocation (shell and
      Python share one `UBENCH_RUN_ID` — UB-14).
- [ ] `outputs/latest` resolves to the newest `outputs/<run_id>/`.
- [ ] `outputs/<run_id>/benchmark_comparison.csv` has one row per trained model.
- [ ] `logs/<run_id>/benchmark_report.txt` states the timing conditions (warm-up
      count, batch size, dtype, device, torch version), labels metrics as
      hard/macro (UB-09/UB-11), and reports VRAM at the fixed probe batch
      (UB-10/UB-28).
- [ ] With `test_subjects` set, the report carries **both** a CV and a TEST
      section (T2.4/M1).

## 7. Capture protocol

Save alongside this checklist:

- [ ] `nvidia-smi -L`, `torch.__version__`, and the installed lock's sha256.
- [ ] `logs/<run_id>/console.log` and `logs/<run_id>/pipeline.log`.
- [ ] `benchmark_comparison.csv` and `benchmark_report.txt`.
- [ ] `logs/<run_id>/run_metadata.json` (git SHA, dirty flag, seed, determinism,
      effective config, normalization mode).

**Any deviation from an expected result becomes a new `UB-##` ledger row**
(CLAUDE.md §5) with the logs attached — not a silent workaround (R4).

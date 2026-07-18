# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It merges the project orientation guide with the **engineering & remediation program**: §1–§4 describe what the repo is and how it actually behaves today; §5–§12 govern how you work on it.

> **Role you assume:** Senior software engineer & computer scientist specialized in AI/ML systems. You verify empirically, you never assert what you have not executed, and you leave every file better than you found it.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## 1. What this is

UBench is a computer vision pipeline for **thermal facial region segmentation** (10 classes: background + 9 facial regions, Portuguese labels). It trains and benchmarks three architectures — **U-Net**, **TransUNet**, and **Swin-UNet++** — under a unified training loop on thermal `.tiff` images (K-fold cross-validation, AMP, hardware-aware batch sizing), then produces comparison reports and plots. All Python source lives in `codes/`; the repo root is kept clean (entry points + config only).

## 2. Current state — READ FIRST

**Phase 1 is complete** (T1.1–T1.8 landed): `./run.sh` works end-to-end, honestly, on smoke-level synthetic CPU data — preprocessing runs automatically, all 3 models train, and a 3-row `benchmark_comparison.csv` is produced. Partial-corpus runs reduce the fold count with a warning (leave-subjects-out CV; <2 subjects is an actionable error). Border-adjacent bounding boxes no longer shift masks (UB-08). Training failures are recorded and fatal: the SUCCESS banner requires zero recorded failures, failed runs exit non-zero with tracebacks in `error_log_*.txt`, and `--fail-fast` aborts on the first failure (UB-07). Interrupted runs resume: `--resume <run_id>` reuses `outputs/<run_id>` + `logs/<run_id>` and continues from epoch checkpoints with the metric history intact; `outputs/latest` points at the newest run (UB-06). `--epochs` is honored unconditionally (UB-13), and shell + Python share one `UBENCH_RUN_ID` — one `outputs/<ts>` and one `logs/<ts>` per `./run.sh` invocation (UB-14). Smoke-green ≠ real-data-green: a full real-data run (10 subjects, K=5) is plausible but **unverified** — `docs/phase1_realdata_checklist.md` is the GPU-box protocol that closes it (fresh clone → LFS pull → dry run → full `./run.sh` → resume drill → artifact checks). Phase 2 (make the *numbers trustworthy*) is underway: UB-09 (unsynced per-epoch timing) and UB-11 (inconsistent metric definitions) are **FIXED** — per-epoch inference timing is gone, and benchmark latency now runs through `timed_inference` (warm-up-discarded, CUDA-synced, conditions disclosed in the report); a single hard-metric authority (`codes/metrics.py`, argmax IoU + Dice, macro excl. absent classes, background separate) serves both trainer and benchmark, and the benchmark loss is the training CE+Dice. UB-10 (VRAM at unequal batch sizes) is **FIXED**: a fixed-batch inference probe (`probe_peak_memory`, "VRAM @ batch=4 (fixed)", CPU→"n/a"). Held-out test subjects landed (T2.4/M1): opt-in `test_subjects` (default empty) are excluded from all CV folds and each fold-model is scored on them, so the benchmark report/CSV carry both a CV and a TEST section. **Phase 2 is now complete** — the numbers are trustworthy: synchronized warm-up timing (UB-09), one hard-metric authority (UB-11), fixed-batch inference VRAM (UB-10), held-out test subjects (T2.4), and seeded reproducibility (T2.5: seeded DataLoader `generator` + per-worker init, a single cuDNN owner via the `deterministic` flag, per-run metadata JSON); dead code and scratch scripts are gone (T2.6: UB-15/UB-22, with UB-24 stray-output-dirs and UB-25 checkpoint-slimming folded in). The one outstanding cross-phase item is the **real-data validation on the human GPU box** (`docs/phase1_realdata_checklist.md`), which a citable run must pair with `test_subjects`. The frontier is now **Phase 3 (make the *science* credible)**, opening with T3.1 (UB-12 config consolidation). Every defect is catalogued in the ledger (§5) with verified locations; treat it as ground truth, do not re-litigate it, and do re-verify each item with a test as you fix it. Work proceeds in phases (§9): make it *run* → make the *numbers trustworthy* → make the *science credible* → enhance.

**Prime directive:** *No task is "done" until its acceptance test passes in a real execution.* Reading code is not verification. If you cannot run something, say so explicitly and mark the task blocked.

The safety net exists: `codes/tests/` is a real committed suite (41 tests, CPU-only) with ruff and CI (§3.5, §7.4). The smoke test — not "running the pipeline" — is how changes are validated. (`codes/tests/` still also holds legacy gitignored debug scripts; their removal is T2.6.)

---

## 3. Commands & Environment

### 3.1 Environment setup (always a venv — never system Python)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip

# Dev/CI: CPU-only torch (fast, deterministic; correctness work needs no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/requirements.txt
```

PyTorch is **not** in `requirements/requirements.txt` — on GPU training boxes it is installed by `codes/setup.py` with a CUDA-specific index URL (`cu121`/`cu118`) to avoid pip's CPU-only default (⚠ UB-20: unpinned + aging index; T3.5 replaces this). Python 3.13 is verified for CPU-only dev work (Phase 0/D1); GPU training boxes remain on 3.10/3.11 until T3.5 verifies CUDA wheels. All remediation work happens **CPU-first**; GPU is required only for Phase-3 retraining and real benchmark numbers.

### 3.2 Entry points and flag forwarding

```bash
./run.sh                                  # Linux/Mac
run.bat                                   # Windows
./run.sh --skip-extract --skip-setup --models unet --epochs 10
./run.sh --models transunet swin          # model choices: unet, transunet, swin
./run.sh --skip-benchmark
```

`run.sh`/`run.bat` handle `--skip-extract` and `--skip-setup` themselves and forward everything else (`--models`, `--epochs`, `--skip-benchmark`, `--fail-fast`, `--resume`) to `codes/main_pipeline.py`. Both scripts export their timestamp as `UBENCH_RUN_ID` so shell and Python share one run identity (UB-14; run.bat's export is UNVERIFIED — no Windows box). When debugging a single stage, call the `codes/*.py` script directly instead of going through `run.sh`.

### 3.3 Running stages directly

```bash
python codes/extract_data.py          # unzip requirements/*.zip -> data/ (needs Git LFS)
python codes/setup.py                 # env validation + CUDA torch install (GPU boxes)
python codes/preprocess_data.py       # writes data/processed/ + metadata.csv; the pipeline runs this
                                      # automatically when metadata.csv is absent (T1.1);
                                      # --force-preprocess on main_pipeline.py rebuilds unconditionally
python codes/main_pipeline.py --models unet --epochs 10 --skip-benchmark
```

**Correction to previous guidance:** `python codes/benchmark_models.py` does **not** run a standalone benchmark — its `__main__` only prints a readiness message. Benchmarking runs via `main_pipeline.py` (or by importing `run_benchmark()` yourself).

### 3.4 Fast dev loops (env vars honored by the code)

```bash
LIMIT_SAMPLES=20 NUM_EPOCHS=1 K_FOLDS=2 NUM_WORKERS=0 \
  python codes/main_pipeline.py --models unet --skip-benchmark
```

`LIMIT_SAMPLES` truncates metadata rows; `NUM_EPOCHS`, `K_FOLDS`, `NUM_WORKERS` override config/hardware values; `UBENCH_LOG_DIR` redirects child-script logging. Every behavior must be provable in seconds with these — **never** launch a 100-epoch or full-dataset run as part of development or CI.

### 3.5 Gates (exist after Phase 0)

```bash
ruff check codes/                          # lint gate
pytest codes/tests -x -q                   # unit/integration gate
pytest codes/tests/test_pipeline_smoke.py -x -q   # E2E gate — the definition of "the repo works"
```

---

## 4. Architecture

### 4.1 Execution flow

```
run.sh / run.bat
  └─ codes/extract_data.py      # unzip requirements/*.zip -> data/, generate annotations
  └─ codes/setup.py             # validate Python/CUDA, install deps
  └─ codes/main_pipeline.py     # orchestrator (Pipeline class)
       ├─ codes/hardware_detector.py   # GPU probe, min-6GB check, batch-size/worker scaling
       ├─ codes/preprocess_data.py     # invoked automatically when data/processed/metadata.csv absent (T1.1)
       ├─ codes/unified_data.py        # Config, S1..S10 discovery, GroupKFold loaders
       ├─ codes/unified_training.py    # UnifiedTrainer: shared train/val loop, checkpoints
       │    ├─ codes/unet_v2.py                (registry: "unet")
       │    ├─ codes/transunet.py              (registry: "transunet")
       │    └─ codes/swin_unet_plus_plus.py    (registry: "swin_unet_plus_plus")
       └─ codes/benchmark_models.py    # cross-model comparison, plots, reports
```

**Data-flow contract:** `extract_data.py` → `preprocess_data.py` (produces `data/processed/{images/*.npy, masks/*.png, metadata.csv}`) → `create_single_fold_loader` (lazy, one fold at a time; `create_kfold_data_loaders` exists but the pipeline uses the single-fold variant to avoid worker/semaphore leaks) → `UnifiedTrainer` (saves `outputs/<run>/models/best_<key>_fold_<n>_model.pth` + full `checkpoints/`; all paths from `codes/naming.py`) → `run_benchmark` (loads those weights via the same authority, writes `benchmark_comparison.csv`, plots, report).

### 4.2 Configuration — where truth actually lives

`codes/unified_data.py:Config` loads **`codes/config.yaml`** (`Path(__file__).parent / "config.yaml"`): image size, `num_classes`, learning rate, epochs, `k_folds`, seed, region names. The root-level `config.yaml` — despite documenting loss weights, optimizer, scheduler, augmentation, and hardware sections — **is loaded by nothing** (UB-12). Do not "fix" behavior by editing the root file; it is dead until T3.1 consolidates to a single validated schema. Within the loaded file, `training.batch_sizes` is also unconsumed, and the `CLASS_WEIGHTS` branch in `unified_training.py` has no producer. `--epochs` on the CLI overrides config via the `NUM_EPOCHS` env var (`apply_epochs_override` in `main_pipeline.py`), exported whenever the flag is given (UB-13 fixed, T1.8).

### 4.3 Hardware auto-scaling

`codes/hardware_detector.py:detect_and_optimize` profiles the GPU at pipeline start and picks per-model batch sizes, worker count, and AMP strategy (GTX-class cards and CPUs get AMP disabled; the 6 GB GTX 1660 Ti is the baseline tier). Batch-size keys are the canonical registry names (`unet`, `transunet`, `swin_unet_plus_plus`) in every tier, consumed via hard `[key]` lookup; per-tier key parity with the model registry is test-enforced (`test_batch_size_keys.py`), so registering a new model forces a batch-size entry (UB-05 fixed, T1.3). Do not disable mixed precision for memory-constrained GPUs.

### 4.4 Multi-dataset discovery and ID namespacing

`codes/unified_data.py:MultiDirectoryDataLoader` scans `data/` for `S1`–`S10` directories and merges them, prefixing sample IDs by source dataset (e.g. `S1/R11104`) to keep them unique. Each dataset's annotations can live in either of two layouts — both are checked automatically: `data/S1_polygonal_masks.json` + `data/S1_bounding_boxes.csv` (root-level, recommended), or `data/S1/polygonal_masks.json` + `data/S1/bounding_boxes.csv` (inside the dataset dir). TIFF resolution goes through `get_tiff_path`, which extracts digits from the sample ID and expects `R{digits}.tiff`.

### 4.5 Splits — what the code actually does

Splitting is `GroupKFold(groups=df['dataset'])`: **whole subject directories are held out per fold** (leave-subjects-out CV). This is the methodologically correct choice for face data (frames of one person are near-duplicates), and it is *not* a stratified per-dataset split. `resolve_fold_count` (in `unified_data.py`, resolved once by the pipeline and by both loader factories) reduces `k_folds` to the subject count with a loud warning and raises an actionable error below 2 subjects (UB-03/04 fixed, T1.4). Held-out **test subjects** landed (T2.4/M1): `config.test_subjects` (default empty; `TEST_SUBJECTS` env override) are removed from the CV pool inside `load_split_metadata` — so the fold-count guard and both loader factories operate on the training pool and no fold can see them — and `create_test_loader` yields exactly those subjects; an unknown id or a holdout leaving <2 CV subjects is a hard error. The benchmark scores each fold-model on the held-out set (CV selects, TEST headlines; never tuned on test).

### 4.6 Shared training loop, model-specific architectures

All three models train through the same `UnifiedTrainer` (`codes/unified_training.py`): identical augmentations, combined Cross-Entropy + Dice loss (`CombinedLoss`), `ReduceLROnPlateau`, gradient clipping, NaN protection, per-epoch checkpointing. This shared loop is what makes cross-model numbers comparable — **when adding a new model architecture**: implement it as an `nn.Module` decorated with `@register_model("<key>")` in a new `codes/<model>.py`, import the module in `main_pipeline.py` so registration runs, add its batch-size key to `hardware_detector.py`, and integrate through `UnifiedTrainer` rather than writing a parallel training loop. Be aware that "identical hyperparameters" is *not* automatically fair across architecture families (UB-18, M4) — per-family recipes arrive in T3.3.

### 4.7 Run identity & resume

Every `main_pipeline.py` invocation writes all artifacts under `outputs/<run_id>/` (models, plots, checkpoints) and `logs/<run_id>/` (per-stage logs, metrics JSON, benchmark report). The run id is resolved in `Pipeline.__init__` with precedence **`--resume` > `UBENCH_RUN_ID` > fresh timestamp**: `run.sh`/`run.bat` export their timestamp as `UBENCH_RUN_ID`, so one `./run.sh` invocation produces exactly one run dir pair (UB-14); `--resume <run_id>` reuses an existing run's dirs so `UnifiedTrainer._find_latest_checkpoint()` picks up its epoch checkpoints and the metric history continues rather than restarting (UB-06). A missing `--resume` run dir is an actionable `FileNotFoundError` listing available run IDs. `outputs/latest` is a symlink to the newest run dir, replaced atomically; nothing from a previous run is overwritten unless you `--resume` it. In `run.sh`, Python's TeeLogger is the sole writer of `logs/<ts>/pipeline.log`; the shell's own capture goes to `console.log`.

### 4.8 Dev/debug scripts

`codes/tests/` is the committed test suite (Phase 0). The former ad-hoc debugging scripts (NaN-in-AMP diagnostics `test_unet_nan*.py`, `test_suite.py`) and one-off inspection utilities (`inspect_ids.py`, `verify_regex.py`, `print_all_missing.py`, `test_edge_cases.py`) were **deleted** in T2.6 (UB-22), along with `inference_comparison.py` (orphaned) and both `collect_ignore_glob` hooks — there is no scratch code left outside the pipeline's execution path.

---

## 5. Known-Defect Ledger (verified — single source of truth)

Update the **Status** column as work lands (`OPEN → IN-PROGRESS → FIXED@<sha>`). Never delete rows; history matters.

| ID | Sev | Location | Defect (verified behavior) | Status |
|----|-----|----------|----------------------------|--------|
| UB-01 | Blocker | `run.sh`, `main_pipeline.py` | `preprocess_data.py` is never invoked by any entry point; loaders require `data/processed/metadata.csv` → fresh clone fails 15/15 train attempts. README omits the step. | FIXED@03e6dbb |
| UB-02 | Blocker | `benchmark_models.py:~443-452` vs `main_pipeline.py:train_model` | Filename contract mismatch: training saves `best_unet_fold_N_model.pth`, `best_swin_unet_plus_plus_fold_N_model.pth` (registry names); benchmark searches `best_u_net_…`/`best_swin_unetplusplus_…` (display names via `_safe_filename`). Only TransUNet matches → comparison silently contains 1 of 3 models. | FIXED@dd5fa8d |
| UB-03 | Blocker | `unified_data.py:~427,~540` | `GroupKFold(n_splits=K_FOLDS)` with `groups=df['dataset']` raises `ValueError` whenever #datasets < K (default 5). | FIXED@73967c8 |
| UB-04 | Blocker/Docs | `unified_data.py`, `README` | Split semantics contradict docs: leave-subjects-out in code vs stratified-per-dataset in README, whose example fold counts GroupKFold cannot produce. Fix the docs, not the split. | FIXED@73967c8 |
| UB-05 | Blocker | `main_pipeline.py:121` + `hardware_detector.py` | Batch-size keys `{unet, transunet, swin}` vs lookup `swin_unet_plus_plus` → `.get(key, 8)` silently returns 8. OOM risk at the advertised 6 GB minimum; waste on large GPUs. | FIXED@cd3afa4 |
| UB-06 | Major | `main_pipeline.py`, `unified_training.py` | Auto-resume dead across restarts: checkpoints under `outputs/<timestamp>/checkpoints`, new timestamp per invocation → `_find_latest_checkpoint()` scans empty dir. *Also found during fix: numpy scalars in the checkpoint's metric history were rejected by `torch.load(weights_only=True)`.* | FIXED@01244fa |
| UB-07 | Major | `main_pipeline.py` | Failures swallowed: per-model/fold `try/except` prints one line, continues; SUCCESS banner + exit 0 possible with zero trained models; `main()`'s error-log writer unreachable. | FIXED@74e1b43 |
| UB-08 | Major | `preprocess_data.py:66-67` vs `crop_to_bbox` | Polygon offset `bbox.min − 10` unclamped while crop origin is `max(0, …)` → masks shifted up to 10 px for border-adjacent faces. Silent label corruption. | FIXED@cc890de |
| UB-09 | Major | `unified_training.py:validate` | Per-epoch "inference time" without `torch.cuda.synchronize()` (measures launch, includes loss). Benchmark syncs correctly but discards no warm-up batches. | FIXED@bc01d07 |
| UB-10 | Major | `benchmark_models.py` | Peak-VRAM compared across models at different batch sizes (worsened by UB-05). | FIXED@3bbe8df |
| UB-11 | Major | `unified_training.py`, `benchmark_models.py` | Metric inconsistency: hard IoU vs *soft* Dice (softmax, incl. background, unweighted ragged batches); benchmark `avg_loss` CE-only vs training CE+Dice. | FIXED@3260638 |
| UB-12 | Major | root `config.yaml`, `codes/config.yaml`, `unified_training.py:176` | Config drift: root YAML loaded by nothing; `training.batch_sizes` unconsumed; `CLASS_WEIGHTS` has no producer; scheduler/optimizer hardcoded. | OPEN |
| UB-13 | Minor | `main_pipeline.py:373` | `--epochs 100` explicitly passed is ignored (`if args.epochs != 100`). Use `default=None`. | FIXED@add5229 |
| UB-14 | Minor | `run.sh` + `main_pipeline.py` | Two timestamps per run → duplicate `logs/<ts>` dirs; run.sh's final message points at the wrong one. | FIXED@add5229 |
| UB-15 | Minor | various | Dead code: `train_iou_metric` never updated; `calculate_iou` unused; no-op `load_shared_data`; 11-panels-in-10-axes grid; `normalize_thermal` returns un-normalized image when min==max. | FIXED@5991f60 |
| UB-16 | Method | `transunet.py` | ViT-B scale (~100M params) trained from scratch on ~1.8k images, plain Adam, no warmup/wd/dropout — guarantees transformer underperformance; invalidates fairness claims. | OPEN |
| UB-17 | Method | `swin_unet_plus_plus.py` | Shifted windows **without attention mask**; **no relative position bias & no positional embedding** (window attention permutation-invariant; deepest-stage shift a no-op); CNN decoder; no deep supervision; redundant third nested column; final 4× bilinear jump from H/4. | OPEN |
| UB-18 | Method | `unified_training.py` | Identical hyperparameters ≠ fair across families; best checkpoint by val *loss* vs headline mIoU; selection and reporting on same folds (no held-out test set). | OPEN |
| UB-19 | Method | `utils.py`, `unified_data.py` | Per-image min–max destroys absolute temperature; `RandomBrightnessContrast` physically dubious on thermal; `np.vectorize` raw→°C is a per-pixel Python loop. | OPEN |
| UB-20 | Method/Env | `unified_data.py`, `setup.py`, requirements | No DataLoader `generator`/`worker_init_fn`; `cudnn.benchmark` contradiction; unpinned `>=` deps (albumentations 2.x breaks `ShiftScaleRotate(value=…)`); `--force-reinstall --no-deps` torch against aging `cu121` index; system-Python installs. *Note: albumentations 2.0.8 rejects `ShiftScaleRotate(value=…)` (Phase 0/D2); pinned `<2.0` in Session 1; API migration lands with T3.4.* | **20a FIXED@c85a5c5** (seeded `generator`+`worker_init_fn`, single cuDNN owner via `deterministic` flag, per-run metadata — T2.5); **20b OPEN** (deps/lockfile/`cu121` index/system-Python → T3.5) |
| UB-21 | Hygiene | `codes/*.py` | Three import styles; no `__init__.py`; model files crash if run directly (relative imports). | OPEN |
| UB-22 | Hygiene | repo root, `.gitignore` | Scratch scripts committed; `inference_comparison.py` orphaned (wire or delete); `codes/tests/*` gitignored while `pytest` is required; `CLAUDE.md` gitignored. | FIXED@d81b7a3 |
| UB-23 | Blocker | `codes/hardware_detector.py` | `detect()` calls `sys.exit(1)` when CUDA is unavailable; contradicts §3.1 CPU-first doctrine; blocks CI E2E. Phase 0 worked around it with a test-only runner that patches detection (removed by the fix: CPU profile behind explicit `UBENCH_ALLOW_CPU=1` opt-in). | FIXED@af14b13 |
| UB-24 | Hygiene | `preprocess_data.py:22`, `unified_data.py:Config.__init__` | `preprocess_all_data()` builds a default `Config()`, whose constructor mkdirs **top-level** `outputs/{models,plots,predictions}` — clutter beside the real `outputs/<run_id>/` dirs on every fresh clone's first run. Found during T1.8's run-dir accounting. Fold into T2.6 or T3.1. | FIXED@3b5d69e |
| UB-25 | Hygiene | `codes/tests/*` subprocess runs | Full-suite runs persist ~10–15 GB of full-state checkpoints across retained pytest temp dirs (3-run default retention); near-full disks turn this into spurious ENOSPC reds. Mitigated by `tmp_path_retention_policy = "failed"` (Session 6); root-cause slimming (e.g., skip per-epoch resume checkpoints under smoke configs) decided in T2.6. | FIXED@0d12ca1 (retain only latest resume checkpoint; UB-06 re-verified) |

---

## 6. Non-Negotiable Engineering Rules

**R1 — Verify empirically.** Every "X works / X is fixed" claim must be backed by a command you ran in this session and its observed output. If execution is impossible, prefix the claim with `UNVERIFIED:`.

**R2 — Test first.** For every ledger item: failing test (red) → minimal fix (green) → refactor → keep the test. A fix without a regression test is not accepted.

**R3 — Minimal, surgical diffs.** Fix the defect; do not reformat files, rename unrelated symbols, or "improve while you're there." Most P0 fixes are < 15 lines.

**R4 — No silent failure paths.** Never widen a `try/except`, add a `.get(key, default)`, or insert a fallback to make an error disappear. Prefer hard lookups that raise, explicit boundary validation, and non-zero exit codes. This codebase is a case study in how defaults and swallowed exceptions convert bugs into lies.

**R5 — One source of truth per contract.** Filenames, config keys, model identifiers: define once, import everywhere. The registry name (`unet`, `transunet`, `swin_unet_plus_plus`) is the canonical `model_key` for **all** file I/O and dict keys; display names ("Swin-UNet++") are for plots and prose only.

**R6 — Conventional commits, one ledger item per commit/PR.** `fix(benchmark): unify checkpoint filename contract (UB-02)`. Update the ledger Status in the same commit.

**R7 — Style & typing.** Python ≥3.10; type hints on all new/modified signatures; `ruff check` clean (config in `pyproject.toml`); Google-style docstrings describing *actual* behavior; absolute imports rooted at `codes.` (after UB-21); no relative imports; no `sys.path` hacks in new code.

**R8 — Security & safety defaults.** `torch.load(..., weights_only=True)` for all weight/checkpoint loading. Validate archive member names in `extract_data.py` before extraction (defense-in-depth). Never execute code from data files.

**R9 — Documentation follows code.** Any behavior change updates README/docstrings/this file in the same PR. Docs must never describe behavior the code does not implement — that is precisely how this repo got here.

**R10 — Honest AI collaboration.** Do not fabricate benchmark numbers, invent training runs, or extrapolate metrics. Truncated runs (`NUM_EPOCHS=1`) produce smoke artifacts, never "results."

---

## 7. Testing Doctrine

### 7.1 Layout & conventions

```
conftest.py                     # root: collect_ignore_glob for legacy ad-hoc debug scripts (removal in T2.6)
codes/tests/
├── __init__.py
├── conftest.py                 # synthetic_dataset fixture (§7.2), collect_ignore_glob, seeding
├── test_pipeline_smoke.py      # E2E gate — THE definition of "the repo works"
├── test_hardware_cpu.py        # UB-23: CPU profile behind UBENCH_ALLOW_CPU=1 opt-in
├── test_filenames.py           # UB-02: train↔benchmark path round-trip
├── test_splits.py              # UB-03/04: fold-count guard, subject exclusivity
├── test_preprocess_offsets.py  # UB-08: border-bbox mask alignment
├── test_failure_honesty.py     # UB-07: injected failure → exit 1, error log, --fail-fast
├── test_resume.py              # UB-06: two-phase --resume, history continues, outputs/latest
├── test_run_identity.py        # UB-13/14: --epochs honored, UBENCH_RUN_ID reuse, precedence
├── test_batch_size_keys.py     # UB-05: hard lookup, per-tier values
├── test_config.py              # UB-12: schema-validated single config (planned, T3.1)
├── test_timing.py              # UB-09: warm-up arithmetic, CUDA-sync gating, validate() drops timing
├── test_metrics.py             # UB-11: hard IoU/Dice == hand-computed, absent-class exclusion, one shared authority
└── test_models_forward.py      # each registered model: (2,1,256,256)→(2,10,256,256) (planned, T3.2)
```

Pytest runner; all tests CPU-only and green in < ~5 min total; `NUM_WORKERS=0` inside tests; seeds fixed in `conftest.py`. Tests are committed (`.gitignore` fix in T0.2). The smoke test runs `codes/main_pipeline.py` as a subprocess with `UBENCH_ALLOW_CPU=1` so hardware detection succeeds on CPU-only machines (UB-23).

### 7.2 Synthetic-dataset fixture (build exactly this)

Session-scoped fixture fabricating a miniature but *structurally faithful* dataset in `tmp_path`, so the full pipeline runs in seconds without the real LFS data:

- **Subjects:** `S1…S5`, 4 frames each → 20 samples (5 groups ⇒ `K_FOLDS=2` splits cleanly; enables subject-exclusivity assertions).
- **Frames:** `data/S{n}/R{n}00{i}.tiff`, 64×64 `uint16`, values in `[29315, 31315]` (→ 20–40 °C via `(raw/100) − 273.15`); filenames must satisfy `get_tiff_path`'s digit extraction (`R{digits}.tiff`).
- **Polygons:** `data/S{n}_polygonal_masks.json` keyed by bare sample id (`"R10001"`; loader prefixes to `S1/R10001`). Region keys must match `Config.REGION_NAMES[1:]` **exactly** (Portuguese: e.g. `"Nariz"`, `"Olho esquerdo"`, `"Testa"`). Two or three small rectangles per frame suffice.
- **BBoxes:** `data/S{n}_bounding_boxes.csv` with `ID,min_x,min_y,max_x,max_y`. **At least one bbox with `min_x = 3`** (closer to the border than the 10 px padding) — the UB-08 tripwire.
- Fixture yields the dataset root; tests `monkeypatch.chdir` there and point `Config` paths at it.

### 7.3 The smoke test (the merge gate)

```python
def test_full_pipeline_smoke(synthetic_dataset, monkeypatch):
    monkeypatch.setenv("LIMIT_SAMPLES", "20")
    monkeypatch.setenv("NUM_EPOCHS", "1")
    monkeypatch.setenv("K_FOLDS", "2")
    monkeypatch.setenv("NUM_WORKERS", "0")
    rc = run_pipeline_subprocess(models=["unet", "transunet", "swin"])
    assert rc == 0
    df = read_latest("outputs/*/benchmark_comparison.csv")
    assert set(df["Model"]) == {"U-Net", "TransUNet", "Swin-UNet++"}   # guards UB-01/02/03/05/07
```

Write it **before** Phase 1, watch it fail for the right reason (UB-01's FileNotFoundError), and let it go green only as the P0 fixes land. It runs in CI forever after.

### 7.4 CI (GitHub Actions, CPU-only)

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements/requirements.txt ruff pytest
      - run: ruff check codes/
      - run: pytest codes/tests -x -q
```

---

## 8. ML & Benchmark Methodology Standards

A benchmark that is *reproducible but methodologically unfair* is worse than none — it launders noise into conclusions.

**M1 — Splits.** Subject-level grouping is mandatory (frames of one person are near-duplicates). Keep `GroupKFold(groups=dataset)`; guard `n_splits = min(K_FOLDS, n_groups)` with a loud warning; error if `n_groups < 2`; rename the concept in docs to *leave-subjects-out CV*. **Shipped (T2.4):** `config.test_subjects` (default empty, opt-in; `TEST_SUBJECTS` env override) are excluded from all CV folds via `load_split_metadata` and evaluated separately; each of the K fold-models is scored on the held-out set and reported as mean ± std (no model is *selected* on test — nothing to select on). Fold metrics select, test metrics headline; never tune on test subjects. An absent test-subject id, or a holdout leaving <2 CV subjects, is a hard error.

**M2 — Metrics.** One definition per metric, shared by trainer and benchmark, via `torchmetrics` on **argmax**: `JaccardIndex(average='none')` + hard `Dice/F1`, macro-averaged excluding absent classes; background reported separately. "Loss" everywhere = the training criterion (CE+Dice). Persist per-class values; aggregate across folds as mean ± std.

**M3 — Timing.** GPU latency = `torch.cuda.synchronize()` before and after; discard ≥5 warm-up batches; report mean ± std per image with batch size, dtype, GPU, torch version attached. Never compare peak VRAM across models at different batch sizes — fixed probe batch size or bytes/sample.

**M4 — Fairness.** Per-family recipes, fully disclosed — CNN: Adam/AdamW 1e-4 + plateau; Transformers: **AdamW** (wd ≈ 0.05) + linear warmup (~5% steps) + cosine, grad-clip 1.0. Select best checkpoint by **val mIoU** (the headline metric). Equal epoch/step budget per model, disclosed.

**M5 — Pretraining policy.** ViT-B from scratch on ~1.8k images is known-degenerate. Prefer pretrained encoders — TransUNet via `timm` R50+ViT-B/16; Swin via `timm`/MONAI (`SwinUNETR`) — adapting 1-channel input by **summing pretrained RGB stem kernels**. Do not re-implement attention that timm/MONAI already provide correctly (UB-17 shows why). From-scratch variants stay only if the study is explicitly scoped as small-data-from-scratch.

**M6 — Reproducibility.** Single `seed_everything` owner; DataLoaders get `generator=torch.Generator().manual_seed(seed)` + `worker_init_fn` seeding numpy/random per worker; one owner for `cudnn.benchmark` (hardware profile), documented determinism trade-off. Lockfile (`uv lock` / `pip-compile`): CPU lock for CI, CUDA lock for training. Each run dir logs git SHA + dirty flag, lockfile hash, torch/CUDA versions, GPU name, effective config dump.

**M7 — Thermal-domain preprocessing.** Config switch `normalization: {per_image_minmax | fixed_range}` with `fixed_range: [20.0, 40.0] °C` recommended (preserves absolute temperature — the modality's core signal, which `get_stats_info` itself reports). Replace `np.vectorize` with `(raw.astype(np.float32) / 100.0) - 273.15`. Replace `RandomBrightnessContrast` with physically plausible additive offset (±0.5 °C sensor drift) + Gaussian noise; keep flip/affine (migrate deprecated `ShiftScaleRotate` → `A.Affine` per pinned version).

**M8 — Statistical claims.** "Outperforms" requires mean ± std across folds on held-out test subjects **and** a paired Wilcoxon signed-rank across folds (report p). Otherwise: "comparable within noise." The report generator computes this, not prose.

**M9 — Honest reporting.** Every table/plot states dataset+split protocol, #params, pretraining status, budget, batch size, hardware, torch version. Smoke artifacts are labeled and never mixed into `benchmark_comparison.csv` history.

---

## 9. Phased Task Plan (execution order, acceptance criteria)

Strict order inside each phase; 0→1→2 sequential, 3 may interleave after 1. One ledger item per PR (R6). **AC = acceptance criterion demonstrated by a run.**

### Phase 0 — Safety net (before touching any bug)

- [x] **T0.1** `codes/tests/` skeleton + synthetic fixture (§7.2) + smoke test (§7.3) as a *failing* test. AC met: smoke test fails with UB-01's FileNotFoundError, tracked as `xfail(strict=True)` so the suite stays green.
- [x] **T0.2** Un-ignore `codes/tests/*` and `CLAUDE.md`; add `codes/__init__.py`, `codes/tests/__init__.py`; CI workflow (§7.4) + `pyproject.toml` with ruff config. AC: suite green; smoke test `xfail(strict=True)` enforcing the current frontier defect.

### Phase 1 — Make `./run.sh` work (P0)

- [x] **T1.1 (UB-01)** Pipeline invokes `preprocess_all_data()` when `metadata.csv` absent; `--force-preprocess`; README gains the step. AC met: smoke trains all 3 models × 2 folds to epoch 1 on the synthetic fixture; frontier advanced to UB-02.
- [x] **T1.2 (UB-02)** `codes/naming.py` → `checkpoint_path(output_dir, model_key, fold, kind)`; trainer and benchmark import it; benchmark receives `model_key` alongside display name. AC met: `test_filenames.py` green; smoke passes with all 3 models in the CSV; xfail marker removed.
- [x] **T1.3 (UB-05)** Canonical registry keys in `hardware_detector`; replace `.get(k, 8)` with `[k]`. AC met: `test_batch_size_keys.py` green — simulated 6 GB tier yields `swin_unet_plus_plus=6`, <5.5 GB tier 3, per-tier key parity with the model registry enforced; smoke exercises the hard lookup on the CPU profile.
- [x] **T1.4 (UB-03/04)** `effective_k = min(K, n_groups)` + warning; error if `<2`; README rewritten to leave-subjects-out; impossible example output deleted (verified already absent). AC met: `test_splits.py` green — 2–3 subjects → runs with reduced K + warning; 1 subject → clear actionable error; no subject in both train and val of any fold; end-to-end 3-subject pipeline run completes at effective K=3.
- [x] **T1.5 (UB-08)** Clamped origin computed once; `crop_to_bbox` returns `(img, origin)`; mask offset uses it. AC met: `test_preprocess_offsets.py` — red on unfixed code (border-bbox centroid 28 px off: (89.5, 89.0) vs (61.5, 89.0)), green after; control bbox (min_x=10) aligned before and after.
- [x] **T1.6 (UB-07)** Failure registry + end-of-run summary; non-zero exit on any failure; `--fail-fast`; `Pipeline()` moved inside `main()`'s try. AC met: `test_failure_honesty.py` — injected failure → exit 1 + failure summary naming model+fold + `error_log_*.txt`; `--fail-fast` aborts before fold 2; constructor crash reaches the error-log writer; smoke asserts rc==0 and no failure summary.
- [x] **T1.7 (UB-06)** `--resume <run_id>` reuses dirs/checkpoints; `outputs/latest` symlink. AC met: `test_resume.py` — epoch-1 run + `--resume` at `NUM_EPOCHS=2` continues to metric history length 2 in both folds' metrics JSON, no second run dir, `outputs/latest` resolves to the run; unknown run id fails actionably (deterministic epoch-boundary variant satisfies the kill-after-epoch-1 AC). En route: `validate()` now returns plain floats — numpy scalars broke the checkpoint's `weights_only=True` load.
- [x] **T1.8 (UB-13/14)** `--epochs default=None`; `run.sh` exports `UBENCH_RUN_ID`, Python reuses it. AC met: `test_run_identity.py` — `UBENCH_RUN_ID=fixed-test-id` names the run dirs with no timestamped dir minted, `--resume` precedence over the env id verified, `--epochs 2` honored e2e with `NUM_EPOCHS` unset, `apply_epochs_override(100)` exports the exact value the old guard dropped; real `./run.sh` invocation produced a single `logs/<ts>` (Session 5 transcript).

### Phase 2 — Trustworthy numbers (P1)

- [x] **T2.1 (UB-09)** Remove per-epoch timing from `validate()`; benchmark adds warm-up discard. AC met: `test_timing.py` — warm-up arithmetic at n_batches∈{1,3,20} via the `timed_inference` helper, `synchronize()` called exactly twice on a CUDA device and zero on CPU (lightweight fakes, no GPU), `validate()` returns a 3-tuple with no `inference_times`; benchmark report/console disclose warm-up count + batch/dtype/device (M9); full suite 52 green, smoke green.
- [x] **T2.2 (UB-11)** Shared `codes/metrics.py` (M2) used by both; benchmark loss = `CombinedLoss`. AC met: `test_metrics.py` — hard IoU/Dice equal hand-computed values on a crafted logits/target pair, class absent from the target excluded from the macro, background reported separately, trainer and benchmark resolve the same `SegmentationMetrics`, accumulation batch-invariant. Derived from `torchmetrics.ConfusionMatrix` (pinned 1.9.0 has no Dice class). Numbers move soft→hard by design; smoke asserts structure and stays green.
- [x] **T2.3 (UB-10)** Separate memory probe at fixed batch size. AC met: `probe_peak_memory` runs one inference forward on a synthetic `(4,1,H,W)` batch shared by all models (`reset_peak_memory_stats`→forward→`max_memory_allocated`); `test_memory.py` — fixed batch regardless of the model's training batch, correct synthetic shape, CPU→`None`/"n/a (CPU)" (no fabricated 0, R10), monkeypatched CUDA calls stats once each; report/CSV column "VRAM @ batch=4 (fixed, inference)"; report/plots guard the all-NaN CPU case.
- [x] **T2.4 (M1)** `test_subjects` excluded from CV; benchmark evaluates test set; per-fold-on-test reporting documented. AC met: `test_test_subjects.py` — two-directional exclusivity (no fold sees a test subject; held-out loader is exactly the test subjects), <2-CV-subject holdout raises, absent id raises, empty default unchanged; smoke reserves S5 and asserts CV+TEST columns; report has CV and TEST sections. Default empty (opt-in); `TEST_SUBJECTS` env override; each fold-model scored on the held-out set, mean ± std, none selected on test.
- [x] **T2.5 (UB-20a/M6)** Seeded `generator` + `worker_init_fn`; cudnn ownership; run-metadata JSON per run. AC met: `test_determinism.py` — direct unit test of `seed_worker` (deterministic per worker id, distinct across ids — the guarantee a 0-worker test can't give), a `NUM_WORKERS=2` spawn test showing the stream is reproducible with an explicit seeded generator and fragile without, the real loader factory wires `generator`+`seed_worker`, and `collect_run_metadata` records provenance. Single cuDNN owner via a `deterministic` config flag (default true; `hardware_detector` no longer touches cuDNN). `run_metadata.json` per run (git SHA/dirty, torch/CUDA, GPU, seed, deterministic, effective config; lockfile hash TODO@T3.5). Modern torch already auto-seeds a worker's numpy from the base seed, so the *seeded generator* — not merely `seed_worker` — is what pins the stream independent of global RNG.
- [x] **T2.6 (UB-15/22, +UB-24/UB-25)** Delete dead code & scratch scripts; wire-or-delete `inference_comparison.py`; fix grid; min==max → zeros. AC met across four atomic commits: UB-22 (delete `inference_comparison.py` + 5 inspection/scratch scripts + 7 gitignored NaN-debug tests + both `collect_ignore_glob`), UB-15 (`calculate_iou`/`train_iou_metric`/`load_shared_data` removed; `normalize_thermal(min==max)→zeros` with `test_utils.py`; `visualize_predictions` grid sized to fit all panels), UB-24 (`_create_output_dirs` no top-level subdirs; smoke asserts none), UB-25 (`_CHECKPOINT_KEEP_LAST=1`; `test_resume` re-verified). ruff + grep prove no dangling references; `verify_regex` deleted (its regex is covered by the pipeline/smoke).

### Phase 3 — Credible science (P2)

- [ ] **T3.1 (UB-12)** One config: delete root `config.yaml`; pydantic schema over `codes/config.yaml` (fail on unknown keys); wire scheduler/optimizer/augmentation/`class_weights` (from train-fold frequencies) or delete keys. AC: bogus key raises; every remaining key demonstrably alters behavior.
- [ ] **T3.2 (UB-16/17, M5)** Pretrained encoders via timm/MONAI; 1-channel stem adaptation; register `transunet_pretrained`, `swinunetr`. AC: forward-shape tests; param counts logged; smoke covers new keys.
- [ ] **T3.3 (UB-18, M4)** Per-family optimizer/scheduler recipes from config; selection by val mIoU. AC: config toggles recipe; transformer logs show warmup.
- [ ] **T3.4 (UB-19, M7)** Normalization switch + physical augmentation + vectorized conversion. AC: unit tests both modes; conversion ≥100× faster on 640×480 (generous timed bound).
- [ ] **T3.5 (UB-20b)** Lockfiles (`uv`); `setup.py` creates venv, drops `--no-deps`, resolves current CUDA index. AC: clean-container install from lockfile → smoke green.
- [ ] **T3.6 (UB-21)** Absolute imports + `__init__.py`; models runnable as `python -m codes.<model>` self-test. AC: `python -m codes.transunet` prints forward shape, no ImportError.
- [ ] **T3.7 (R9)** README reconciled to code (protocol, config table, quickstart incl. preprocessing, hardware notes). AC: every README command executed verbatim in a fresh venv.

### Phase 4 — Enhancements (post-green only)

- [ ] **T4.1 (M8)** Wilcoxon paired test in report generator. — [ ] **T4.2** TensorBoard logging. — [ ] **T4.3** `pyproject.toml` packaging + `ubench train|benchmark` entry points. — [ ] **T4.4** Modern AMP (`torch.amp.GradScaler('cuda')`, bf16 on Ampere+). — [ ] **T4.5** ONNX export + deployment-latency table. — [ ] **T4.6** mypy in CI.

---

## 10. Verification Gates & Definition of Done

**Session Entry Protocol:** every session begins by verifying the predecessor's claimed end-state from the working tree — run the tests it claims green, check the ledger rows it claims flipped, confirm the key files it claims exist — before any new work. A failed check halts the session: report the gap, do not build on the unverified base, and do not "quickly fix" it in passing. This exists because trusting reports over trees is how drift starts.

**Per task:** its new test passes; full `pytest codes/tests -x -q` green; `ruff check codes/` clean; ledger row → `FIXED@<sha>`; docs updated if behavior changed; diff reviewed for scope creep (R3).

**Per phase:** smoke test green; phase-specific integration criteria demonstrated with a pasted run transcript (commands + output tail) in the PR description.

**Project DoD:** fresh clone → `./run.sh` → 3-model `benchmark_comparison.csv` with zero manual steps; CI green; every README claim executable; benchmark reports test-subject metrics with cross-fold variance and disclosed budgets (M9).

---

## 11. Anti-Patterns — Forbidden Actions

Do **not**: mark tasks done without an executed passing test (R1); weaken or delete a failing test to go green; widen exception handling or add dict-defaults to silence errors (R4); derive storage filenames from display strings (UB-02's root cause); edit the dead root `config.yaml` expecting behavior change, or re-introduce a second config file (UB-12); run full-scale training in dev/CI; commit `data/`, `outputs/`, `logs/`, weights, or lockfile-violating deps; reformat files wholesale; fabricate, extrapolate, or embellish metrics/runs (R10); leave README describing unimplemented behavior (R9); hand-roll attention mechanisms timm/MONAI already provide (M5); compare VRAM/latency across unequal batch sizes or without synchronize+warm-up (M3); tune anything on test subjects (M1).

---

## 12. Quick Command Reference

```bash
# Dev loop
source .venv/bin/activate
ruff check codes/ && pytest codes/tests -x -q                    # gate
LIMIT_SAMPLES=20 NUM_EPOCHS=1 K_FOLDS=2 NUM_WORKERS=0 \
  python codes/main_pipeline.py --models unet --skip-benchmark   # fast pipeline probe
pytest codes/tests/test_pipeline_smoke.py -x -q                  # E2E gate

# Focused verifications
python - <<'PY'                                                  # filename contract (UB-02, post-T1.2)
from pathlib import Path
from codes.naming import checkpoint_path
print(checkpoint_path(Path("outputs/run"), "swin_unet_plus_plus", fold=1, kind="best"))
PY
python -m codes.transunet                                        # import sanity (UB-21, post-T3.6)

# Hygiene
git status --porcelain | grep -E 'data/|outputs/|logs/|\.pth' && echo "DO NOT COMMIT" || echo "clean"
```

*This is a living document (R9): when reality and this file disagree, fix the code or fix the file — in the same PR.*

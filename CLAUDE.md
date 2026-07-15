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

The repository **does not work end-to-end**: a fresh clone running `./run.sh` now preprocesses and trains (UB-01 fixed in T1.1), but the final benchmark silently drops 2 of the 3 models (filename contract mismatch — UB-02) and still exits 0. Every defect is catalogued in the ledger (§5) with verified locations; treat it as ground truth, do not re-litigate it, and do re-verify each item with a test as you fix it. Work proceeds in phases (§9): make it *run* → make the *numbers trustworthy* → make the *science credible* → enhance.

**Prime directive:** *No task is "done" until its acceptance test passes in a real execution.* Reading code is not verification. If you cannot run something, say so explicitly and mark the task blocked.

There is currently **no test suite, linter, or CI** (`codes/tests/` holds ad-hoc debug scripts and is gitignored). Phase 0 (§9) builds the safety net before any bug is touched; after T0.1 lands, the smoke test — not "running the pipeline" — is how changes are validated.

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
./run.sh                                  # Linux/Mac   (⚠ benchmark silently drops 2 of 3 models — UB-02)
run.bat                                   # Windows
./run.sh --skip-extract --skip-setup --models unet --epochs 10
./run.sh --models transunet swin          # model choices: unet, transunet, swin
./run.sh --skip-benchmark
```

`run.sh`/`run.bat` handle `--skip-extract` and `--skip-setup` themselves and forward everything else (`--models`, `--epochs`, `--skip-benchmark`) to `codes/main_pipeline.py`. When debugging a single stage, call the `codes/*.py` script directly instead of going through `run.sh`. Note `--epochs 100` passed explicitly is currently ignored (UB-13).

### 3.3 Running stages directly

```bash
python codes/extract_data.py          # unzip requirements/*.zip -> data/ (needs Git LFS)
python codes/setup.py                 # env validation + CUDA torch install (GPU boxes)
python codes/preprocess_data.py       # writes data/processed/ + metadata.csv; the pipeline runs this
                                      # automatically when metadata.csv is absent (T1.1);
                                      # --force-preprocess on main_pipeline.py rebuilds unconditionally
python codes/main_pipeline.py --models unet --epochs 10 --skip-benchmark
python codes/inference_comparison.py  # standalone side-by-side prediction viewer for trained
                                      # checkpoints; not wired into the pipeline (UB-22: wire or delete)
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

**Data-flow contract:** `extract_data.py` → `preprocess_data.py` (produces `data/processed/{images/*.npy, masks/*.png, metadata.csv}`) → `create_single_fold_loader` (lazy, one fold at a time; `create_kfold_data_loaders` exists but the pipeline uses the single-fold variant to avoid worker/semaphore leaks) → `UnifiedTrainer` (saves `outputs/<run>/models/best_<key>_model.pth` + full `checkpoints/`) → `run_benchmark` (loads those weights, writes `benchmark_comparison.csv`, plots, report).

### 4.2 Configuration — where truth actually lives

`codes/unified_data.py:Config` loads **`codes/config.yaml`** (`Path(__file__).parent / "config.yaml"`): image size, `num_classes`, learning rate, epochs, `k_folds`, seed, region names. The root-level `config.yaml` — despite documenting loss weights, optimizer, scheduler, augmentation, and hardware sections — **is loaded by nothing** (UB-12). Do not "fix" behavior by editing the root file; it is dead until T3.1 consolidates to a single validated schema. Within the loaded file, `training.batch_sizes` is also unconsumed, and the `CLASS_WEIGHTS` branch in `unified_training.py` has no producer. `--epochs` on the CLI overrides config via the `NUM_EPOCHS` env var (`main_pipeline.py:main`), modulo UB-13.

### 4.3 Hardware auto-scaling

`codes/hardware_detector.py:detect_and_optimize` profiles the GPU at pipeline start and picks per-model batch sizes, worker count, and AMP strategy (GTX-class cards and CPUs get AMP disabled; the 6 GB GTX 1660 Ti is the baseline tier). ⚠ UB-05: the profile keys are `{unet, transunet, swin}` but training looks up `swin_unet_plus_plus` and silently falls back to 8 — Swin's VRAM-aware value is ignored during training until T1.3. Do not disable mixed precision for memory-constrained GPUs.

### 4.4 Multi-dataset discovery and ID namespacing

`codes/unified_data.py:MultiDirectoryDataLoader` scans `data/` for `S1`–`S10` directories and merges them, prefixing sample IDs by source dataset (e.g. `S1/R11104`) to keep them unique. Each dataset's annotations can live in either of two layouts — both are checked automatically: `data/S1_polygonal_masks.json` + `data/S1_bounding_boxes.csv` (root-level, recommended), or `data/S1/polygonal_masks.json` + `data/S1/bounding_boxes.csv` (inside the dataset dir). TIFF resolution goes through `get_tiff_path`, which extracts digits from the sample ID and expects `R{digits}.tiff`.

### 4.5 Splits — what the code actually does

Splitting is `GroupKFold(groups=df['dataset'])`: **whole subject directories are held out per fold** (leave-subjects-out CV). This is the methodologically correct choice for face data (frames of one person are near-duplicates), but it is *not* the stratified per-dataset split older docs described, and it **crashes** whenever the number of `S*` directories is smaller than `k_folds` (default 5) — UB-03/04. Fold counts, guards, and a held-out test-subject set are handled in T1.4 and T2.4.

### 4.6 Shared training loop, model-specific architectures

All three models train through the same `UnifiedTrainer` (`codes/unified_training.py`): identical augmentations, combined Cross-Entropy + Dice loss (`CombinedLoss`), `ReduceLROnPlateau`, gradient clipping, NaN protection, per-epoch checkpointing. This shared loop is what makes cross-model numbers comparable — **when adding a new model architecture**: implement it as an `nn.Module` decorated with `@register_model("<key>")` in a new `codes/<model>.py`, import the module in `main_pipeline.py` so registration runs, add its batch-size key to `hardware_detector.py`, and integrate through `UnifiedTrainer` rather than writing a parallel training loop. Be aware that "identical hyperparameters" is *not* automatically fair across architecture families (UB-18, M4) — per-family recipes arrive in T3.3.

### 4.7 Timestamped run isolation

Every `main_pipeline.py` invocation creates a fresh timestamp and writes all artifacts under `outputs/<timestamp>/` (models, plots, checkpoints) and `logs/<timestamp>/` (per-stage logs, metrics JSON, benchmark report). Nothing from a previous run is overwritten. ⚠ Consequences until fixed: auto-resume can never find prior checkpoints (UB-06), and `run.sh` mints a *second* timestamp of its own so each run produces two log dirs and the shell's final message points at the wrong one (UB-14).

### 4.8 Dev/debug scripts

`codes/tests/` currently holds ad-hoc debugging scripts (NaN-in-AMP diagnostics etc.) and is excluded via `.gitignore` — it is not a real test suite. `codes/inspect_ids.py`, `codes/verify_regex.py`, `codes/print_all_missing.py`, `codes/test_edge_cases.py` are one-off inspection utilities outside the pipeline's execution path. Phase 0 (T0.2) inverts this: real tests get committed, scratch scripts get deleted or ported into tests (UB-22).

---

## 5. Known-Defect Ledger (verified — single source of truth)

Update the **Status** column as work lands (`OPEN → IN-PROGRESS → FIXED@<sha>`). Never delete rows; history matters.

| ID | Sev | Location | Defect (verified behavior) | Status |
|----|-----|----------|----------------------------|--------|
| UB-01 | Blocker | `run.sh`, `main_pipeline.py` | `preprocess_data.py` is never invoked by any entry point; loaders require `data/processed/metadata.csv` → fresh clone fails 15/15 train attempts. README omits the step. | FIXED@03e6dbb |
| UB-02 | Blocker | `benchmark_models.py:~443-452` vs `main_pipeline.py:train_model` | Filename contract mismatch: training saves `best_unet_fold_N_model.pth`, `best_swin_unet_plus_plus_fold_N_model.pth` (registry names); benchmark searches `best_u_net_…`/`best_swin_unetplusplus_…` (display names via `_safe_filename`). Only TransUNet matches → comparison silently contains 1 of 3 models. | OPEN |
| UB-03 | Blocker | `unified_data.py:~427,~540` | `GroupKFold(n_splits=K_FOLDS)` with `groups=df['dataset']` raises `ValueError` whenever #datasets < K (default 5). | OPEN |
| UB-04 | Blocker/Docs | `unified_data.py`, `README` | Split semantics contradict docs: leave-subjects-out in code vs stratified-per-dataset in README, whose example fold counts GroupKFold cannot produce. Fix the docs, not the split. | OPEN |
| UB-05 | Blocker | `main_pipeline.py:121` + `hardware_detector.py` | Batch-size keys `{unet, transunet, swin}` vs lookup `swin_unet_plus_plus` → `.get(key, 8)` silently returns 8. OOM risk at the advertised 6 GB minimum; waste on large GPUs. | OPEN |
| UB-06 | Major | `main_pipeline.py`, `unified_training.py` | Auto-resume dead across restarts: checkpoints under `outputs/<timestamp>/checkpoints`, new timestamp per invocation → `_find_latest_checkpoint()` scans empty dir. | OPEN |
| UB-07 | Major | `main_pipeline.py` | Failures swallowed: per-model/fold `try/except` prints one line, continues; SUCCESS banner + exit 0 possible with zero trained models; `main()`'s error-log writer unreachable. | OPEN |
| UB-08 | Major | `preprocess_data.py:66-67` vs `crop_to_bbox` | Polygon offset `bbox.min − 10` unclamped while crop origin is `max(0, …)` → masks shifted up to 10 px for border-adjacent faces. Silent label corruption. | OPEN |
| UB-09 | Major | `unified_training.py:validate` | Per-epoch "inference time" without `torch.cuda.synchronize()` (measures launch, includes loss). Benchmark syncs correctly but discards no warm-up batches. | OPEN |
| UB-10 | Major | `benchmark_models.py` | Peak-VRAM compared across models at different batch sizes (worsened by UB-05). | OPEN |
| UB-11 | Major | `unified_training.py`, `benchmark_models.py` | Metric inconsistency: hard IoU vs *soft* Dice (softmax, incl. background, unweighted ragged batches); benchmark `avg_loss` CE-only vs training CE+Dice. | OPEN |
| UB-12 | Major | root `config.yaml`, `codes/config.yaml`, `unified_training.py:176` | Config drift: root YAML loaded by nothing; `training.batch_sizes` unconsumed; `CLASS_WEIGHTS` has no producer; scheduler/optimizer hardcoded. | OPEN |
| UB-13 | Minor | `main_pipeline.py:373` | `--epochs 100` explicitly passed is ignored (`if args.epochs != 100`). Use `default=None`. | OPEN |
| UB-14 | Minor | `run.sh` + `main_pipeline.py` | Two timestamps per run → duplicate `logs/<ts>` dirs; run.sh's final message points at the wrong one. | OPEN |
| UB-15 | Minor | various | Dead code: `train_iou_metric` never updated; `calculate_iou` unused; no-op `load_shared_data`; 11-panels-in-10-axes grid; `normalize_thermal` returns un-normalized image when min==max. | OPEN |
| UB-16 | Method | `transunet.py` | ViT-B scale (~100M params) trained from scratch on ~1.8k images, plain Adam, no warmup/wd/dropout — guarantees transformer underperformance; invalidates fairness claims. | OPEN |
| UB-17 | Method | `swin_unet_plus_plus.py` | Shifted windows **without attention mask**; **no relative position bias & no positional embedding** (window attention permutation-invariant; deepest-stage shift a no-op); CNN decoder; no deep supervision; redundant third nested column; final 4× bilinear jump from H/4. | OPEN |
| UB-18 | Method | `unified_training.py` | Identical hyperparameters ≠ fair across families; best checkpoint by val *loss* vs headline mIoU; selection and reporting on same folds (no held-out test set). | OPEN |
| UB-19 | Method | `utils.py`, `unified_data.py` | Per-image min–max destroys absolute temperature; `RandomBrightnessContrast` physically dubious on thermal; `np.vectorize` raw→°C is a per-pixel Python loop. | OPEN |
| UB-20 | Method/Env | `unified_data.py`, `setup.py`, requirements | No DataLoader `generator`/`worker_init_fn`; `cudnn.benchmark` contradiction; unpinned `>=` deps (albumentations 2.x breaks `ShiftScaleRotate(value=…)`); `--force-reinstall --no-deps` torch against aging `cu121` index; system-Python installs. *Note: albumentations 2.0.8 rejects `ShiftScaleRotate(value=…)` (Phase 0/D2); pinned `<2.0` in Session 1; API migration lands with T3.4.* | OPEN |
| UB-21 | Hygiene | `codes/*.py` | Three import styles; no `__init__.py`; model files crash if run directly (relative imports). | OPEN |
| UB-22 | Hygiene | repo root, `.gitignore` | Scratch scripts committed; `inference_comparison.py` orphaned (wire or delete); `codes/tests/*` gitignored while `pytest` is required; `CLAUDE.md` gitignored. | OPEN |
| UB-23 | Blocker | `codes/hardware_detector.py` | `detect()` calls `sys.exit(1)` when CUDA is unavailable; contradicts §3.1 CPU-first doctrine; blocks CI E2E. Phase 0 worked around it with a test-only runner that patches detection (removed by the fix: CPU profile behind explicit `UBENCH_ALLOW_CPU=1` opt-in). | FIXED@af14b13 |

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
├── test_batch_size_keys.py     # UB-05: hard lookup, per-tier values
├── test_config.py              # UB-12: schema-validated single config
├── test_metrics.py             # UB-11: hard-Dice == manual computation
└── test_models_forward.py      # each registered model: (2,1,256,256)→(2,10,256,256)
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

**M1 — Splits.** Subject-level grouping is mandatory (frames of one person are near-duplicates). Keep `GroupKFold(groups=dataset)`; guard `n_splits = min(K_FOLDS, n_groups)` with a loud warning; error if `n_groups < 2`; rename the concept in docs to *leave-subjects-out CV*. Reserve **held-out test subjects** (e.g. S9–S10) excluded from all CV: fold metrics select, test metrics headline. Never tune on test subjects.

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
- [ ] **T1.2 (UB-02)** `codes/naming.py` → `checkpoint_path(model_key, fold, kind)`; trainer and benchmark import it; benchmark receives `model_key` alongside display name. AC: `test_filenames.py` round-trip; smoke's 3-row assertion passes.
- [ ] **T1.3 (UB-05)** Canonical registry keys in `hardware_detector`; replace `.get(k, 8)` with `[k]`. AC: `test_batch_size_keys.py`; simulated 6 GB profile logs swin batch 6.
- [ ] **T1.4 (UB-03/04)** `effective_k = min(K, n_groups)` + warning; error if `<2`; README rewritten to leave-subjects-out; impossible example output deleted. AC: `test_splits.py` — runs with 1–3 subject dirs; no subject in both train and val of any fold.
- [ ] **T1.5 (UB-08)** Clamped origin computed once; `crop_to_bbox` returns `(img, origin)`; mask offset uses it. AC: `test_preprocess_offsets.py` — border-bbox mask centroid within 1 px of expectation.
- [ ] **T1.6 (UB-07)** Failure registry + end-of-run summary; non-zero exit on any failure; `--fail-fast`; `Pipeline()` moved inside `main()`'s try. AC: injected failure → exit 1 + `error_log_*.txt`; smoke asserts rc==0 only when all models trained.
- [ ] **T1.7 (UB-06)** `--resume <run_id>` reuses dirs/checkpoints; `outputs/latest` symlink. AC: kill after epoch 1 of 2, resume completes epoch 2, metric history length == 2.
- [ ] **T1.8 (UB-13/14)** `--epochs default=None`; `run.sh` exports `UBENCH_RUN_ID`, Python reuses it. AC: one `logs/<ts>` per run; `--epochs 100` honored.

### Phase 2 — Trustworthy numbers (P1)

- [ ] **T2.1 (UB-09)** Remove per-epoch timing from `validate()`; benchmark adds warm-up discard. AC: monkeypatched sync assertions; report shows "warm-up=5".
- [ ] **T2.2 (UB-11)** Shared `codes/metrics.py` (M2) used by both; benchmark loss = `CombinedLoss`. AC: hard-Dice equals hand-computed value on a crafted tensor.
- [ ] **T2.3 (UB-10)** Separate memory probe at fixed batch size. AC: report labels "VRAM @ batch=4 (fixed)".
- [ ] **T2.4 (M1)** `test_subjects: [S9, S10]` excluded from CV; benchmark evaluates test set; document weight/ensemble choice. AC: exclusion proven by test; report has CV and TEST sections.
- [ ] **T2.5 (UB-20a/M6)** Seeded `generator` + `worker_init_fn`; cudnn ownership; run-metadata JSON per run. AC: two identical seeded 1-epoch CPU runs → identical loss curves.
- [ ] **T2.6 (UB-15/22)** Delete dead code & scratch scripts (port `verify_regex` into a test); wire-or-delete `inference_comparison.py`; fix 2×6 grid; min==max → zeros. AC: ruff + grep prove no references remain.

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
from codes.naming import checkpoint_path
print(checkpoint_path("swin_unet_plus_plus", fold=1, kind="best"))
PY
python -m codes.transunet                                        # import sanity (UB-21, post-T3.6)

# Hygiene
git status --porcelain | grep -E 'data/|outputs/|logs/|\.pth' && echo "DO NOT COMMIT" || echo "clean"
```

*This is a living document (R9): when reality and this file disagree, fix the code or fix the file — in the same PR.*

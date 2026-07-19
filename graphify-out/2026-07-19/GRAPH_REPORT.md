# Graph Report - UBench  (2026-07-18)

## Corpus Check
- 78 files · ~76,074 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1175 nodes · 1846 edges · 77 communities (69 shown, 8 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `74771c82`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Benchmark & Orchestration Layer
- Model Registry & Swin-UNet++
- Hardware Detection & CPU Mode
- Environment Setup (setup.py)
- Training Loop Internals
- Data Extraction & LFS
- Graphify Skill Docs
- Engineering Guide & Preprocessing
- TransUNet Architecture
- test_preprocess_offsets.py
- NaN Debug Scripts (Ad-hoc)
- Pipeline Stages (main_pipeline)
- Loss Functions (Dice/Combined)
- Multi-Dataset Data Loading
- ModelBenchmark Runner
- E2E Smoke Test Gate
- Synthetic Test Fixture
- CI & Phase 0 Kickoff
- Shell Entry Points
- Active Config (codes/config.yaml)
- Dead Root Config (UB-12)
- Watch Folder Feature
- FalkorDB Export
- Neo4j Export
- ID Inspection Utility
- Missing Data Printer
- Edge Case Tests
- Regex Verification Utility
- Root Conftest
- CLAUDE.md
- UBench — Thermal Face Detection Benchmark Pipeline
- What You Must Do When Invoked
- test_suite.py
- hardware_detector.py
- test_batch_size_keys.py
- swin_unet_plus_plus.py
- DoubleConv
- Config
- unified_data.py
- Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session
- graphify reference: extra exports and benchmark
- Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01)
- HardwareProfile
- Session 6 — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric definition — Phase-2 opener
- ThermalFaceDataset
- Session 0 — Build the safety net (Phase 0 of CLAUDE.md)
- graphify reference: query, path, explain
- DiceLoss
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- Session 6 (final) — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric authority — Phase-2 opener
- Phase-1 real-data validation checklist (GPU box)
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- Session 3 — T1.3 (UB-05) + T1.4 (UB-03/04): batch-size contract & fold-count guard
- graphify
- extraction-spec.md
- detect_and_optimize
- UnifiedTrainer
- create_test_loader
- Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep
- Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep
- SegmentationMetrics
- model_registry.py
- SwinUNetPlusPlus
- OptimizerConfig
- Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener)
- Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener)
- _raw_to_celsius
- .__init__
- Pipeline
- Session 10 — T3.2 (UB-16/17, M5): pretrained encoders via timm/MONAI
- Session 10 — T3.2 (UB-16/17, M5): pretrained encoders — timm route

## God Nodes (most connected - your core abstractions)
1. `UnifiedTrainer` - 33 edges
2. `Config` - 32 edges
3. `OptimizerConfig` - 28 edges
4. `SchedulerConfig` - 28 edges
5. `RecipesConfig` - 23 edges
6. `SetupManager` - 20 edges
7. `Pipeline` - 19 edges
8. `SegmentationMetrics` - 19 edges
9. `MultiDirectoryDataLoader` - 18 edges
10. `Graphify Full Pipeline` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [EXTRACTED]
  prompts/phase1_session1.md → CLAUDE.md
- `Automatic Offline Preprocessing (--force-preprocess)` --conceptually_related_to--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [INFERRED]
  README.md → CLAUDE.md
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `UB-02: Checkpoint Filename Contract Mismatch (current frontier)`  [EXTRACTED]
  prompts/phase1_session1.md → CLAUDE.md
- `ModelBenchmark` --uses--> `SegmentationMetrics`  [INFERRED]
  codes/benchmark_models.py → codes/metrics.py
- `ModelBenchmark` --uses--> `Config`  [INFERRED]
  codes/benchmark_models.py → codes/unified_data.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Extraction Pipeline (AST + Semantic + Cache)** — _claude_skills_graphify_skill_md_ast_extraction, _claude_skills_graphify_skill_md_semantic_extraction, _claude_skills_graphify_skill_md_extraction_cache [EXTRACTED 1.00]
- **UB-01 Fix Chain (T1.1 auto-preprocessing)** — claude_ub01_missing_preprocessing, codes_main_pipeline, codes_preprocess_data, readme_automatic_offline_preprocessing, claude_smoke_test_gate [INFERRED 0.85]
- **UB-23 CPU Enablement (UBENCH_ALLOW_CPU)** — claude_ub23_cpu_hard_exit, claude_ubench_allow_cpu, codes_hardware_detector, codes_tests_test_hardware_cpu [INFERRED 0.85]

## Communities (77 total, 8 thin omitted)

### Community 0 - "Benchmark & Orchestration Layer"
Cohesion: 0.14
Nodes (15): Measure per-image inference latency honestly (UB-09, M3).      Two correctness p, timed_inference(), _fake_loader(), _FakeImages, _FakeModel, MonkeyPatch, Timing honesty tests (UB-09, T2.1).  Two guarantees:  * ``benchmark_models.timed, CPU device → synchronize() is never called (nothing to synchronize). (+7 more)

### Community 1 - "Model Registry & Swin-UNet++"
Cohesion: 0.11
Nodes (23): compute_segmentation_metrics(), Tensor, Segmentation metric authority (UB-11, T2.2).  One definition per metric, shared, One-shot metrics for a single ``(logits, target)`` pair.      Convenience wrappe, Accumulate a confusion matrix and derive hard IoU / Dice from it.      Usage mir, Clear all accumulated state., Accumulate one batch.          Args:             logits_or_preds: either raw log, Return hard IoU/Dice with macro (excl. absent) and background split.          Re (+15 more)

### Community 2 - "Hardware Detection & CPU Mode"
Cohesion: 0.22
Nodes (8): End-of-session report — same format:, Session 4 — T1.5 (UB-08) + T1.6 (UB-07): mask-offset clamp & failure honesty, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (mandatory, per CLAUDE.md §10), Step 1 — Commit 1: T1.5 / UB-08 — clamped crop origin, aligned mask offsets, Step 2 — Commit 2: T1.6 / UB-07 — failure registry, honest exit codes, Step 3 — Docs commit: ledger flips (UB-08, UB-07 → FIXED@<sha>); §2 current-state rewrite (remaining Phase-1: UB-06, UB-13/14); §3.2 run.sh warning note removed; commit this session's prompt file., Step 4 — Push & remote CI

### Community 3 - "Environment Setup (setup.py)"
Cohesion: 0.05
Nodes (28): Console Logger (Tee) ==================== Redirects sys.stdout and sys.stderr so, Intercepts sys.stdout and sys.stderr and mirrors them to a log file.      Parame, Start a TeeLogger by reading the log directory from the     ``UBENCH_LOG_DIR`` e, A file-like object that writes to two streams simultaneously.     One stream is, Prefix every complete line with a timestamp., start_from_env(), TeeLogger, _TeeStream (+20 more)

### Community 4 - "Training Loop Internals"
Cohesion: 0.14
Nodes (19): checkpoint_path(), Path, Return the canonical checkpoint path for (model_key, fold, kind).      Args:, _path_for(), Path, UB-02: train↔benchmark checkpoint path contract (CLAUDE.md §7.1).  ``codes/namin, Every produced filename contains the registry key verbatim., UB-02 tombstone: display-name derivations never appear in any path. (+11 more)

### Community 5 - "Data Extraction & LFS"
Cohesion: 0.13
Nodes (24): check_data_status(), _discover_zip_files(), extract_all_data(), _extract_zip(), _generate_annotations(), _is_data_already_extracted(), _is_lfs_pointer(), _process_extracted_contents() (+16 more)

### Community 6 - "Graphify Skill Docs"
Cohesion: 0.09
Nodes (25): Add URL to Corpus (/graphify add), MCP Server (graphify.serve), Wiki Export (--wiki), Confidence Score Rubric, Node ID Format Rules, Extraction Subagent Prompt Template, GitHub Clone and Cross-Repo Merge, Native CLAUDE.md Integration (+17 more)

### Community 7 - "Engineering Guide & Preprocessing"
Cohesion: 0.18
Nodes (17): _metadata_frame(), DataFrame, UB-03/UB-04: fold-count guard with leave-subjects-out semantics.  ``GroupKFold(g, Tiny stand-in for metadata.csv: sample_id + dataset columns only., 3 subjects, K=5 → effective_k == 3, loud warning naming both., 2 subjects, K=5 → effective_k == 2 (the minimum viable CV)., 1 subject → clear leave-subjects-out error, not sklearn's., K <= n_groups → requested K unchanged, no warning emitted. (+9 more)

### Community 8 - "TransUNet Architecture"
Cohesion: 0.10
Nodes (12): CNNEncoder, MLP, MultiHeadAttention, CNN encoder for feature extraction (ResNet-50 style), Multi-head self-attention mechanism, TransUNet: Transformer-CNN Hybrid Architecture, MLP block for transformer, Transformer encoder block (+4 more)

### Community 9 - "test_preprocess_offsets.py"
Cohesion: 0.05
Nodes (48): UB-01: Preprocessing Never Invoked (FIXED@03e6dbb), preprocess_all_data(), Offline Data Preprocessing Script ================================= Pre-computes, _assert_aligned(), _centroid(), _expected_mask(), offset_dataset(), MonkeyPatch (+40 more)

### Community 10 - "NaN Debug Scripts (Ad-hoc)"
Cohesion: 0.16
Nodes (9): Model Registry ============== Dynamically load and register model architectures., Decorator to register a model class, register_model(), Thermal Facial Region Detection System - TransUNet =============================, DoubleConv, Thermal Facial Region Detection System - U-Net Model ===========================, Double convolution block for U-Net, U-Net architecture for semantic segmentation (+1 more)

### Community 11 - "Pipeline Stages (main_pipeline)"
Cohesion: 0.25
Nodes (7): Create DataLoaders for a single model + fold (lazy, on-demand).          Uses ``, Run comprehensive benchmark on all trained models, aggregating across all folds, create_test_loader(), DataLoader, Create a DataLoader over exactly the held-out test subjects (M1).      Returns `, Explicitly shut down DataLoader worker processes and free OS resources.      On, shutdown_data_loaders()

### Community 12 - "Loss Functions (Dice/Combined)"
Cohesion: 0.12
Nodes (9): DiceLoss, DataLoader, Module, Initialize trainer          Args:             model: PyTorch model to train, Return a GradScaler when the active GPU can benefit from AMP, else None., Construct the optimizer from a resolved recipe (adam | adamw)., Construct the LR scheduler from a resolved recipe.          Returns ``(scheduler, Dice Loss for segmentation.      NOTE: softmax on fp16 logits overflows (exp of (+1 more)

### Community 13 - "Multi-Dataset Data Loading"
Cohesion: 0.17
Nodes (19): _cfg(), Held-out test-subject tests (M1, T2.4).  Subjects listed in ``config.test_subjec, With no test subjects: full CV pool, empty test set, no test loader., Fabricate data/processed/metadata.csv for the given subjects., No fold's train or val split may contain a held-out test subject., The test loader contains exactly the configured test subjects, nothing else., Reserving subjects that leave <2 in the CV pool raises the >=2 guard., A test subject not present in the data is a hard error, not a silent no-op. (+11 more)

### Community 14 - "ModelBenchmark Runner"
Cohesion: 0.16
Nodes (15): evaluate_accuracy(), ModelBenchmark, DataFrame, DataLoader, Module, Path, Accuracy of one model on one loader via the shared authority (UB-11/R5).      Re, Comprehensive benchmark for model comparison (+7 more)

### Community 15 - "E2E Smoke Test Gate"
Cohesion: 0.06
Nodes (44): _error_logs_in_run_dir(), Path, UB-07: training failures must be recorded, reported, and fatal.  Before T1.6, th, error_log_*.txt inside the run's own log dir, parsed from the banner     so erro, Injected per-model failure → exit 1 + named summary + error log.      Without ``, --fail-fast: the first failure aborts the run — no later folds., A Pipeline() constructor crash reaches main()'s error-log writer.      An empty, _set_fast_env() (+36 more)

### Community 16 - "Synthetic Test Fixture"
Cohesion: 0.28
Nodes (8): _build_subject(), Shared pytest fixtures for UBench test suite.  Key fixture: ``synthetic_dataset`, Session-scoped synthetic dataset root (see CLAUDE.md §7.2).      Layout::, Return 4-corner polygon [[x,y], ...] for a rectangle., Create all synthetic files for one subject S{subject_n}., _rect_polygon(), synthetic_dataset(), TempPathFactory

### Community 17 - "CI & Phase 0 Kickoff"
Cohesion: 0.29
Nodes (7): GitHub Actions CI Workflow, Ruff Lint Gate (CI), Pytest Test Gate (CI), Phase 0 Kickoff Prompt (Session 0), T0.1 Synthetic Fixture + Smoke Test Task, T0.2 Repo Plumbing + CI Task, strict xfail Pattern for UB-01 Smoke Test

### Community 18 - "Shell Entry Points"
Cohesion: 0.48
Nodes (6): main(), run_extract(), run_pipeline(), run_setup(), run.sh script, UBENCH_RUN_ID

### Community 19 - "Active Config (codes/config.yaml)"
Cohesion: 0.40
Nodes (5): batch_sizes (unconsumed in config), k_folds (5), num_classes (10), Region Names (Portuguese, 10 classes), UBench codes/config.yaml (active config)

### Community 20 - "Dead Root Config (UB-12)"
Cohesion: 0.40
Nodes (5): Augmentation Configuration Section (root config), Loss Configuration Section (root config), Optimizer Configuration Section (root config), Root config.yaml (dead - loaded by nothing), Scheduler Configuration Section (root config)

### Community 25 - "ID Inspection Utility"
Cohesion: 0.10
Nodes (24): load_config(), Top-level config document (the whole of ``codes/config.yaml``)., Load and validate ``codes/config.yaml`` into a :class:`RootConfig`.      Args:, RootConfig, Path, A non-'balanced' string for class_weights fails schema validation., Write ``data`` as a config.yaml under ``tmp_path`` and return its path., The committed codes/config.yaml passes the schema. (+16 more)

### Community 26 - "Missing Data Printer"
Cohesion: 0.15
Nodes (16): _cfg_with_metadata(), _NoisyDataset, Dataset, Reproducibility tests (UB-20a, M6, T2.5).  Two seeded runs must produce identica, create_single_fold_loader sets a seeded generator and seed_worker (no iteration), seed_worker gives the same seed for a given worker id, different across ids., Each item draws from numpy — i.e. depends on the worker's numpy RNG., A per-loader seeded generator makes the 2-worker stream reproducible     regardl (+8 more)

### Community 27 - "Edge Case Tests"
Cohesion: 0.15
Nodes (20): BaseModel, FamilyRecipe, ModelConfig, PathsConfig, Typed schema for the single UBench config (``codes/config.yaml``).  UB-12 (T3.1), Per-family optimizer/scheduler override (``recipes.families.<name>``).      A ``, Per-family recipe assignments (``recipes:`` section, T3.3/UB-18/M4).      ``mode, Resolve the effective (optimizer, scheduler) recipe for a model (M4).      An un (+12 more)

### Community 30 - "Regex Verification Utility"
Cohesion: 0.11
Nodes (17): epoch_checkpoint_glob(), Reject anything that is not a lowercase snake_case registry key.      Display na, Glob pattern matching every epoch checkpoint of one (key, fold) series.      Use, _validate_model_key(), Path, Unified training pipeline for all models     Ensures consistent training, valida, Advance a per-batch scheduler (warmup_cosine) after an optimizer step., Return the expected path for a given epoch checkpoint (UB-02/R5). (+9 more)

### Community 31 - "Root Conftest"
Cohesion: 0.09
Nodes (17): NestedConvBlock, PatchEmbed, PatchMerging, Thermal Facial Region Detection System - Swin-UNet++ ===========================, Patch Merging Layer for downsampling, Image to Patch Embedding, Nested convolution block for dense skip connections, Swin-UNet++ architecture with nested dense skip connections (+9 more)

### Community 32 - "CLAUDE.md"
Cohesion: 0.05
Nodes (35): 10. Verification Gates & Definition of Done, 11. Anti-Patterns — Forbidden Actions, 12. Quick Command Reference, 1. What this is, 2. Current state — READ FIRST, 3.1 Environment setup (always a venv — never system Python), 3.2 Entry points and flag forwarding, 3.3 Running stages directly (+27 more)

### Community 33 - "UBench — Thermal Face Detection Benchmark Pipeline"
Cohesion: 0.05
Nodes (36): 1. Data Verification Checklist, 1. Major Updates Implemented, 1. What Happens Automatically, 2. Code changes details, 2. Common Issues & Troubleshooting, 2. Execution Flow Diagram, 3. Benefits, 3. Multi-Dataset Combination Details (+28 more)

### Community 34 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 35 - "test_suite.py"
Cohesion: 0.16
Nodes (18): _format_vram(), probe_peak_memory(), Peak GPU memory (MB) for one inference forward on a FIXED synthetic batch., Render a VRAM figure, or 'n/a (CPU)' when it was not measured., _patch_cuda(), MonkeyPatch, Fixed-batch VRAM probe tests (UB-10, T2.3).  The benchmark used to read peak VRA, Records the shape of the tensor it is called with; ``eval()`` -> self. (+10 more)

### Community 36 - "hardware_detector.py"
Cohesion: 0.05
Nodes (38): ML & Benchmark Methodology Standards M1-M9, Non-Negotiable Engineering Rules R1-R10, Known-Defect Ledger (UB-01..UB-23), Leave-Subjects-Out Cross-Validation (GroupKFold), Phased Task Plan (Phase 0-4), E2E Smoke Test Merge Gate (strict xfail frontier), Synthetic Dataset Fixture (S1-S5, 64x64 uint16), UB-20: Dependency & Reproducibility Hygiene (+30 more)

### Community 37 - "test_batch_size_keys.py"
Cohesion: 0.20
Nodes (9): Docs, End-of-session report — same format, plus:, Session 11 — T3.3 (UB-18, M4): per-family recipes + selection by val mIoU, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Suite budget + push & CI, Tests (red-first), The defect (UB-18) (+1 more)

### Community 38 - "swin_unet_plus_plus.py"
Cohesion: 0.18
Nodes (14): get_registered_models(), Return a list of registered model names, _profile(), UB-05: batch-size dicts keyed by canonical registry names, hard lookup.  The har, Simulate a hardware tier without any GPU present (pure Python)., Ledger reference value: the 6 GB (GTX 1660 Ti) tier gives Swin 6., Ledger reference value: the <5.5 GB tier gives Swin 3., Every tier's keys == the registered model keys (parity invariant). (+6 more)

### Community 39 - "DoubleConv"
Cohesion: 0.22
Nodes (8): End-of-session report — same format:, Session 5 — T1.7 (UB-06) + T1.8 (UB-13/14): resume & run-identity — Phase-1 closeout, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (mandatory, per CLAUDE.md §10), Step 1 — Commit 1: T1.7 / UB-06 — `--resume <run_id>` and `outputs/latest`, Step 2 — Commit 2: T1.8 / UB-13+UB-14 — epochs flag honored, one run id per run, Step 3 — Docs commit: ledger flips (UB-06, UB-13, UB-14); §2 rewritten to **Phase-1 complete** (all §9 Phase-1 boxes checked; remaining risk shifts to Phase-2 number-trust items UB-09/10/11); §3.2 "--epochs 100 ignored" note removed; §4.7 consequences paragraph updated (auto-resume works via --resume; single timestamp per run); commit this session's prompt file., Step 4 — Push & remote CI

### Community 40 - "Config"
Cohesion: 0.21
Nodes (8): ndarray, Visualize predicted regions, Print a formatted report of thermal statistics for all regions, Inference class for detecting facial regions in thermal images, Normalize thermal image to [0, 1] range.          A flat image (min == max) norm, Predict facial regions in a thermal image          Args:             thermal_ima, Calculate statistics information for each region using original thermal data in, ThermalFaceDetector

### Community 41 - "unified_data.py"
Cohesion: 0.13
Nodes (15): UB-02: Checkpoint Filename Contract Mismatch (current frontier), Comprehensive Model Benchmarking Suite ====================================== Co, Main Training Pipeline Orchestrator =================================== Manages, Single authority for checkpoint file naming (UB-02, R5).  Registry keys (``unet`, Config, Initialize and validate paths                  Args:             output_dir: Ove, Validate required data paths exist, Create the run's output/log roots only.          The ``models``/``plots``/``pred (+7 more)

### Community 42 - "Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session"
Cohesion: 0.20
Nodes (9): End-of-session report — same format, plus:, Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session, Standing guardrails (new + repeated), Step 0 — Environment & gates baseline, Step 1 — RED: `codes/tests/test_filenames.py`, Step 2 — GREEN: `codes/naming.py` + wiring both sides, Step 3 — Marker removal is atomic with the fix, Step 4 — Docs & ledger (same commit or trailing docs commit) (+1 more)

### Community 43 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 44 - "Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01)"
Cohesion: 0.22
Nodes (8): End-of-session report — same format as Phase 0, plus:, Hard scope exclusions this session, Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01), Step 0 — Environment & branch, Step 1 — Commit 1 (docs only): make CLAUDE.md true again, Step 2 — Commit 2: UB-23 — real CPU mode; delete the test-only bypass, Step 3 — Commit 3: UB-20 (partial) — pin albumentations, Step 4 — Commit 4: T1.1 / UB-01 — wire preprocessing into the pipeline

### Community 45 - "HardwareProfile"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 7 — T2.3 (UB-10) + T2.4 (M1): fixed-batch VRAM probe & held-out test subjects, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — Commit 1: T2.3 / UB-10 — fixed-batch-size VRAM probe, Step 2 — Commit 2: T2.4 / M1 — held-out test subjects, CV vs TEST report, Step 3 — Docs, Step 4 — Suite budget + push & CI

### Community 46 - "Session 6 — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric definition — Phase-2 opener"
Cohesion: 0.18
Nodes (10): End-of-session report — same format, plus:, Session 6 (final) — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric authority — Phase-2 opener, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10) + one-time phase-boundary digest, Step 1 — Commit 1 (chore): test-suite disk retention + UB-25, Step 2 — Commit 2: T2.1 / UB-09 — honest timing, Step 3 — Commit 3: T2.2 / UB-11 — one metric authority, Step 4 — Deliverable for the human: `docs/phase1_realdata_checklist.md` (+2 more)

### Community 47 - "ThermalFaceDataset"
Cohesion: 0.14
Nodes (14): load_test_metadata(), DataFrame, Dataset, Unified Data Loading Module - Multi-Directory Support ==========================, Convert raw thermal sensor value to degrees Celsius.      Defined at module leve, Unified PyTorch Dataset reading from offline preprocessed arrays, Read every processed metadata row (test subjects included).      Applies the ``L, Return the configured held-out test subjects, validated against the data.      H (+6 more)

### Community 48 - "Session 0 — Build the safety net (Phase 0 of CLAUDE.md)"
Cohesion: 0.29
Nodes (6): End-of-session report (required format), Scope lock — Phase 0 only (T0.1 + T0.2), Session 0 — Build the safety net (Phase 0 of CLAUDE.md), Step 0 — Environment & reconnaissance, Step 1 — T0.1: synthetic fixture + smoke test, Step 2 — T0.2: repo plumbing + CI

### Community 49 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 50 - "DiceLoss"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 7 — T2.3 (UB-10) + T2.4 (M1): fixed-batch VRAM probe & held-out test subjects, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — Commit 1: T2.3 / UB-10 — fixed-batch-size VRAM probe, Step 2 — Commit 2: T2.4 / M1 — held-out test subjects, CV vs TEST report, Step 3 — Docs, Step 4 — Suite budget + push & CI

### Community 51 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 52 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 53 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 54 - "Session 6 (final) — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric authority — Phase-2 opener"
Cohesion: 0.18
Nodes (10): End-of-session report — same format, plus:, Session 6 (final) — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric authority — Phase-2 opener, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10) + one-time phase-boundary digest, Step 1 — Commit 1 (chore): test-suite disk retention + UB-25, Step 2 — Commit 2: T2.1 / UB-09 — honest timing, Step 3 — Commit 3: T2.2 / UB-11 — one metric authority, Step 4 — Deliverable for the human: `docs/phase1_realdata_checklist.md` (+2 more)

### Community 55 - "Phase-1 real-data validation checklist (GPU box)"
Cohesion: 0.20
Nodes (9): 0. Preconditions, 1. Fresh clone + data, 2. Environment (GPU path, §3.1), 3. Dry run (single model, 2 epochs), 4. Full run, 5. Resume drill (the Phase-1 headline behavior — UB-06), 6. Expected artifacts, 7. Capture protocol (+1 more)

### Community 58 - "Session 3 — T1.3 (UB-05) + T1.4 (UB-03/04): batch-size contract & fold-count guard"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 3 — T1.3 (UB-05) + T1.4 (UB-03/04): batch-size contract & fold-count guard, Standing guardrails, Step 0 — PREDECESSOR VERIFICATION GATE (mandatory, before any work), Step 1 — Commit 1: T1.3 / UB-05 — canonical batch-size keys, hard lookup, Step 2 — Commit 2: T1.4 / UB-03+UB-04 — fold-count guard & split-semantics docs, Step 3 — Docs commit (or fold into Step 2's): ledger flips, Session Entry Protocol added to §10, §2 "Current state" updated (partial-corpus crash resolved; remaining Phase-1 items: UB-08, UB-07, UB-06, UB-13/14)., Step 4 — Push & remote CI

### Community 61 - "detect_and_optimize"
Cohesion: 0.17
Nodes (11): detect_and_optimize(), Main function to detect hardware and return optimized profile      Args:, collect_run_metadata(), _git(), Path, Initialize pipeline          Args:             models_to_train: List of model na, Point ``outputs/latest`` at this run's output dir (UB-06).          The replacem, Best-effort ``git`` invocation for run provenance; never raises. (+3 more)

### Community 62 - "UnifiedTrainer"
Cohesion: 0.13
Nodes (20): create_model(), Module, Instantiate a model by name, Tensor, Shared 1-channel stem adaptation for pretrained (RGB) encoders (M5).  Pretrained, Resolve the pretrained flag shared by the pretrained encoders (M5).      An expl, Sum a pretrained conv's 3 RGB input-channel kernels into 1 channel.      Args:, resolve_pretrained() (+12 more)

### Community 63 - "create_test_loader"
Cohesion: 0.25
Nodes (5): Dataset, 4 tiny (1, 8, 8) frames with integer masks in ``[0, num_classes)``., validate() returns exactly (loss, mIoU, dice) and holds no timing state., test_validate_emits_no_timing(), _TinyDataset

### Community 64 - "Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — Commit 1: T2.5 / UB-20a + M6 — seeded determinism + per-run metadata, Step 2 — T2.6 / UB-15, UB-22, UB-24, UB-25 — dead-code & scratch sweep, Step 3 — Docs & Phase-2 close-out, Step 4 — Suite budget + push & CI

### Community 65 - "Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — Commit 1: T2.5 / UB-20a + M6 — seeded determinism + per-run metadata, Step 2 — T2.6 / UB-15, UB-22, UB-24, UB-25 — dead-code & scratch sweep, Step 3 — Docs & Phase-2 close-out, Step 4 — Suite budget + push & CI

### Community 66 - "SegmentationMetrics"
Cohesion: 0.15
Nodes (20): LossConfig, Combined Cross-Entropy + Dice loss (``loss:`` section).      ``class_weights`` c, _make_trainer(), UB-12 (T3.1): the single validated config authority.  ``codes/config.yaml`` is l, Build a UnifiedTrainer on CPU with a lightweight stand-in config.      Exercises, test_class_weights_bad_string_raises(), test_class_weights_balanced_is_nonuniform(), test_class_weights_explicit_list() (+12 more)

### Community 67 - "model_registry.py"
Cohesion: 0.22
Nodes (8): Docs — ledger UB-18 → `FIXED@<sha>` (all three sub-claims now closed: per-family recipes, selection by headline metric, and the held-out test set landed in T2.4 — say so in the row). §4.6 + §8 M4 reconciled; §2 frontier → T3.4 (UB-19/M7). README config table gains the family overrides. Save this prompt (R9)., End-of-session report — same format, plus:, Session 11 — T3.3 (UB-18, M4): per-family recipes + selection by val mIoU, Standing guardrails — excluded: thermal preprocessing/augmentation (T3.4), lockfiles/CUDA index (T3.5/UB-20b), import restructure (T3.6), Wilcoxon (T4.1). No fabricated comparisons (R10) — the per-family + pretrained-vs-scratch numbers come from the human GPU box. Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait., Step 0 — SESSION ENTRY PROTOCOL (§10) + first timm-era CI check ▲, Suite budget + push & CI — measure (was 4:44 local; the runner is slower — Step 0.4 may already have split CI). Push; paste Actions conclusion or `UNVERIFIED` + the FF/push one-liner., Tests (red-first) — draft's list + the cadence-contract test, warmup arithmetic, and updated `test_resume`. No new subprocess runs; smoke trio unchanged (transformers inside it now run AdamW+warmup — structure-asserting, stays green, and the degenerate-warmup guard is what keeps it so)., The work — adopt the draft's three commits, with these amendments

### Community 69 - "OptimizerConfig"
Cohesion: 0.17
Nodes (24): OptimizerConfig, LR scheduler recipe (``scheduler:`` section or a per-family override).      ``na, Optimizer recipe (``optimizer:`` section or a per-family override).      ``name`, SchedulerConfig, collect_run_metadata records git/env/seed/config provenance (M6)., test_run_metadata_has_provenance_keys(), test_single_train_step(), _cfg() (+16 more)

### Community 70 - "Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener)"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener), Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — the work (schema + wire/delete). Keep commits per-concern (R6); suggested split below., Step 2 — Docs, Step 3 — Suite budget + push & CI, The exact current behavior T3.1 must preserve (do not move these numbers)

### Community 71 - "Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener)"
Cohesion: 0.25
Nodes (7): End-of-session report — same format, plus:, Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener), Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — Commit 1: T3.1 / UB-12 — single validated config, Step 2 — Docs, Step 3 — Suite budget + push & CI

### Community 72 - "_raw_to_celsius"
Cohesion: 0.22
Nodes (7): _DoubleConv, Tensor, Pretrained SwinV2 encoder + UNet decoder (M5; supersedes UB-17).  The hand-rolle, Bilinear-upsample ×2, concatenate the encoder skip, then DoubleConv., ImageNet-pretrained SwinV2-tiny encoder + U-Net-style conv decoder., SwinV2UNet, _Up

### Community 73 - ".__init__"
Cohesion: 0.21
Nodes (7): _DoubleConv, Tensor, Pretrained R50+ViT-B/16 hybrid encoder + upsampling decoder (M5; UB-16).  ``tran, Bilinear-upsample ×2 then DoubleConv (no skip — option (a))., ImageNet-pretrained R50+ViT-B/16 hybrid encoder + upsampling decoder., TransUNetPretrained, _UpConv

### Community 74 - "Pipeline"
Cohesion: 0.13
Nodes (15): apply_epochs_override(), main(), Pipeline, Export ``--epochs`` to the ``NUM_EPOCHS`` env var Config reads (UB-13).      Exp, Persist *text* to a timestamped error log and return its path.      Single autho, Main training pipeline orchestrator, Run offline preprocessing when its outputs are missing (UB-01, T1.1).          T, Shut down DataLoader workers and free GPU memory between runs. (+7 more)

### Community 76 - "Session 10 — T3.2 (UB-16/17, M5): pretrained encoders via timm/MONAI"
Cohesion: 0.20
Nodes (9): Context you must carry from Session 9, End-of-session report, Session 10 — T3.2 (UB-16/17, M5): pretrained encoders via timm/MONAI, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Step 1 — the work (M5). Keep commits per-concern (R6)., Step 2 — the network/dependency reality (decide + encode) ▲, Step 3 — tests (red-first where practical) (+1 more)

### Community 77 - "Session 10 — T3.2 (UB-16/17, M5): pretrained encoders — timm route"
Cohesion: 0.22
Nodes (8): Docs, End-of-session report — same format, plus:, Session 10 — T3.2 (UB-16/17, M5): pretrained encoders — timm route, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (§10), Suite budget + push & CI, Tests (red-first where practical) — ▲ smoke is NOT extended, The work (M5) — per-concern commits (R6)

## Knowledge Gaps
- **271 isolated node(s):** `UBENCH_RUN_ID`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+266 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UnifiedTrainer` connect `Regex Verification Utility` to `Benchmark & Orchestration Layer`, `Model Registry & Swin-UNet++`, `SegmentationMetrics`, `SwinUNetPlusPlus`, `OptimizerConfig`, `unified_data.py`, `Pipeline`, `Loss Functions (Dice/Combined)`, `UnifiedTrainer`, `create_test_loader`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `UBench README (User Documentation)` connect `unified_data.py` to `UBench — Thermal Face Detection Benchmark Pipeline`, `Environment Setup (setup.py)`, `hardware_detector.py`, `ThermalFaceDataset`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `Config` connect `unified_data.py` to `Config`, `test_preprocess_offsets.py`, `Pipeline`, `Pipeline Stages (main_pipeline)`, `Loss Functions (Dice/Combined)`, `Multi-Dataset Data Loading`, `ModelBenchmark Runner`, `ThermalFaceDataset`, `Missing Data Printer`, `detect_and_optimize`, `Regex Verification Utility`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `UnifiedTrainer` (e.g. with `Pipeline` and `_SpyScheduler`) actually correct?**
  _`UnifiedTrainer` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Config` (e.g. with `ModelBenchmark` and `Pipeline`) actually correct?**
  _`Config` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `OptimizerConfig` (e.g. with `_NoisyDataset` and `_SpyScheduler`) actually correct?**
  _`OptimizerConfig` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SchedulerConfig` (e.g. with `_NoisyDataset` and `_SpyScheduler`) actually correct?**
  _`SchedulerConfig` has 5 INFERRED edges - model-reasoned connections that need verification._
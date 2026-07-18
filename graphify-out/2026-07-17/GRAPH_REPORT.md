# Graph Report - UBench  (2026-07-17)

## Corpus Check
- 74 files · ~61,950 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 957 nodes · 1438 edges · 62 communities (55 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0ca7985e`
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

## God Nodes (most connected - your core abstractions)
1. `Config` - 37 edges
2. `UnifiedTrainer` - 23 edges
3. `Pipeline` - 20 edges
4. `SetupManager` - 20 edges
5. `MultiDirectoryDataLoader` - 20 edges
6. `SegmentationMetrics` - 19 edges
7. `Graphify Full Pipeline` - 16 edges
8. `run_benchmark()` - 15 edges
9. `run_pipeline_subprocess()` - 15 edges
10. `load_split_metadata()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Automatic Offline Preprocessing (--force-preprocess)` --conceptually_related_to--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [INFERRED]
  README.md → CLAUDE.md
- `ModelBenchmark` --uses--> `SegmentationMetrics`  [INFERRED]
  codes/benchmark_models.py → codes/metrics.py
- `ModelBenchmark` --uses--> `Config`  [INFERRED]
  codes/benchmark_models.py → codes/unified_data.py
- `ModelBenchmark` --uses--> `CombinedLoss`  [INFERRED]
  codes/benchmark_models.py → codes/unified_training.py
- `Pipeline` --uses--> `HardwareProfile`  [INFERRED]
  codes/main_pipeline.py → codes/hardware_detector.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Extraction Pipeline (AST + Semantic + Cache)** — _claude_skills_graphify_skill_md_ast_extraction, _claude_skills_graphify_skill_md_semantic_extraction, _claude_skills_graphify_skill_md_extraction_cache [EXTRACTED 1.00]
- **UB-01 Fix Chain (T1.1 auto-preprocessing)** — claude_ub01_missing_preprocessing, codes_main_pipeline, codes_preprocess_data, readme_automatic_offline_preprocessing, claude_smoke_test_gate [INFERRED 0.85]
- **UB-23 CPU Enablement (UBENCH_ALLOW_CPU)** — claude_ub23_cpu_hard_exit, claude_ubench_allow_cpu, codes_hardware_detector, codes_tests_test_hardware_cpu [INFERRED 0.85]

## Communities (62 total, 7 thin omitted)

### Community 0 - "Benchmark & Orchestration Layer"
Cohesion: 0.09
Nodes (20): Measure per-image inference latency honestly (UB-09, M3).      Two correctness p, timed_inference(), _fake_loader(), _FakeImages, _FakeModel, Dataset, MonkeyPatch, Timing honesty tests (UB-09, T2.1).  Two guarantees:  * ``benchmark_models.timed (+12 more)

### Community 1 - "Model Registry & Swin-UNet++"
Cohesion: 0.14
Nodes (16): compute_segmentation_metrics(), Tensor, One-shot metrics for a single ``(logits, target)`` pair.      Convenience wrappe, Accumulate one batch.          Args:             logits_or_preds: either raw log, Return hard IoU/Dice with macro (excl. absent) and background split.          Re, _logits_from_argmax(), Tensor, Build logits whose argmax over dim=1 is exactly ``pred`` (N, H, W). (+8 more)

### Community 2 - "Hardware Detection & CPU Mode"
Cohesion: 0.22
Nodes (8): End-of-session report — same format:, Session 4 — T1.5 (UB-08) + T1.6 (UB-07): mask-offset clamp & failure honesty, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (mandatory, per CLAUDE.md §10), Step 1 — Commit 1: T1.5 / UB-08 — clamped crop origin, aligned mask offsets, Step 2 — Commit 2: T1.6 / UB-07 — failure registry, honest exit codes, Step 3 — Docs commit: ledger flips (UB-08, UB-07 → FIXED@<sha>); §2 current-state rewrite (remaining Phase-1: UB-06, UB-13/14); §3.2 run.sh warning note removed; commit this session's prompt file., Step 4 — Push & remote CI

### Community 3 - "Environment Setup (setup.py)"
Cohesion: 0.09
Nodes (17): Upgrade pip to latest version, Detect the CUDA driver version using nvidia-smi.         Returns a version tuple, Return True if an already-installed PyTorch build has CUDA support.         Uses, Install the correct CUDA-enabled PyTorch build.         Plain `pip install torch, Manages cross-platform setup and dependency installation, Install non-PyTorch dependencies from requirements.txt, Install all dependencies (PyTorch CUDA first, then the rest), Check CUDA availability using a subprocess.          Using a subprocess (rather (+9 more)

### Community 4 - "Training Loop Internals"
Cohesion: 0.06
Nodes (40): checkpoint_path(), epoch_checkpoint_glob(), Path, Single authority for checkpoint file naming (UB-02, R5).  Registry keys (``unet`, Reject anything that is not a lowercase snake_case registry key.      Display na, Return the canonical checkpoint path for (model_key, fold, kind).      Args:, Glob pattern matching every epoch checkpoint of one (key, fold) series.      Use, _validate_model_key() (+32 more)

### Community 5 - "Data Extraction & LFS"
Cohesion: 0.07
Nodes (35): check_data_status(), _discover_zip_files(), extract_all_data(), _extract_zip(), _generate_annotations(), _is_data_already_extracted(), _is_lfs_pointer(), _process_extracted_contents() (+27 more)

### Community 6 - "Graphify Skill Docs"
Cohesion: 0.09
Nodes (25): Add URL to Corpus (/graphify add), MCP Server (graphify.serve), Wiki Export (--wiki), Confidence Score Rubric, Node ID Format Rules, Extraction Subagent Prompt Template, GitHub Clone and Cross-Repo Merge, Native CLAUDE.md Integration (+17 more)

### Community 7 - "Engineering Guide & Preprocessing"
Cohesion: 0.15
Nodes (16): _metadata_frame(), DataFrame, UB-03/UB-04: fold-count guard with leave-subjects-out semantics.  ``GroupKFold(g, Tiny stand-in for metadata.csv: sample_id + dataset columns only., 3 subjects, K=5 → effective_k == 3, loud warning naming both., 2 subjects, K=5 → effective_k == 2 (the minimum viable CV)., 1 subject → clear leave-subjects-out error, not sklearn's., K <= n_groups → requested K unchanged, no warning emitted. (+8 more)

### Community 8 - "TransUNet Architecture"
Cohesion: 0.11
Nodes (13): CNNEncoder, MLP, MultiHeadAttention, Thermal Facial Region Detection System - TransUNet =============================, CNN encoder for feature extraction (ResNet-50 style), Multi-head self-attention mechanism, TransUNet: Transformer-CNN Hybrid Architecture, MLP block for transformer (+5 more)

### Community 9 - "test_preprocess_offsets.py"
Cohesion: 0.06
Nodes (41): preprocess_all_data(), Offline Data Preprocessing Script ================================= Pre-computes, _assert_aligned(), _centroid(), _expected_mask(), offset_dataset(), MonkeyPatch, ndarray (+33 more)

### Community 10 - "NaN Debug Scripts (Ad-hoc)"
Cohesion: 0.10
Nodes (12): get_latest_checkpoint(), main(), Path, Inference & Model Comparison Script =================================== Loads th, Search for a model checkpoint across standard locations., DoubleConv, Thermal Facial Region Detection System - U-Net Model ===========================, Double convolution block for U-Net (+4 more)

### Community 11 - "Pipeline Stages (main_pipeline)"
Cohesion: 0.13
Nodes (12): Pipeline, Run offline preprocessing when its outputs are missing (UB-01, T1.1).          T, Load annotations once (lightweight) — DataLoaders are created lazily., Create DataLoaders for a single model + fold (lazy, on-demand).          Uses ``, Shut down DataLoader workers and free GPU memory between runs., Train a dynamic model from the registry on a specific fold, Train all selected models sequentially over all folds, Run comprehensive benchmark on all trained models, aggregating across all folds (+4 more)

### Community 12 - "Loss Functions (Dice/Combined)"
Cohesion: 0.11
Nodes (13): test_detector_initialization_and_prediction(), DiceLoss, Module, ndarray, Tensor, Dice Loss for segmentation.      NOTE: softmax on fp16 logits overflows (exp of, Inference class for detecting facial regions in thermal images, Normalize thermal image to [0, 1] range (+5 more)

### Community 13 - "Multi-Dataset Data Loading"
Cohesion: 0.16
Nodes (20): _cfg(), Held-out test-subject tests (M1, T2.4).  Subjects listed in ``config.test_subjec, With no test subjects: full CV pool, empty test set, no test loader., Fabricate data/processed/metadata.csv for the given subjects., No fold's train or val split may contain a held-out test subject., Reserving subjects that leave <2 in the CV pool raises the >=2 guard., A test subject not present in the data is a hard error, not a silent no-op., _subject_of() (+12 more)

### Community 14 - "ModelBenchmark Runner"
Cohesion: 0.23
Nodes (11): evaluate_accuracy(), _format_vram(), DataLoader, Module, Path, Render a VRAM figure, or 'n/a (CPU)' when it was not measured., Accuracy of one model on one loader via the shared authority (UB-11/R5).      Re, Load trained model weights (+3 more)

### Community 15 - "E2E Smoke Test Gate"
Cohesion: 0.06
Nodes (42): _error_logs_in_run_dir(), Path, UB-07: training failures must be recorded, reported, and fatal.  Before T1.6, th, error_log_*.txt inside the run's own log dir, parsed from the banner     so erro, Injected per-model failure → exit 1 + named summary + error log.      Without ``, --fail-fast: the first failure aborts the run — no later folds., A Pipeline() constructor crash reaches main()'s error-log writer.      An empty, _set_fast_env() (+34 more)

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

### Community 32 - "CLAUDE.md"
Cohesion: 0.05
Nodes (35): 10. Verification Gates & Definition of Done, 11. Anti-Patterns — Forbidden Actions, 12. Quick Command Reference, 1. What this is, 2. Current state — READ FIRST, 3.1 Environment setup (always a venv — never system Python), 3.2 Entry points and flag forwarding, 3.3 Running stages directly (+27 more)

### Community 33 - "UBench — Thermal Face Detection Benchmark Pipeline"
Cohesion: 0.06
Nodes (35): 1. Data Verification Checklist, 1. Major Updates Implemented, 1. What Happens Automatically, 2. Code changes details, 2. Common Issues & Troubleshooting, 2. Execution Flow Diagram, 3. Benefits, 3. Multi-Dataset Combination Details (+27 more)

### Community 34 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 35 - "test_suite.py"
Cohesion: 0.17
Nodes (16): probe_peak_memory(), Peak GPU memory (MB) for one inference forward on a FIXED synthetic batch., _patch_cuda(), MonkeyPatch, Fixed-batch VRAM probe tests (UB-10, T2.3).  The benchmark used to read peak VRA, Records the shape of the tensor it is called with; ``eval()`` -> self., Make the CUDA branch runnable on a CPU box; return a call counter., On CPU there is nothing to measure: return None, render 'n/a (CPU)'. (+8 more)

### Community 36 - "hardware_detector.py"
Cohesion: 0.05
Nodes (43): ML & Benchmark Methodology Standards M1-M9, Non-Negotiable Engineering Rules R1-R10, Known-Defect Ledger (UB-01..UB-23), Leave-Subjects-Out Cross-Validation (GroupKFold), Phased Task Plan (Phase 0-4), E2E Smoke Test Merge Gate (strict xfail frontier), Synthetic Dataset Fixture (S1-S5, 64x64 uint16), UB-01: Preprocessing Never Invoked (FIXED@03e6dbb) (+35 more)

### Community 37 - "test_batch_size_keys.py"
Cohesion: 0.15
Nodes (12): The test loader contains exactly the configured test subjects, nothing else., test_held_out_loader_is_exactly_the_test_subjects(), create_single_fold_loader(), create_test_loader(), DataLoader, Dataset, Unified PyTorch Dataset reading from offline preprocessed arrays, Create DataLoaders for a **single** K-Fold split.      Unlike ``create_kfold_dat (+4 more)

### Community 38 - "swin_unet_plus_plus.py"
Cohesion: 0.05
Nodes (40): create_model(), get_registered_models(), Module, Model Registry ============== Dynamically load and register model architectures., Decorator to register a model class, Return a list of registered model names, Instantiate a model by name, register_model() (+32 more)

### Community 39 - "DoubleConv"
Cohesion: 0.22
Nodes (8): End-of-session report — same format:, Session 5 — T1.7 (UB-06) + T1.8 (UB-13/14): resume & run-identity — Phase-1 closeout, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (mandatory, per CLAUDE.md §10), Step 1 — Commit 1: T1.7 / UB-06 — `--resume <run_id>` and `outputs/latest`, Step 2 — Commit 2: T1.8 / UB-13+UB-14 — epochs flag honored, one run id per run, Step 3 — Docs commit: ledger flips (UB-06, UB-13, UB-14); §2 rewritten to **Phase-1 complete** (all §9 Phase-1 boxes checked; remaining risk shifts to Phase-2 number-trust items UB-09/10/11); §3.2 "--epochs 100 ignored" note removed; §4.7 consequences paragraph updated (auto-resume works via --resume; single timestamp per run); commit this session's prompt file., Step 4 — Push & remote CI

### Community 40 - "Config"
Cohesion: 0.29
Nodes (6): ModelBenchmark, DataFrame, Comprehensive benchmark for model comparison, Generate comparison visualizations and reports, Create comprehensive comparison plots, Generate detailed text report

### Community 41 - "unified_data.py"
Cohesion: 0.14
Nodes (14): Comprehensive Model Benchmarking Suite ====================================== Co, Segmentation metric authority (UB-11, T2.2).  One definition per metric, shared, Accumulate a confusion matrix and derive hard IoU / Dice from it.      Usage mir, Clear all accumulated state., SegmentationMetrics, Metric authority tests (UB-11, T2.2).  One definition of IoU and Dice, shared by, Both modules resolve the same metric class (no divergent local copies)., test_trainer_and_benchmark_share_one_authority() (+6 more)

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
Cohesion: 0.13
Nodes (17): Config, load_test_metadata(), DataFrame, Unified Data Loading Module - Multi-Directory Support ==========================, Initialize and validate paths                  Args:             output_dir: Ove, Validate required data paths exist, Create output directories if they don't exist, Convert raw thermal sensor value to degrees Celsius.      Defined at module leve (+9 more)

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
Cohesion: 0.16
Nodes (12): apply_epochs_override(), main(), Path, Main Training Pipeline Orchestrator =================================== Manages, Point ``outputs/latest`` at this run's output dir (UB-06).          The replacem, Export ``--epochs`` to the ``NUM_EPOCHS`` env var Config reads (UB-13).      Exp, Main entry point with global error catcher for crash-safe operation., Persist *text* to a timestamped error log and return its path.      Single autho (+4 more)

## Knowledge Gaps
- **214 isolated node(s):** `UBENCH_RUN_ID`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+209 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UBench README (User Documentation)` connect `unified_data.py` to `UBench — Thermal Face Detection Benchmark Pipeline`, `hardware_detector.py`, `Data Extraction & LFS`, `ThermalFaceDataset`, `detect_and_optimize`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Why does `Config` connect `ThermalFaceDataset` to `Training Loop Internals`, `test_batch_size_keys.py`, `swin_unet_plus_plus.py`, `Config`, `unified_data.py`, `NaN Debug Scripts (Ad-hoc)`, `Pipeline Stages (main_pipeline)`, `test_preprocess_offsets.py`, `Multi-Dataset Data Loading`, `ModelBenchmark Runner`, `Loss Functions (Dice/Combined)`, `detect_and_optimize`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `UnifiedTrainer` connect `Training Loop Internals` to `Benchmark & Orchestration Layer`, `unified_data.py`, `Pipeline Stages (main_pipeline)`, `ThermalFaceDataset`, `detect_and_optimize`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Config` (e.g. with `ModelBenchmark` and `Pipeline`) actually correct?**
  _`Config` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `UnifiedTrainer` (e.g. with `Pipeline` and `_FakeImages`) actually correct?**
  _`UnifiedTrainer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Pipeline` (e.g. with `HardwareProfile` and `TeeLogger`) actually correct?**
  _`Pipeline` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `UBENCH_RUN_ID`, `graphify`, `Usage` to the rest of the system?**
  _214 weakly-connected nodes found - possible documentation gaps or missing edges._
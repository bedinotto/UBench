# Graph Report - UBench  (2026-07-16)

## Corpus Check
- 59 files · ~46,601 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 725 nodes · 1055 edges · 51 communities (44 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1c4b32b9`
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
- hardware_detector.py
- Config
- unified_data.py
- Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session
- graphify reference: extra exports and benchmark
- Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01)
- ThermalFaceDataset
- Session 0 — Build the safety net (Phase 0 of CLAUDE.md)
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- .__init__
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- graphify
- extraction-spec.md

## God Nodes (most connected - your core abstractions)
1. `Config` - 31 edges
2. `SetupManager` - 20 edges
3. `Pipeline` - 18 edges
4. `MultiDirectoryDataLoader` - 18 edges
5. `UnifiedTrainer` - 17 edges
6. `Graphify Full Pipeline` - 16 edges
7. `run_benchmark()` - 13 edges
8. `HardwareProfile` - 13 edges
9. `HardwareDetector` - 13 edges
10. `checkpoint_path()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [EXTRACTED]
  prompts/phase1_session1.md → CLAUDE.md
- `Automatic Offline Preprocessing (--force-preprocess)` --conceptually_related_to--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [INFERRED]
  README.md → CLAUDE.md
- `ModelBenchmark` --uses--> `Config`  [INFERRED]
  codes/benchmark_models.py → codes/unified_data.py
- `Pipeline` --uses--> `HardwareProfile`  [INFERRED]
  codes/main_pipeline.py → codes/hardware_detector.py
- `Pipeline` --uses--> `TeeLogger`  [INFERRED]
  codes/main_pipeline.py → codes/logger.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Extraction Pipeline (AST + Semantic + Cache)** — _claude_skills_graphify_skill_md_ast_extraction, _claude_skills_graphify_skill_md_semantic_extraction, _claude_skills_graphify_skill_md_extraction_cache [EXTRACTED 1.00]
- **UB-01 Fix Chain (T1.1 auto-preprocessing)** — claude_ub01_missing_preprocessing, codes_main_pipeline, codes_preprocess_data, readme_automatic_offline_preprocessing, claude_smoke_test_gate [INFERRED 0.85]
- **UB-23 CPU Enablement (UBENCH_ALLOW_CPU)** — claude_ub23_cpu_hard_exit, claude_ubench_allow_cpu, codes_hardware_detector, codes_tests_test_hardware_cpu [INFERRED 0.85]

## Communities (51 total, 7 thin omitted)

### Community 0 - "Benchmark & Orchestration Layer"
Cohesion: 0.22
Nodes (8): End-of-session report — same format, plus:, Session 3 — T1.3 (UB-05) + T1.4 (UB-03/04): batch-size contract & fold-count guard, Standing guardrails, Step 0 — PREDECESSOR VERIFICATION GATE (mandatory, before any work), Step 1 — Commit 1: T1.3 / UB-05 — canonical batch-size keys, hard lookup, Step 2 — Commit 2: T1.4 / UB-03+UB-04 — fold-count guard & split-semantics docs, Step 3 — Docs commit (or fold into Step 2's): ledger flips, Session Entry Protocol added to §10, §2 "Current state" updated (partial-corpus crash resolved; remaining Phase-1 items: UB-08, UB-07, UB-06, UB-13/14)., Step 4 — Push & remote CI

### Community 1 - "Model Registry & Swin-UNet++"
Cohesion: 0.05
Nodes (40): create_model(), get_registered_models(), Module, Model Registry ============== Dynamically load and register model architectures., Decorator to register a model class, Return a list of registered model names, Instantiate a model by name, register_model() (+32 more)

### Community 2 - "Hardware Detection & CPU Mode"
Cohesion: 0.22
Nodes (8): End-of-session report — same format:, Session 4 — T1.5 (UB-08) + T1.6 (UB-07): mask-offset clamp & failure honesty, Standing guardrails, Step 0 — SESSION ENTRY PROTOCOL (mandatory, per CLAUDE.md §10), Step 1 — Commit 1: T1.5 / UB-08 — clamped crop origin, aligned mask offsets, Step 2 — Commit 2: T1.6 / UB-07 — failure registry, honest exit codes, Step 3 — Docs commit: ledger flips (UB-08, UB-07 → FIXED@<sha>); §2 current-state rewrite (remaining Phase-1: UB-06, UB-13/14); §3.2 run.sh warning note removed; commit this session's prompt file., Step 4 — Push & remote CI

### Community 3 - "Environment Setup (setup.py)"
Cohesion: 0.05
Nodes (28): Console Logger (Tee) ==================== Redirects sys.stdout and sys.stderr so, Intercepts sys.stdout and sys.stderr and mirrors them to a log file.      Parame, Start a TeeLogger by reading the log directory from the     ``UBENCH_LOG_DIR`` e, A file-like object that writes to two streams simultaneously.     One stream is, Prefix every complete line with a timestamp., start_from_env(), TeeLogger, _TeeStream (+20 more)

### Community 4 - "Training Loop Internals"
Cohesion: 0.07
Nodes (32): checkpoint_path(), epoch_checkpoint_glob(), Path, Single authority for checkpoint file naming (UB-02, R5).  Registry keys (``unet`, Reject anything that is not a lowercase snake_case registry key.      Display na, Return the canonical checkpoint path for (model_key, fold, kind).      Args:, Glob pattern matching every epoch checkpoint of one (key, fold) series.      Use, _validate_model_key() (+24 more)

### Community 5 - "Data Extraction & LFS"
Cohesion: 0.12
Nodes (26): check_data_status(), _discover_zip_files(), extract_all_data(), _extract_zip(), _generate_annotations(), _is_data_already_extracted(), _is_lfs_pointer(), _process_extracted_contents() (+18 more)

### Community 6 - "Graphify Skill Docs"
Cohesion: 0.09
Nodes (25): Add URL to Corpus (/graphify add), MCP Server (graphify.serve), Wiki Export (--wiki), Confidence Score Rubric, Node ID Format Rules, Extraction Subagent Prompt Template, GitHub Clone and Cross-Repo Merge, Native CLAUDE.md Integration (+17 more)

### Community 7 - "Engineering Guide & Preprocessing"
Cohesion: 0.23
Nodes (12): UB-01: Preprocessing Never Invoked (FIXED@03e6dbb), preprocess_all_data(), Offline Data Preprocessing Script ================================= Pre-computes, normalize_thermal(), preprocess_mask(), preprocess_thermal_image(), ndarray, Utilities for Data Processing ============================= Shared functions for (+4 more)

### Community 8 - "TransUNet Architecture"
Cohesion: 0.10
Nodes (12): CNNEncoder, MLP, MultiHeadAttention, CNN encoder for feature extraction (ResNet-50 style), Multi-head self-attention mechanism, TransUNet: Transformer-CNN Hybrid Architecture, MLP block for transformer, Transformer encoder block (+4 more)

### Community 10 - "NaN Debug Scripts (Ad-hoc)"
Cohesion: 0.06
Nodes (25): get_latest_checkpoint(), main(), Path, Inference & Model Comparison Script =================================== Loads th, Search for a model checkpoint across standard locations., DoubleConv, Thermal Facial Region Detection System - U-Net Model ===========================, Double convolution block for U-Net (+17 more)

### Community 11 - "Pipeline Stages (main_pipeline)"
Cohesion: 0.12
Nodes (15): main(), Pipeline, Run offline preprocessing when its outputs are missing (UB-01, T1.1).          T, Load annotations once (lightweight) — DataLoaders are created lazily., Create DataLoaders for a single model + fold (lazy, on-demand).          Uses ``, Shut down DataLoader workers and free GPU memory between runs., Train a dynamic model from the registry on a specific fold, Train all selected models sequentially over all folds (+7 more)

### Community 12 - "Loss Functions (Dice/Combined)"
Cohesion: 0.19
Nodes (9): test_detector_initialization_and_prediction(), ndarray, Inference class for detecting facial regions in thermal images, Normalize thermal image to [0, 1] range, Predict facial regions in a thermal image          Args:             thermal_ima, Calculate statistics information for each region using original thermal data in, Visualize predicted regions, Print a formatted report of thermal statistics for all regions (+1 more)

### Community 13 - "Multi-Dataset Data Loading"
Cohesion: 0.20
Nodes (6): ndarray, Load thermal image from TIFF file, Get the actual TIFF path for a given sample ID, handling variations, Load thermal image for a given sample ID                  Args:             samp, Crop thermal image to bounding box region, Create segmentation mask from polygon annotations

### Community 14 - "ModelBenchmark Runner"
Cohesion: 0.12
Nodes (17): ModelBenchmark, DataFrame, DataLoader, Module, Path, Generate comparison visualizations and reports, Create comprehensive comparison plots, Generate detailed text report (+9 more)

### Community 15 - "E2E Smoke Test Gate"
Cohesion: 0.10
Nodes (24): DataFrame, E2E smoke test — the merge gate (CLAUDE.md §7.3).  This test is the single defin, Invoke the real pipeline entry point in a subprocess.      Runs ``codes/main_pip, Return the most recently modified CSV matching *glob_pattern*.      Searched rel, Full end-to-end pipeline smoke (UB-01/02/03/05/07 guard).      Smoke artifacts p, read_latest(), run_pipeline_subprocess(), test_full_pipeline_smoke() (+16 more)

### Community 16 - "Synthetic Test Fixture"
Cohesion: 0.28
Nodes (8): _build_subject(), Shared pytest fixtures for UBench test suite.  Key fixture: ``synthetic_dataset`, Session-scoped synthetic dataset root (see CLAUDE.md §7.2).      Layout::, Return 4-corner polygon [[x,y], ...] for a rectangle., Create all synthetic files for one subject S{subject_n}., _rect_polygon(), synthetic_dataset(), TempPathFactory

### Community 17 - "CI & Phase 0 Kickoff"
Cohesion: 0.29
Nodes (7): GitHub Actions CI Workflow, Ruff Lint Gate (CI), Pytest Test Gate (CI), Phase 0 Kickoff Prompt (Session 0), T0.1 Synthetic Fixture + Smoke Test Task, T0.2 Repo Plumbing + CI Task, strict xfail Pattern for UB-01 Smoke Test

### Community 18 - "Shell Entry Points"
Cohesion: 0.60
Nodes (5): main(), run_extract(), run_pipeline(), run_setup(), run.sh script

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

### Community 36 - "hardware_detector.py"
Cohesion: 0.06
Nodes (39): ML & Benchmark Methodology Standards M1-M9, Non-Negotiable Engineering Rules R1-R10, Known-Defect Ledger (UB-01..UB-23), Leave-Subjects-Out Cross-Validation (GroupKFold), Phased Task Plan (Phase 0-4), E2E Smoke Test Merge Gate (strict xfail frontier), Synthetic Dataset Fixture (S1-S5, 64x64 uint16), UB-02: Checkpoint Filename Contract Mismatch (current frontier) (+31 more)

### Community 40 - "Config"
Cohesion: 0.14
Nodes (11): test_discovery_and_loading(), Config, MultiDirectoryDataLoader, Validate required data paths exist, Create output directories if they don't exist, Data loader that automatically discovers and loads from multiple dataset directo, Discover all Sx directories in data folder                  Returns:, Load annotations for a specific dataset directory                  Args: (+3 more)

### Community 41 - "unified_data.py"
Cohesion: 0.15
Nodes (20): Comprehensive Model Benchmarking Suite ====================================== Co, Main Training Pipeline Orchestrator =================================== Manages, create_kfold_data_loaders(), create_single_fold_loader(), load_split_metadata(), DataLoader, Unified Data Loading Module - Multi-Directory Support ==========================, Convert raw thermal sensor value to degrees Celsius.      Defined at module leve (+12 more)

### Community 42 - "Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session"
Cohesion: 0.20
Nodes (9): End-of-session report — same format, plus:, Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session, Standing guardrails (new + repeated), Step 0 — Environment & gates baseline, Step 1 — RED: `codes/tests/test_filenames.py`, Step 2 — GREEN: `codes/naming.py` + wiring both sides, Step 3 — Marker removal is atomic with the fix, Step 4 — Docs & ledger (same commit or trailing docs commit) (+1 more)

### Community 43 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 44 - "Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01)"
Cohesion: 0.22
Nodes (8): End-of-session report — same format as Phase 0, plus:, Hard scope exclusions this session, Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01), Step 0 — Environment & branch, Step 1 — Commit 1 (docs only): make CLAUDE.md true again, Step 2 — Commit 2: UB-23 — real CPU mode; delete the test-only bypass, Step 3 — Commit 3: UB-20 (partial) — pin albumentations, Step 4 — Commit 4: T1.1 / UB-01 — wire preprocessing into the pipeline

### Community 47 - "ThermalFaceDataset"
Cohesion: 0.29
Nodes (4): DataFrame, Unified PyTorch Dataset reading from offline preprocessed arrays, ThermalFaceDataset, Dataset

### Community 48 - "Session 0 — Build the safety net (Phase 0 of CLAUDE.md)"
Cohesion: 0.29
Nodes (6): End-of-session report (required format), Scope lock — Phase 0 only (T0.1 + T0.2), Session 0 — Build the safety net (Phase 0 of CLAUDE.md), Step 0 — Environment & reconnaissance, Step 1 — T0.1: synthetic fixture + smoke test, Step 2 — T0.2: repo plumbing + CI

### Community 49 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 51 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 52 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 53 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 55 - ".__init__"
Cohesion: 0.50
Nodes (3): Initialize pipeline          Args:             models_to_train: List of model na, Set global seeds for reproducibility., seed_everything()

## Knowledge Gaps
- **166 isolated node(s):** `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)`, `Step 1 - Ensure graphify is installed` (+161 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UBench README (User Documentation)` connect `unified_data.py` to `UBench — Thermal Face Detection Benchmark Pipeline`, `NaN Debug Scripts (Ad-hoc)`, `Environment Setup (setup.py)`, `hardware_detector.py`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `Config` connect `Config` to `Model Registry & Swin-UNet++`, `Engineering Guide & Preprocessing`, `unified_data.py`, `NaN Debug Scripts (Ad-hoc)`, `Pipeline Stages (main_pipeline)`, `Loss Functions (Dice/Combined)`, `ModelBenchmark Runner`, `ThermalFaceDataset`, `.__init__`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Config` (e.g. with `ModelBenchmark` and `Pipeline`) actually correct?**
  _`Config` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Pipeline` (e.g. with `HardwareProfile` and `TeeLogger`) actually correct?**
  _`Pipeline` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `UnifiedTrainer` (e.g. with `Pipeline` and `Config`) actually correct?**
  _`UnifiedTrainer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Usage`, `What graphify is for` to the rest of the system?**
  _166 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Model Registry & Swin-UNet++` be split into smaller, more focused modules?**
  _Cohesion score 0.050816696914700546 - nodes in this community are weakly interconnected._
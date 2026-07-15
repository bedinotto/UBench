# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~41,542 words - fits in a single context window. You may not need a graph.

## Summary
- 488 nodes · 779 edges · 32 communities (29 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.64)
- Token cost: 12,000 input · 3,500 output

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
- Logging (TeeLogger)
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

## God Nodes (most connected - your core abstractions)
1. `Config` - 30 edges
2. `SetupManager` - 20 edges
3. `Pipeline` - 18 edges
4. `MultiDirectoryDataLoader` - 18 edges
5. `UnifiedTrainer` - 17 edges
6. `Graphify Full Pipeline` - 16 edges
7. `HardwareDetector` - 13 edges
8. `run_benchmark()` - 12 edges
9. `extract_all_data()` - 11 edges
10. `HardwareProfile` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `UBench Engineering & Remediation Guide`  [EXTRACTED]
  prompts/phase1_session1.md → CLAUDE.md
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [EXTRACTED]
  prompts/phase1_session1.md → CLAUDE.md
- `Automatic Offline Preprocessing (--force-preprocess)` --conceptually_related_to--> `UB-01: Preprocessing Never Invoked (FIXED@03e6dbb)`  [INFERRED]
  README.md → CLAUDE.md
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `UB-02: Checkpoint Filename Contract Mismatch (current frontier)`  [EXTRACTED]
  prompts/phase1_session1.md → CLAUDE.md
- `Phase 1 Session 1 Plan (UB-23, UB-20 pin, T1.1)` --references--> `albumentations>=1.4,<2.0 Pin`  [EXTRACTED]
  prompts/phase1_session1.md → requirements/requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Extraction Pipeline (AST + Semantic + Cache)** — _claude_skills_graphify_skill_md_ast_extraction, _claude_skills_graphify_skill_md_semantic_extraction, _claude_skills_graphify_skill_md_extraction_cache [EXTRACTED 1.00]
- **UB-01 Fix Chain (T1.1 auto-preprocessing)** — claude_ub01_missing_preprocessing, codes_main_pipeline, codes_preprocess_data, readme_automatic_offline_preprocessing, claude_smoke_test_gate [INFERRED 0.85]
- **UB-23 CPU Enablement (UBENCH_ALLOW_CPU)** — claude_ub23_cpu_hard_exit, claude_ubench_allow_cpu, codes_hardware_detector, codes_tests_test_hardware_cpu [INFERRED 0.85]

## Communities (32 total, 3 thin omitted)

### Community 0 - "Benchmark & Orchestration Layer"
Cohesion: 0.07
Nodes (41): UB-02: Checkpoint Filename Contract Mismatch (current frontier), Comprehensive Model Benchmarking Suite ====================================== Co, get_latest_checkpoint(), main(), Path, Inference & Model Comparison Script =================================== Loads th, Search for a model checkpoint across standard locations., Main Training Pipeline Orchestrator =================================== Manages (+33 more)

### Community 1 - "Model Registry & Swin-UNet++"
Cohesion: 0.06
Nodes (29): create_model(), get_registered_models(), Module, Model Registry ============== Dynamically load and register model architectures., Decorator to register a model class, Return a list of registered model names, Instantiate a model by name, register_model() (+21 more)

### Community 2 - "Hardware Detection & CPU Mode"
Cohesion: 0.07
Nodes (28): UB-23: Hardware Detector CPU Hard Exit (FIXED@af14b13), UBENCH_ALLOW_CPU Opt-in CPU Mode, detect_and_optimize(), HardwareDetector, HardwareProfile, Hardware Detection and Optimization Module =====================================, Convert to dictionary, Detect and validate hardware capabilities (+20 more)

### Community 3 - "Environment Setup (setup.py)"
Cohesion: 0.09
Nodes (17): Upgrade pip to latest version, Detect the CUDA driver version using nvidia-smi.         Returns a version tuple, Return True if an already-installed PyTorch build has CUDA support.         Uses, Install the correct CUDA-enabled PyTorch build.         Plain `pip install torch, Manages cross-platform setup and dependency installation, Install non-PyTorch dependencies from requirements.txt, Install all dependencies (PyTorch CUDA first, then the rest), Check CUDA availability using a subprocess.          Using a subprocess (rather (+9 more)

### Community 4 - "Training Loop Internals"
Cohesion: 0.11
Nodes (16): DataLoader, Path, Unified training pipeline for all models     Ensures consistent training, valida, Initialize trainer          Args:             model: PyTorch model to train, Return a GradScaler when the active GPU can benefit from AMP, else None., Return the expected path for a given epoch checkpoint., Scan the checkpoint directory for the most recent epoch file., Save a full training checkpoint at the end of *epoch* (0-indexed).          The (+8 more)

### Community 5 - "Data Extraction & LFS"
Cohesion: 0.13
Nodes (24): check_data_status(), _discover_zip_files(), extract_all_data(), _extract_zip(), _generate_annotations(), _is_data_already_extracted(), _is_lfs_pointer(), _process_extracted_contents() (+16 more)

### Community 6 - "Graphify Skill Docs"
Cohesion: 0.09
Nodes (25): Add URL to Corpus (/graphify add), MCP Server (graphify.serve), Wiki Export (--wiki), Confidence Score Rubric, Node ID Format Rules, Extraction Subagent Prompt Template, GitHub Clone and Cross-Repo Merge, Native CLAUDE.md Integration (+17 more)

### Community 7 - "Engineering Guide & Preprocessing"
Cohesion: 0.11
Nodes (23): ML & Benchmark Methodology Standards M1-M9, Non-Negotiable Engineering Rules R1-R10, Known-Defect Ledger (UB-01..UB-23), Leave-Subjects-Out Cross-Validation (GroupKFold), Phased Task Plan (Phase 0-4), UB-01: Preprocessing Never Invoked (FIXED@03e6dbb), UB-20: Dependency & Reproducibility Hygiene, UBench Engineering & Remediation Guide (+15 more)

### Community 8 - "TransUNet Architecture"
Cohesion: 0.10
Nodes (12): CNNEncoder, MLP, MultiHeadAttention, CNN encoder for feature extraction (ResNet-50 style), Multi-head self-attention mechanism, TransUNet: Transformer-CNN Hybrid Architecture, MLP block for transformer, Transformer encoder block (+4 more)

### Community 9 - "Logging (TeeLogger)"
Cohesion: 0.13
Nodes (11): Console Logger (Tee) ==================== Redirects sys.stdout and sys.stderr so, Intercepts sys.stdout and sys.stderr and mirrors them to a log file.      Parame, Start a TeeLogger by reading the log directory from the     ``UBENCH_LOG_DIR`` e, A file-like object that writes to two streams simultaneously.     One stream is, Prefix every complete line with a timestamp., start_from_env(), TeeLogger, _TeeStream (+3 more)

### Community 10 - "NaN Debug Scripts (Ad-hoc)"
Cohesion: 0.12
Nodes (7): DoubleConv, Thermal Facial Region Detection System - U-Net Model ===========================, Double convolution block for U-Net, U-Net architecture for semantic segmentation, UNet, CombinedLoss, Combined Cross-Entropy and Dice Loss.      The forward pass is wrapped with auto

### Community 11 - "Pipeline Stages (main_pipeline)"
Cohesion: 0.13
Nodes (13): main(), Pipeline, Run offline preprocessing when its outputs are missing (UB-01, T1.1).          T, Load annotations once (lightweight) — DataLoaders are created lazily., Create DataLoaders for a single model + fold (lazy, on-demand).          Uses ``, Shut down DataLoader workers and free GPU memory between runs., Train a dynamic model from the registry on a specific fold, Train all selected models sequentially over all folds (+5 more)

### Community 12 - "Loss Functions (Dice/Combined)"
Cohesion: 0.12
Nodes (12): test_detector_initialization_and_prediction(), DiceLoss, Module, ndarray, Dice Loss for segmentation.      NOTE: softmax on fp16 logits overflows (exp of, Inference class for detecting facial regions in thermal images, Normalize thermal image to [0, 1] range, Predict facial regions in a thermal image          Args:             thermal_ima (+4 more)

### Community 13 - "Multi-Dataset Data Loading"
Cohesion: 0.13
Nodes (12): test_discovery_and_loading(), MultiDirectoryDataLoader, ndarray, Data loader that automatically discovers and loads from multiple dataset directo, Discover all Sx directories in data folder                  Returns:, Load annotations for a specific dataset directory                  Args:, Load annotations from all discovered dataset directories, Load thermal image from TIFF file (+4 more)

### Community 14 - "ModelBenchmark Runner"
Cohesion: 0.17
Nodes (13): ModelBenchmark, DataFrame, DataLoader, Module, Path, Generate comparison visualizations and reports, Create comprehensive comparison plots, Comprehensive benchmark for model comparison (+5 more)

### Community 15 - "E2E Smoke Test Gate"
Cohesion: 0.20
Nodes (11): E2E Smoke Test Merge Gate (strict xfail frontier), Synthetic Dataset Fixture (S1-S5, 64x64 uint16), DataFrame, E2E smoke test — the merge gate (CLAUDE.md §7.3).  This test is the single defin, Full end-to-end pipeline smoke (UB-01/02/03/05/07 guard).      Once UB-02 is fix, Invoke the real pipeline entry point in a subprocess.      Runs ``codes/main_pip, Return the most recently modified CSV matching *glob_pattern*.      Searched rel, read_latest() (+3 more)

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

## Knowledge Gaps
- **35 isolated node(s):** `Graphify Skill Reference`, `Community Detection`, `Fast Path Query (Existing Graph)`, `Extraction Cache (semantic cache)`, `Add URL to Corpus (/graphify add)` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Benchmark & Orchestration Layer` to `Model Registry & Swin-UNet++`, `Training Loop Internals`, `Engineering Guide & Preprocessing`, `NaN Debug Scripts (Ad-hoc)`, `Pipeline Stages (main_pipeline)`, `Loss Functions (Dice/Combined)`, `Multi-Dataset Data Loading`, `ModelBenchmark Runner`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **Why does `SetupManager` connect `Environment Setup (setup.py)` to `Logging (TeeLogger)`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `UBench README (User Documentation)` connect `Benchmark & Orchestration Layer` to `Logging (TeeLogger)`, `Hardware Detection & CPU Mode`, `Engineering Guide & Preprocessing`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Config` (e.g. with `ModelBenchmark` and `Pipeline`) actually correct?**
  _`Config` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Pipeline` (e.g. with `HardwareProfile` and `TeeLogger`) actually correct?**
  _`Pipeline` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `UnifiedTrainer` (e.g. with `Pipeline` and `Config`) actually correct?**
  _`UnifiedTrainer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Graphify Skill Reference`, `Community Detection`, `Fast Path Query (Existing Graph)` to the rest of the system?**
  _35 weakly-connected nodes found - possible documentation gaps or missing edges._
# Graph Report - .  (2026-08-09)

## Corpus Check
- 152 files · ~195,325 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1367 nodes · 2447 edges · 95 communities (78 shown, 17 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 94 edges (avg confidence: 0.65)
- Token cost: 190,136 input · 0 output

## Community Hubs (Navigation)
- Thermal Augmentation & Lateral Flip
- Landmark Mask Derivation
- Checkpoint Naming Authority
- Phase 5 Post-Run Defects
- Model Registry & Pretrained Stems
- CI & Reproducibility Commits
- Fixed-Batch VRAM Probe
- Environment Setup & CUDA Install
- Swin-UNet++ Architecture
- Pipeline Orchestrator Entry
- Segmentation Metric Authority
- Hardware Detection & Scaling
- Pipeline Class Lifecycle
- Benchmark & Data Loading
- Optimizer & Scheduler Schema
- Preprocess Manifest Guard
- TransUNet Encoder & Attention
- Console Tee Logger
- Model Benchmark Runner
- Reported Run Results
- Synchronized Inference Timing
- Combined Loss Configuration
- Crop Origin & Multi-Dataset Loading
- Phase 2 Trustworthy Numbers
- Held-Out Test Subjects
- Run Provenance & Commits
- Config Loading & Validation
- Normalization & Determinism Tests
- Data Extraction & Annotations
- Mask Offset Regression Tests
- Landmark Scheme Census Script
- Metric Consistency Fix
- Typed Config Schema
- Leave-Subjects-Out Split Tests
- Phase 3 Credible Science
- Training Config Keys & Env
- Run Identity & Epochs Tests
- Synthetic Fixture & Validate
- Phase 0-1 Safety Net
- Failure Honesty Tests
- Engineering Rules & Batch Keys
- Dice & Combined Loss
- Thermal Stats & Visualization
- Per-Subject Wilcoxon Script
- Qualitative Figures Script
- Fold-Count Guard
- Run Identity Defects
- U-Net Decoder Blocks
- Batch-Size Key Tests
- Pretrained Decoder Blocks
- Latency Measurement Script
- Config Drift Consolidation
- Mask Alignment Defect
- Per-Family Recipe Resolution
- Thermal Dataset Loading
- CLAUDE.md Governance Sections
- Filename Contract Defect
- Hardware Profile Sizing
- Offline Preprocessing Script
- Virtualenv & uv Detection
- Normalization Commutativity Tests
- Resume Tests
- Mask Defect Census Script
- Thermal Preprocessing Standard
- Load-Time Normalization
- Pytest Fixtures
- Distance Stratification Script
- U-Net Architecture
- Shell Entry Point
- Pretraining Policy Defects
- End-to-End Smoke Gate
- Pareto Frontier Script
- Warmup-Cosine Split Tests
- Output Directory Validation
- Landmark Side Evidence Script
- Model CLI Choices
- LIMIT_SAMPLES Override
- Convergence Figures Script
- Epochs Override Helper
- Analysis Scripts README
- Hardware Detect Entry
- Error Log Artifact
- Fail-Fast Flag
- Hardware Profile Artifact
- Thermal Normalization Helper
- Latest Run Symlink
- Legacy Min-Max Normalization
- Pipeline Log Artifact
- UBench Project Root
- Model Registration Decorator
- Legacy Requirements File
- Running SHA Record

## God Nodes (most connected - your core abstractions)
1. `Config` - 34 edges
2. `UnifiedTrainer` - 33 edges
3. `OptimizerConfig` - 28 edges
4. `SchedulerConfig` - 28 edges
5. `RecipesConfig` - 23 edges
6. `SetupManager` - 23 edges
7. `SegmentationMetrics` - 21 edges
8. `Pipeline` - 19 edges
9. `checkpoint_path()` - 18 edges
10. `MultiDirectoryDataLoader` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Phase 5 — Post-run corrections` ----> `frozen run artifacts`  [EXTRACTED]
  CLAUDE.md → data/scripts/README.md
- `data/scripts/_common.py` ----> `UB-02 — checkpoint filename contract mismatch`  [EXTRACTED]
  data/scripts/README.md → CLAUDE.md
- `recipes.model_families` ----> `resolve_recipe`  [EXTRACTED]
  codes/config.yaml → CLAUDE.md
- `probe_vram()` --calls--> `probe_peak_memory()`  [EXTRACTED]
  data/scripts/a07_fig_pareto_frontier.py → codes/benchmark_models.py
- `main()` --calls--> `probe_peak_memory()`  [EXTRACTED]
  data/scripts/a10_vram_isolated_probe.py → codes/benchmark_models.py

## Import Cycles
- None detected.

## Communities (95 total, 17 thin omitted)

### Community 0 - "Thermal Augmentation & Lateral Flip"
Cohesion: 0.07
Nodes (44): build_thermal_transform(), lateral_index_pairs(), LateralAwareHorizontalFlip, ndarray, Physically-plausible thermal augmentation (T3.4/M7).  Applied in **Celsius**, be, Build the training augmentation pipeline from an ``AugmentationConfig``.      Si, Additive per-image sensor drift + per-pixel Gaussian noise, in °C.      Sensor d, Class-index pairs whose semantics swap under a horizontal flip.      Derived fro (+36 more)

### Community 1 - "Landmark Mask Derivation"
Cohesion: 0.08
Nodes (47): _count_filled_landmarks(), _declared_landmark_columns(), _drop_occluded_side(), generate_all(), generate_bounding_boxes(), generate_polygonal_masks(), Return *mapping* without the lateral regions on the occluded side.      The occl, Count the ``x{i}`` landmark columns a CSV *declares* in its header. (+39 more)

### Community 2 - "Checkpoint Naming Authority"
Cohesion: 0.06
Nodes (35): checkpoint_path(), epoch_checkpoint_glob(), Path, Single authority for checkpoint file naming (UB-02, R5).  Registry keys (``unet`, Reject anything that is not a lowercase snake_case registry key.      Display na, Return the canonical checkpoint path for (model_key, fold, kind).      Args:, Glob pattern matching every epoch checkpoint of one (key, fold) series.      Use, _validate_model_key() (+27 more)

### Community 3 - "Phase 5 Post-Run Defects"
Cohesion: 0.07
Nodes (41): commit 831c04d, commit 9376336, commit 9712d2c, LANDMARK_MAPPINGS_43, LateralAwareHorizontalFlip, M8 — Statistical claims, Phase 5 — Post-run corrections, T4.1 — Wilcoxon paired test in report generator (+33 more)

### Community 4 - "Model Registry & Pretrained Stems"
Cohesion: 0.10
Nodes (28): create_model(), Module, Decorator to register a model class, Instantiate a model by name, register_model(), Tensor, Shared 1-channel stem adaptation for pretrained (RGB) encoders (M5).  Pretrained, Resolve the pretrained flag shared by the pretrained encoders (M5).      An expl (+20 more)

### Community 5 - "CI & Reproducibility Commits"
Cohesion: 0.07
Nodes (32): .github/workflows/ci.yml, commit 0cc7724, commit 0d12ca1, commit 1800abf, commit 3b5d69e, commit 5991f60, M6 — Reproducibility, R2 — Test first (+24 more)

### Community 6 - "Fixed-Batch VRAM Probe"
Cohesion: 0.14
Nodes (24): probe_peak_memory(), Inference VRAM (MB) for one forward on a FIXED synthetic batch.      Two compara, _patch_cuda(), MonkeyPatch, Fixed-batch, isolation-safe VRAM probe tests (UB-10 T2.3; UB-28 T5.2).  Two dist, On CPU there is nothing to measure: return None, render 'n/a (CPU)'., The forward sees a synthetic (FIXED_BATCH, 1, H, W) tensor., The probe batch comes from its own argument, not the model's training batch. (+16 more)

### Community 7 - "Environment Setup & CUDA Install"
Cohesion: 0.10
Nodes (15): Verify Git LFS is installed and large files are downloaded, Upgrade pip to latest version, Detect the CUDA driver version using nvidia-smi.         Returns a version tuple, Manages cross-platform setup and dependency installation, Resolve the torch wheel backend: 'cpu' (default) or a CUDA tag.          Honors, Check CUDA availability using a subprocess.          Using a subprocess (rather, Confirm torch imports in the target interpreter (CPU path).          The CPU ins, Create required directory structure (+7 more)

### Community 8 - "Swin-UNet++ Architecture"
Cohesion: 0.09
Nodes (17): NestedConvBlock, PatchEmbed, PatchMerging, Thermal Facial Region Detection System - Swin-UNet++ ===========================, Patch Merging Layer for downsampling, Image to Patch Embedding, Nested convolution block for dense skip connections, Swin-UNet++ architecture with nested dense skip connections (+9 more)

### Community 9 - "Pipeline Orchestrator Entry"
Cohesion: 0.11
Nodes (26): apply_epochs_override(), _git(), main(), _model_choices(), Path, Main Training Pipeline Orchestrator =================================== Manages, Export ``--epochs`` to the ``NUM_EPOCHS`` env var Config reads (UB-13).      Exp, Persist *text* to a timestamped error log and return its path.      Single autho (+18 more)

### Community 10 - "Segmentation Metric Authority"
Cohesion: 0.11
Nodes (23): compute_segmentation_metrics(), Tensor, Segmentation metric authority (UB-11, T2.2).  One definition per metric, shared, One-shot metrics for a single ``(logits, target)`` pair.      Convenience wrappe, Accumulate a confusion matrix and derive hard IoU / Dice from it.      Usage mir, Clear all accumulated state., Accumulate one batch.          Args:             logits_or_preds: either raw log, Return hard IoU/Dice with macro (excl. absent) and background split.          Re (+15 more)

### Community 11 - "Hardware Detection & Scaling"
Cohesion: 0.10
Nodes (19): detect_and_optimize(), HardwareDetector, Hardware Detection and Optimization Module =====================================, Detect and validate hardware capabilities, Detect hardware and create profile, Build a CPU-only profile (explicit opt-in via UBENCH_ALLOW_CPU=1).          For, Detect GPU name and memory, Validate GPU meets minimum requirements (+11 more)

### Community 12 - "Pipeline Class Lifecycle"
Cohesion: 0.10
Nodes (17): Pipeline, Main training pipeline orchestrator, Initialize pipeline          Args:             models_to_train: List of model na, Point ``outputs/latest`` at this run's output dir (UB-06).          The replacem, Run offline preprocessing when its outputs are missing (UB-01, T1.1).          T, Create DataLoaders for a single model + fold (lazy, on-demand).          Uses ``, Shut down DataLoader workers and free GPU memory between runs., Train a dynamic model from the registry on a specific fold (+9 more)

### Community 13 - "Benchmark & Data Loading"
Cohesion: 0.15
Nodes (21): Comprehensive Model Benchmarking Suite ====================================== Co, Config, create_kfold_data_loaders(), create_single_fold_loader(), load_split_metadata(), load_test_metadata(), DataFrame, DataLoader (+13 more)

### Community 14 - "Optimizer & Scheduler Schema"
Cohesion: 0.18
Nodes (21): OptimizerConfig, LR scheduler recipe (``scheduler:`` section or a per-family override).      ``na, Optimizer recipe (``optimizer:`` section or a per-family override).      ``name`, SchedulerConfig, test_single_train_step(), _cfg(), Per-family recipes + scheduler cadence (T3.3, UB-18, M4).  Covers recipe resolut, Records every step() call and its args; stands in for the real scheduler. (+13 more)

### Community 15 - "Preprocess Manifest Guard"
Cohesion: 0.13
Nodes (23): Path, Processed-data schema manifest (T3.4/M7).  A standalone module (no dependency on, Load and validate the manifest next to ``metadata.csv`` (R4/M7).      Raises:, Write the schema manifest recording version + provenance.      ``normalization``, verify_preprocess_manifest(), write_preprocess_manifest(), Thermal normalization + processed-data manifest guard (T3.4, UB-19, M7).  Design, test_apply_normalization_unknown_mode_raises() (+15 more)

### Community 16 - "TransUNet Encoder & Attention"
Cohesion: 0.10
Nodes (12): CNNEncoder, MLP, MultiHeadAttention, CNN encoder for feature extraction (ResNet-50 style), Multi-head self-attention mechanism, TransUNet: Transformer-CNN Hybrid Architecture, MLP block for transformer, Transformer encoder block (+4 more)

### Community 17 - "Console Tee Logger"
Cohesion: 0.13
Nodes (11): Console Logger (Tee) ==================== Redirects sys.stdout and sys.stderr so, Intercepts sys.stdout and sys.stderr and mirrors them to a log file.      Parame, Start a TeeLogger by reading the log directory from the     ``UBENCH_LOG_DIR`` e, A file-like object that writes to two streams simultaneously.     One stream is, Prefix every complete line with a timestamp., start_from_env(), TeeLogger, _TeeStream (+3 more)

### Community 18 - "Model Benchmark Runner"
Cohesion: 0.15
Nodes (17): evaluate_accuracy(), _format_vram(), ModelBenchmark, DataFrame, DataLoader, Module, Path, Render a VRAM figure, or 'n/a (CPU)' when it was not measured. (+9 more)

### Community 19 - "Reported Run Results"
Cohesion: 0.13
Nodes (22): M9 — Honest reporting, UBENCH_PRETRAINED, Swin-UNet++ reported results, TransUNet reported results, U-Net reported results, recipes.model_families, config recipes section, codes/pretrained_stem.py (+14 more)

### Community 20 - "Synchronized Inference Timing"
Cohesion: 0.14
Nodes (15): Measure per-image inference latency honestly (UB-09, M3).      Two correctness p, timed_inference(), _fake_loader(), _FakeImages, _FakeModel, MonkeyPatch, Timing honesty tests (UB-09, T2.1).  Two guarantees:  * ``benchmark_models.timed, CPU device → synchronize() is never called (nothing to synchronize). (+7 more)

### Community 21 - "Combined Loss Configuration"
Cohesion: 0.15
Nodes (20): LossConfig, Combined Cross-Entropy + Dice loss (``loss:`` section).      ``class_weights`` c, _make_trainer(), UB-12 (T3.1): the single validated config authority.  ``codes/config.yaml`` is l, Build a UnifiedTrainer on CPU with a lightweight stand-in config.      Exercises, test_class_weights_bad_string_raises(), test_class_weights_balanced_is_nonuniform(), test_class_weights_explicit_list() (+12 more)

### Community 22 - "Crop Origin & Multi-Dataset Loading"
Cohesion: 0.12
Nodes (13): crop_to_bbox reports the origin it actually used — clamped, never     negative —, test_crop_to_bbox_returns_clamped_origin(), MultiDirectoryDataLoader, ndarray, Data loader that automatically discovers and loads from multiple dataset directo, Discover all Sx directories in data folder                  Returns:, Load annotations for a specific dataset directory                  Args:, Load annotations from all discovered dataset directories (+5 more)

### Community 23 - "Phase 2 Trustworthy Numbers"
Cohesion: 0.12
Nodes (21): commit 3bbe8df, M1 — Splits, M3 — Timing, Phase 2 — Trustworthy numbers (P1), R10 — Honest AI collaboration, T2.1 — honest timing, T2.3 — fixed-batch VRAM probe, T2.4 — held-out test subjects (+13 more)

### Community 24 - "Held-Out Test Subjects"
Cohesion: 0.14
Nodes (20): _cfg(), Held-out test-subject tests (M1, T2.4).  Subjects listed in ``config.test_subjec, A test subject not present in the data is a hard error, not a silent no-op., With no test subjects: full CV pool, empty test set, no test loader., Fabricate data/processed/metadata.csv for the given subjects., No fold's train or val split may contain a held-out test subject., The test loader contains exactly the configured test subjects, nothing else., Reserving subjects that leave <2 in the CV pool raises the >=2 guard. (+12 more)

### Community 25 - "Run Provenance & Commits"
Cohesion: 0.12
Nodes (20): commit 01244fa, commit 11b64d5, commit 5a5b4b8, tree 885fc55 (codes/), R1 — Verify empirically, R8 — Security & safety defaults, T1.7 — resume, TEST_SUBJECTS (+12 more)

### Community 26 - "Config Loading & Validation"
Cohesion: 0.12
Nodes (20): load_config(), Load and validate ``codes/config.yaml`` into a :class:`RootConfig`.      Args:, Path, A non-'balanced' string for class_weights fails schema validation., Write ``data`` as a config.yaml under ``tmp_path`` and return its path., A typo'd key inside a section is a hard error, not a silent no-op., An unknown top-level section raises., A wrong-typed value raises instead of crashing deep in the pipeline. (+12 more)

### Community 27 - "Normalization & Determinism Tests"
Cohesion: 0.14
Nodes (16): PreprocessingConfig, Thermal-domain preprocessing (``preprocessing:`` section, T3.4/M7).      ``norma, _cfg_with_metadata(), _NoisyDataset, Dataset, Reproducibility tests (UB-20a, M6, T2.5).  Two seeded runs must produce identica, create_single_fold_loader sets a seeded generator and seed_worker (no iteration), seed_worker gives the same seed for a given worker id, different across ids. (+8 more)

### Community 28 - "Data Extraction & Annotations"
Cohesion: 0.18
Nodes (18): check_data_status(), _discover_zip_files(), extract_all_data(), _extract_zip(), _generate_annotations(), _is_data_already_extracted(), _is_lfs_pointer(), _process_extracted_contents() (+10 more)

### Community 29 - "Mask Offset Regression Tests"
Cohesion: 0.17
Nodes (18): _assert_aligned(), _centroid(), _expected_mask(), offset_dataset(), MonkeyPatch, ndarray, Path, UB-08: polygon/mask offsets must use the *clamped* crop origin.  ``crop_to_bbox` (+10 more)

### Community 30 - "Landmark Scheme Census Script"
Cohesion: 0.16
Nodes (17): main(), A1 — Conta imagens no caminho de 43 vs 73 marcos faciais (landmarks).  Bug do ra, build_model(), classify_landmarks(), _ensure_models_registered(), landmark_files(), landmark_xy_columns(), load_fold_model() (+9 more)

### Community 31 - "Metric Consistency Fix"
Cohesion: 0.15
Nodes (17): commit 3260638, CombinedLoss, M2 — Metrics, SegmentationMetrics, T2.2 — one metric authority, UB-11 — inconsistent metric definitions, UnifiedTrainer, loss.class_weights (+9 more)

### Community 32 - "Typed Config Schema"
Cohesion: 0.15
Nodes (16): BaseModel, FamilyRecipe, ModelConfig, PathsConfig, Typed schema for the single UBench config (``codes/config.yaml``).  UB-12 (T3.1), Per-family optimizer/scheduler override (``recipes.families.<name>``).      A ``, Top-level config document (the whole of ``codes/config.yaml``)., Base model that rejects unknown keys so typos raise (R4). (+8 more)

### Community 33 - "Leave-Subjects-Out Split Tests"
Cohesion: 0.15
Nodes (16): _metadata_frame(), DataFrame, UB-03/UB-04: fold-count guard with leave-subjects-out semantics.  ``GroupKFold(g, Tiny stand-in for metadata.csv: sample_id + dataset columns only., 3 subjects, K=5 → effective_k == 3, loud warning naming both., 2 subjects, K=5 → effective_k == 2 (the minimum viable CV)., 1 subject → clear leave-subjects-out error, not sklearn's., K <= n_groups → requested K unchanged, no warning emitted. (+8 more)

### Community 34 - "Phase 3 Credible Science"
Cohesion: 0.13
Nodes (16): M4 — Fairness, Phase 3 — Credible science (P2), Phase 4 — Enhancements, R3 — Minimal, surgical diffs, R9 — Documentation follows code, T3.1 — one validated config, T3.3 — per-family recipes, T3.7 — README reconciliation (+8 more)

### Community 35 - "Training Config Keys & Env"
Cohesion: 0.13
Nodes (15): CLAUDE.md §4 Architecture, NUM_EPOCHS, UBENCH_DETERMINISTIC, training.deterministic, training.k_folds, training.num_epochs, training.random_seed, config training section (+7 more)

### Community 36 - "Run Identity & Epochs Tests"
Cohesion: 0.16
Nodes (14): UB-13 + UB-14: the --epochs flag must always be honored, and one run identity mu, --epochs 2 with NUM_EPOCHS unset → exactly 2 epochs trained (UB-13 AC)., The exact value the old ``if args.epochs != 100`` guard dropped (UB-13)., No flag → no NUM_EPOCHS export; config stays the authority (UB-13)., run.sh must export its timestamp as UBENCH_RUN_ID (UB-14, grep-level)., Names of real run directories under outputs/ (symlinks excluded)., UBENCH_RUN_ID names the run's dirs; --resume overrides it (UB-14).      Phase 2, _run_dirs() (+6 more)

### Community 37 - "Synthetic Fixture & Validate"
Cohesion: 0.14
Nodes (10): Dataset, 4 tiny (1, 8, 8) frames with integer masks in ``[0, num_classes)``., validate() returns exactly (loss, mIoU, dice) and holds no timing state., test_validate_emits_no_timing(), _TinyDataset, Unified training pipeline for all models     Ensures consistent training, valida, Return a GradScaler when the active GPU can benefit from AMP, else None., Construct the optimizer from a resolved recipe (adam | adamw). (+2 more)

### Community 38 - "Phase 0-1 Safety Net"
Cohesion: 0.15
Nodes (14): commit 03e6dbb, commit 74e1b43, Phase 0 — Safety net, Phase 1 — Make ./run.sh work (P0), T0.1 — tests skeleton + synthetic fixture + smoke test, T0.2 — un-ignore tests, add __init__, CI + ruff, T1.1 — auto preprocessing, T1.6 — failure honesty (+6 more)

### Community 39 - "Failure Honesty Tests"
Cohesion: 0.21
Nodes (13): _error_logs_in_run_dir(), Path, UB-07: training failures must be recorded, reported, and fatal.  Before T1.6, th, error_log_*.txt inside the run's own log dir, parsed from the banner     so erro, Injected per-model failure → exit 1 + named summary + error log.      Without ``, --fail-fast: the first failure aborts the run — no later folds., A Pipeline() constructor crash reaches main()'s error-log writer.      An empty, _set_fast_env() (+5 more)

### Community 40 - "Engineering Rules & Batch Keys"
Cohesion: 0.17
Nodes (13): CLAUDE.md §6 Non-Negotiable Engineering Rules, R4 — No silent failure paths, T1.3 — canonical batch-size keys, UB-05 — batch-size key mismatch, UB-23 — hardware detector exits on CPU, commit af14b13, mixed precision (AMP), commit cd3afa4 (+5 more)

### Community 41 - "Dice & Combined Loss"
Cohesion: 0.18
Nodes (7): CombinedLoss, DiceLoss, DataLoader, Module, Initialize trainer          Args:             model: PyTorch model to train, Dice Loss for segmentation.      NOTE: softmax on fp16 logits overflows (exp of, Combined Cross-Entropy and Dice Loss.      The forward pass is wrapped with auto

### Community 42 - "Thermal Stats & Visualization"
Cohesion: 0.21
Nodes (8): ndarray, Visualize predicted regions, Print a formatted report of thermal statistics for all regions, Inference class for detecting facial regions in thermal images, Normalize via the single shared authority (R5/T3.4).          Routes to :func:`c, Predict facial regions in a thermal image          Args:             thermal_ima, Calculate statistics information for each region using original thermal data in, ThermalFaceDetector

### Community 43 - "Per-Subject Wilcoxon Script"
Cohesion: 0.22
Nodes (11): evaluate_subject(), main(), DataFrame, A3 — Avaliação por sujeito (n=10) + teste pareado de Wilcoxon.  Bugs do rascunho, Score one subject's images with one fold-model.      Returns the subject-level a, main(), A9 — Avaliação por sujeito na variante PRIMÁRIA (época final) → n = 10.  Por que, fold_subject_splits() (+3 more)

### Community 44 - "Qualitative Figures Script"
Cohesion: 0.24
Nodes (12): auto_select(), build_candidate_table(), lateral_anomaly_figure(), load_sample(), main(), DataFrame, ndarray, Tensor (+4 more)

### Community 45 - "Fold-Count Guard"
Cohesion: 0.21
Nodes (12): commit 73967c8, MultiDirectoryDataLoader, T1.4 — fold-count guard, UB-03 — GroupKFold raises when subjects < K, UB-04 — split semantics contradict docs, codes/unified_data.py, create_single_fold_loader, create_test_loader (+4 more)

### Community 46 - "Run Identity Defects"
Cohesion: 0.20
Nodes (12): T1.8 — run identity and --epochs, UB-13 — --epochs 100 ignored, UB-14 — two timestamps per run, UBENCH_RUN_ID, UnifiedTrainer._find_latest_checkpoint, commit add5229, codes/extract_data.py, --resume RUN_ID (+4 more)

### Community 47 - "U-Net Decoder Blocks"
Cohesion: 0.24
Nodes (6): _DoubleConv, Tensor, Bilinear-upsample ×2, concatenate the encoder skip, then DoubleConv., ImageNet-pretrained SwinV2-tiny encoder + U-Net-style conv decoder., SwinV2UNet, _Up

### Community 48 - "Batch-Size Key Tests"
Cohesion: 0.23
Nodes (11): _profile(), UB-05: batch-size dicts keyed by canonical registry names, hard lookup.  The har, Simulate a hardware tier without any GPU present (pure Python)., Ledger reference value: the 6 GB (GTX 1660 Ti) tier gives Swin 6., Ledger reference value: the <5.5 GB tier gives Swin 3., Every tier's keys == the registered model keys (parity invariant)., The retired short key must raise, never silently default (R4)., test_batch_size_keys_match_registry() (+3 more)

### Community 49 - "Pretrained Decoder Blocks"
Cohesion: 0.23
Nodes (6): _DoubleConv, Tensor, Bilinear-upsample ×2 then DoubleConv (no skip — option (a))., ImageNet-pretrained R50+ViT-B/16 hybrid encoder + upsampling decoder., TransUNetPretrained, _UpConv

### Community 50 - "Latency Measurement Script"
Cohesion: 0.18
Nodes (9): main(), A2 — Reinstrumenta latência de inferência com lote fixo (batch=4).  Bugs do rasc, main(), A6 — Painéis de convergência (uma figura por arquitetura, 5 curvas de fold).  Nã, main(), A10 — Re-medição de VRAM com a sonda isolada (UB-28).  Motivo. Os valores de VRA, get_device(), load_config() (+1 more)

### Community 51 - "Config Drift Consolidation"
Cohesion: 0.18
Nodes (11): commit 72c59f4, Config, UB-12 — config drift, config model section, config optimizer section, config paths section, config scheduler section, codes/config.yaml (+3 more)

### Community 52 - "Mask Alignment Defect"
Cohesion: 0.18
Nodes (11): CLAUDE.md §7 Testing Doctrine, T1.5 — clamped crop origin, UB-08 — unclamped polygon offset shifts masks, commit cc890de, codes/preprocess_data.py, codes/tests/conftest.py, crop_to_bbox, data/processed/metadata.csv (+3 more)

### Community 53 - "Per-Family Recipe Resolution"
Cohesion: 0.25
Nodes (11): Per-family recipe assignments (``recipes:`` section, T3.3/UB-18/M4).      ``mode, Resolve the effective (optimizer, scheduler) recipe for a model (M4).      An un, RecipesConfig, resolve_recipe(), collect_run_metadata(), Collect per-run provenance (M6): git state, env, seed, and effective config., collect_run_metadata records git/env/seed/config provenance (M6)., test_run_metadata_has_provenance_keys() (+3 more)

### Community 54 - "Thermal Dataset Loading"
Cohesion: 0.24
Nodes (8): Dataset, Unified PyTorch Dataset reading from offline preprocessed arrays, ThermalFaceDataset, evaluate_subject_boundary(), main(), Tensor, A8 — HD95 e NSD (métricas de fronteira) — CORTÁVEL.  Não havia rascunho para est, to_onehot()

### Community 55 - "CLAUDE.md Governance Sections"
Cohesion: 0.20
Nodes (8): CLAUDE.md §10 Verification Gates & Definition of Done, CLAUDE.md §11 Anti-Patterns, CLAUDE.md §3 Commands & Environment, CLAUDE.md §5 Known-Defect Ledger, CLAUDE.md §9 Phased Task Plan, R6 — Conventional commits, one ledger item per commit/PR, UBench, Session Entry Protocol

### Community 56 - "Filename Contract Defect"
Cohesion: 0.22
Nodes (10): R5 — One source of truth per contract, T1.2 — naming.py checkpoint contract, UB-02 — checkpoint filename contract mismatch, UB-26 — hand-maintained --models choices, codes/main_pipeline.py, commit dc3fe29, commit dd5fa8d, run.bat (+2 more)

### Community 57 - "Hardware Profile Sizing"
Cohesion: 0.24
Nodes (5): HardwareProfile, Convert to dictionary, Hardware profile with optimization parameters, Calculate optimal batch sizes based on GPU memory.          Keys are the canonic, Calculate the optimal number of DataLoader prefetch workers.          DataLoader

### Community 58 - "Offline Preprocessing Script"
Cohesion: 0.31
Nodes (8): preprocess_all_data(), Offline Data Preprocessing Script ================================= Pre-computes, preprocess_mask(), preprocess_thermal_image(), ndarray, Utilities for Data Processing ============================= Shared functions for, Resize a thermal image to ``target_size`` (bilinear).      T3.4/design (i): norm, Resize mask image using nearest neighbor interpolation.

### Community 59 - "Virtualenv & uv Detection"
Cohesion: 0.20
Nodes (5): True when running inside a virtual environment (not system Python).          set, Return True if the ``uv`` resolver is available on PATH., Return the lockfile Path for *backend*, generating the CUDA lock if         abse, Install the full environment from the generated lockfile via uv.          Reprod, CUDA torch wheels are published only for Python 3.8–3.12.          With 3.13+, p

### Community 60 - "Normalization Commutativity Tests"
Cohesion: 0.24
Nodes (9): Why design (i) moves per_image_minmax numbers: min/max change under resize., test_per_image_minmax_does_not_commute_with_resize(), Thermal utility tests (UB-15, T2.6)., A flat image (min == max) normalizes to zeros, not the raw values (UB-15)., A varying image is min-max scaled into [0, 1]., test_normalize_thermal_flat_image_returns_zeros(), test_normalize_thermal_scales_to_unit_range(), normalize_thermal() (+1 more)

### Community 61 - "Resume Tests"
Cohesion: 0.29
Nodes (9): UB-06: --resume must reuse a previous run's directories and checkpoints.  Before, --resume with a nonexistent run id → non-zero exit + actionable error., Names of real run directories under outputs/ (symlinks excluded)., Epoch 1 run + --resume with NUM_EPOCHS=2 → history length 2 (T1.7 AC)., _run_dirs(), _run_id_from(), _set_fast_env(), test_resume_continues_metric_history() (+1 more)

### Community 62 - "Mask Defect Census Script"
Cohesion: 0.33
Nodes (9): build_bucket_index(), census_masks(), census_polygons(), main(), DataFrame, A11 — Censo completo do defeito de derivação de máscaras (UB-27).  A análise que, sample_id → bucket (43/73), deduplicado, para as imagens processadas.      Os CS, (a) pontos por região no JSON de polígonos, por grupo. (+1 more)

### Community 63 - "Thermal Preprocessing Standard"
Cohesion: 0.25
Nodes (9): commit 4a0d65c, commit 5f8e15b, CLAUDE.md §8 ML & Benchmark Methodology Standards, M7 — Thermal-domain preprocessing, T3.4 — thermal preprocessing and augmentation, UB-19 — thermal preprocessing destroys temperature, commit df517ee, test_augmentation.py (+1 more)

### Community 64 - "Load-Time Normalization"
Cohesion: 0.22
Nodes (9): ThermalFaceDataset, apply_normalization, preprocessing.fixed_range_celsius, preprocessing.normalization, config preprocessing section, codes/utils.py, fixed_range normalization, --force-preprocess (+1 more)

### Community 65 - "Pytest Fixtures"
Cohesion: 0.28
Nodes (8): _build_subject(), Shared pytest fixtures for UBench test suite.  Key fixture: ``synthetic_dataset`, Session-scoped synthetic dataset root (see CLAUDE.md §7.2).      Layout::, Return 4-corner polygon [[x,y], ...] for a rectangle., Create all synthetic files for one subject S{subject_n}., _rect_polygon(), synthetic_dataset(), TempPathFactory

### Community 66 - "Distance Stratification Script"
Cohesion: 0.33
Nodes (8): main(), part_a_distance_stratification(), part_b_pixels_per_class(), part_c_spearman_bonus(), ndarray, A5 — Estratificação por distância + pixels por classe (+ bônus Spearman).  Não h, load_raw_subject_csv(), Read one subject's raw CSV (landmarks + Distance/env-temp/...).      Adds `sampl

### Community 67 - "U-Net Architecture"
Cohesion: 0.29
Nodes (4): DoubleConv, Double convolution block for U-Net, U-Net architecture for semantic segmentation, UNet

### Community 68 - "Shell Entry Point"
Cohesion: 0.39
Nodes (7): main(), NO_ALBUMENTATIONS_UPDATE, run_extract(), run_pipeline(), run_setup(), run.sh script, UBENCH_RUN_ID

### Community 69 - "Pretraining Policy Defects"
Cohesion: 0.43
Nodes (7): M5 — Pretraining policy, T3.2 — pretrained encoders, UB-16 — ViT-B trained from scratch, UB-17 — defective Swin attention, codes/transunet.py, commit faa9a2d, test_models_forward.py

### Community 70 - "End-to-End Smoke Gate"
Cohesion: 0.33
Nodes (6): DataFrame, E2E smoke test — the merge gate (CLAUDE.md §7.3).  This test is the single defin, Full end-to-end pipeline smoke (UB-01/02/03/05/07 guard).      Smoke artifacts p, Return the most recently modified CSV matching *glob_pattern*.      Searched rel, read_latest(), test_full_pipeline_smoke()

### Community 71 - "Pareto Frontier Script"
Cohesion: 0.43
Nodes (6): load_final_miou(), load_latency(), main(), probe_vram(), A7 — Fronteira de compromisso (velocidade x mIoU x VRAM).  Não havia rascunho pa, Peak VRAM per architecture at the fixed probe batch (UB-10, M3).      Uses fresh

### Community 72 - "Warmup-Cosine Split Tests"
Cohesion: 0.33
Nodes (5): test_warmup_cosine_split(), test_warmup_cosine_split_needs_two_steps(), Split total optimizer steps into ``(warmup_steps, cosine_steps)`` (M4).      Gua, Construct the LR scheduler from a resolved recipe.          Returns ``(scheduler, warmup_cosine_split()

### Community 73 - "Output Directory Validation"
Cohesion: 0.33
Nodes (3): Initialize and validate paths                  Args:             output_dir: Ove, Validate required data paths exist, Create the run's output/log roots only.          The ``models``/``plots``/``pred

### Community 74 - "Landmark Side Evidence Script"
Cohesion: 0.40
Nodes (5): _carregar(), main(), DataFrame, A13 — Evidência reproduzível da regra de lado visível do esquema de 43 marcos (U, Retorna (frontais, perfis) com centróides x já normalizados pela face.

## Knowledge Gaps
- **125 isolated node(s):** `ubench`, `NO_ALBUMENTATIONS_UPDATE`, `UBENCH_RUN_ID`, `data/scripts/README.md`, `CLAUDE.md §3 Commands & Environment` (+120 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `timm` connect `Model Registry & Pretrained Stems` to `Reported Run Results`?**
  _High betweenness centrality (0.331) - this node is a cross-community bridge._
- **Why does `swin_pretrained` connect `Reported Run Results` to `Model Registry & Pretrained Stems`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `codes/hardware_detector.py` connect `Engineering Rules & Batch Keys` to `Filename Contract Defect`, `Reported Run Results`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Config` (e.g. with `ModelBenchmark` and `Pipeline`) actually correct?**
  _`Config` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `UnifiedTrainer` (e.g. with `Pipeline` and `_SpyScheduler`) actually correct?**
  _`UnifiedTrainer` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `OptimizerConfig` (e.g. with `_NoisyDataset` and `_SpyScheduler`) actually correct?**
  _`OptimizerConfig` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SchedulerConfig` (e.g. with `_NoisyDataset` and `_SpyScheduler`) actually correct?**
  _`SchedulerConfig` has 5 INFERRED edges - model-reasoned connections that need verification._
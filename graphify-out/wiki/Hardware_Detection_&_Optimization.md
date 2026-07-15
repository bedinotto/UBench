# Hardware Detection & Optimization

> 31 nodes · cohesion 0.09

## Key Concepts

- **HardwareProfile** (12 connections) — `codes/hardware_detector.py`
- **HardwareDetector** (9 connections) — `codes/hardware_detector.py`
- **detect_and_optimize()** (8 connections) — `codes/hardware_detector.py`
- **hardware_detector.py** (6 connections) — `codes/hardware_detector.py`
- **.detect()** (6 connections) — `codes/hardware_detector.py`
- **_smoke_runner.py** (6 connections) — `codes/tests/_smoke_runner.py`
- **.save_profile()** (4 connections) — `codes/hardware_detector.py`
- **._detect_gpu()** (3 connections) — `codes/hardware_detector.py`
- **.get_cuda_env_vars()** (3 connections) — `codes/hardware_detector.py`
- **._validate_gpu()** (3 connections) — `codes/hardware_detector.py`
- **._calculate_batch_sizes()** (3 connections) — `codes/hardware_detector.py`
- **._calculate_workers()** (3 connections) — `codes/hardware_detector.py`
- **.__init__()** (3 connections) — `codes/hardware_detector.py`
- **.to_dict()** (3 connections) — `codes/hardware_detector.py`
- **_cpu_profile()** (3 connections) — `codes/tests/_smoke_runner.py`
- **.__init__()** (1 connections) — `codes/hardware_detector.py`
- **.__str__()** (1 connections) — `codes/hardware_detector.py`
- **Hardware Detection and Optimization Module =====================================** (1 connections) — `codes/hardware_detector.py`
- **Convert to dictionary** (1 connections) — `codes/hardware_detector.py`
- **Detect and validate hardware capabilities** (1 connections) — `codes/hardware_detector.py`
- **Detect hardware and create profile** (1 connections) — `codes/hardware_detector.py`
- **Detect GPU name and memory** (1 connections) — `codes/hardware_detector.py`
- **Validate GPU meets minimum requirements** (1 connections) — `codes/hardware_detector.py`
- **Save hardware profile to JSON** (1 connections) — `codes/hardware_detector.py`
- **Hardware profile with optimization parameters** (1 connections) — `codes/hardware_detector.py`
- *... and 6 more nodes in this community*

## Relationships

- [Inference & Model Registry](Inference_%26_Model_Registry.md) (4 shared connections)
- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (2 shared connections)
- [Reproducibility Seeding](Reproducibility_Seeding.md) (1 shared connections)

## Source Files

- `codes/hardware_detector.py`
- `codes/tests/_smoke_runner.py`

## Audit Trail

- EXTRACTED: 90 (99%)
- INFERRED: 1 (1%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*
# Logging Infrastructure (TeeLogger)

> 24 nodes · cohesion 0.13

## Key Concepts

- **TeeLogger** (11 connections) — `codes/logger.py`
- **_TeeStream** (9 connections) — `codes/logger.py`
- **logger.py** (7 connections) — `codes/logger.py`
- **start_from_env()** (7 connections) — `codes/logger.py`
- **.start()** (6 connections) — `codes/logger.py`
- **setup.py** (6 connections) — `codes/setup.py`
- **.stop()** (4 connections) — `codes/logger.py`
- **.write()** (4 connections) — `codes/logger.py`
- **main()** (4 connections) — `codes/setup.py`
- **.flush()** (3 connections) — `codes/logger.py`
- **._write_timestamped()** (3 connections) — `codes/logger.py`
- **.__enter__()** (2 connections) — `codes/logger.py`
- **.__exit__()** (2 connections) — `codes/logger.py`
- **.__init__()** (2 connections) — `codes/logger.py`
- **Console Logger (Tee) ==================== Redirects sys.stdout and sys.stderr so** (1 connections) — `codes/logger.py`
- **Intercepts sys.stdout and sys.stderr and mirrors them to a log file.      Parame** (1 connections) — `codes/logger.py`
- **Start a TeeLogger by reading the log directory from the     ``UBENCH_LOG_DIR`` e** (1 connections) — `codes/logger.py`
- **A file-like object that writes to two streams simultaneously.     One stream is** (1 connections) — `codes/logger.py`
- **Prefix every complete line with a timestamp.** (1 connections) — `codes/logger.py`
- **.__init__()** (1 connections) — `codes/logger.py`
- **.fileno()** (1 connections) — `codes/logger.py`
- **.__getattr__()** (1 connections) — `codes/logger.py`
- **Cross-Platform Setup Script =========================== Automatically installs d** (1 connections) — `codes/setup.py`
- **TextIOWrapper** (1 connections)

## Relationships

- [Data Extraction & LFS Management](Data_Extraction_%26_LFS_Management.md) (3 shared connections)
- [Environment Setup & CUDA Management](Environment_Setup_%26_CUDA_Management.md) (3 shared connections)
- [Inference & Model Registry](Inference_%26_Model_Registry.md) (2 shared connections)
- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (1 shared connections)
- [Reproducibility Seeding](Reproducibility_Seeding.md) (1 shared connections)

## Source Files

- `codes/logger.py`
- `codes/setup.py`

## Audit Trail

- EXTRACTED: 79 (99%)
- INFERRED: 1 (1%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*
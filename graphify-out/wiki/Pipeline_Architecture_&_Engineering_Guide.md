# Pipeline Architecture & Engineering Guide

> 57 nodes · cohesion 0.05

## Key Concepts

- **Known-Defect Ledger (UB-01 to UB-22)** (9 connections) — `CLAUDE.md`
- **UBench Pipeline** (8 connections) — `CLAUDE.md`
- **Phase 1: Make run.sh Work** (7 connections) — `CLAUDE.md`
- **E2E Smoke Test (§7.3)** (7 connections) — `CLAUDE.md`
- **UBench Overview (README)** (7 connections) — `README.md`
- **UBench codes/config.yaml (active config)** (6 connections) — `codes/config.yaml`
- **Root config.yaml (dead - loaded by nothing)** (6 connections) — `config.yaml`
- **Phase 3: Credible Science** (5 connections) — `CLAUDE.md`
- **Phased Task Plan (§9)** (5 connections) — `CLAUDE.md`
- **UB-01: Preprocess Not Invoked** (5 connections) — `CLAUDE.md`
- **UB-05: Batch-Size Key Mismatch (swin)** (5 connections) — `CLAUDE.md`
- **UB-12: Config Drift (dead root config.yaml)** (5 connections) — `CLAUDE.md`
- **GitHub Actions CI Workflow** (4 connections) — `.github/workflows/ci.yml`
- **Phase 0: Safety Net** (4 connections) — `CLAUDE.md`
- **Synthetic Dataset Fixture (§7.2)** (4 connections) — `CLAUDE.md`
- **Testing Doctrine (§7)** (4 connections) — `CLAUDE.md`
- **UB-03: GroupKFold ValueError (<K datasets)** (4 connections) — `CLAUDE.md`
- **Phase 0 Kickoff Prompt (Session 0)** (4 connections) — `prompts/phase0_kickoff.md`
- **ML Benchmark Methodology Standards (§8)** (3 connections) — `CLAUDE.md`
- **Phase 2: Trustworthy Numbers** (3 connections) — `CLAUDE.md`
- **UB-02: Filename Contract Mismatch** (3 connections) — `CLAUDE.md`
- **Region Names (Portuguese, 10 classes)** (3 connections) — `codes/config.yaml`
- **T0.1 Synthetic Fixture + Smoke Test Task** (3 connections) — `prompts/phase0_kickoff.md`
- **CombinedLoss (CE + Dice)** (2 connections) — `CLAUDE.md`
- **Data-Flow Contract (extract→preprocess→train→benchmark)** (2 connections) — `CLAUDE.md`
- *... and 32 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `.github/workflows/ci.yml`
- `CLAUDE.md`
- `README.md`
- `codes/config.yaml`
- `config.yaml`
- `prompts/phase0_kickoff.md`

## Audit Trail

- EXTRACTED: 152 (93%)
- INFERRED: 12 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*
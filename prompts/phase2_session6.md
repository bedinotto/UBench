# Session 6 (final) — T2.1 (UB-09) + T2.2 (UB-11): honest timing & one metric authority — Phase-2 opener

This version supersedes the self-drafted `prompts/phase2_session6_prompt.md` after external review and remote tree verification at `ff59d05`. The draft was sound; this adds six upgrades (catch-up digest, tiny-loader warm-up guard, disk-retention chore + UB-25, suite-budget conditional, human real-data checklist, prompt-record commit). Read CLAUDE.md §5 (UB-09, UB-11), §8 M2/M3/M9, R4, R5, §9 T2.1/T2.2 before acting.

Branch: `phase-2/ub09-ub11-metrics` from `main` **after** main is fast-forwarded to the `phase-1/ub06-resume-runid` tip (`ff59d05`). This repo merges by FF push with linear history. **If main still lacks the Phase-1 closeout, halt after Step 0 and ask the human to FF first. Do not stack.**

Disk pre-flight: the box sits near capacity and full-suite runs churn ~10–15 GB in pytest temp (full-state TransUNet checkpoints). Clear old `pytest-*` dirs before long runs — ENOSPC reds are spurious reds.

## Step 0 — SESSION ENTRY PROTOCOL (§10) + one-time phase-boundary digest

Verify Session 5's end-state empirically (the draft's five checks stand):
1. `test -f codes/tests/test_resume.py && test -f codes/tests/test_run_identity.py`.
2. `pytest codes/tests/test_resume.py codes/tests/test_run_identity.py -q` → green (7 tests).
3. Ledger rows: UB-06 `FIXED@01244fa`, UB-13/UB-14 `FIXED@add5229`.
4. `--resume` present in `main_pipeline.py`; `UBENCH_RUN_ID` exported in `run.sh`.
5. Full suite green (41 tests) — paste count and wall time.

**If any check fails:** stop, report, and if Session 5 never ran, execute `prompts/phase1_session5_prompt.md` first.

**0.6 — Catch-up digest (one-time, ≤40 lines in the report):** Sessions 3–5 were never externally reviewed. From `git log` + ledger + tests, produce: every ledger flip with SHA (externally pre-verified: UB-03/04@73967c8, UB-05@cd3afa4, UB-08@cc890de, UB-07@74e1b43, UB-06@01244fa, UB-13/14@add5229); every NEW row added (UB-24 confirmed — list any others verbatim); deviations from §9 as planned; outstanding `UNVERIFIED` flags; test-inventory growth 3→41 by session. This closes the review gap; future sessions return to normal reporting.

## Step 1 — Commit 1 (chore): test-suite disk retention + UB-25

- `pyproject.toml`: set pytest `tmp_path_retention_policy = "failed"` (retain temp dirs only for failed runs; pytest ≥7.3 — satisfied).
- Ledger: **add UB-25** — Sev: Hygiene — `codes/tests/*` subprocess runs — "Full-suite runs persist ~10–15 GB of full-state checkpoints across retained pytest temp dirs (3-run default retention); near-full disks turn this into spurious ENOSPC reds. Mitigated by failed-only retention; root-cause slimming (e.g., skip per-epoch resume checkpoints under smoke configs) decided in T2.6." Status: OPEN (mitigated).
- AC: after a full suite run, `du -sh` of the pytest basetemp shows only failed-run retention. Paste it.

Commit: `chore(test): retain pytest tmp only on failure; ledger UB-25 disk footprint`

## Step 2 — Commit 2: T2.1 / UB-09 — honest timing

- Delete per-epoch inference timing from `UnifiedTrainer.validate()` — the value is unsynced kernel-launch time including loss. Sweep every consumer coherently: trainer history, metrics-JSON key, history-plot panel, checkpoint dict, and `test_resume.py` assertions. **Resume compat:** checkpoints written before this commit contain `inference_times`; `load_checkpoint` must tolerate both (verify the `.get(...)` path empirically, don't assume).
- Benchmark: extract a **`timed_inference` helper** (unit-testable) that (a) calls `torch.cuda.synchronize()` before/after iff device is cuda, (b) discards warm-up batches with the tiny-loader guard `warmup = min(5, max(0, n_batches - 1))` — the smoke's val loaders have <6 batches; timing zero batches is a bug, not an excuse — and (c) returns per-image mean±std plus `n_warmup`/`n_measured`. Report line states warm-up count, batch size, dtype, device (M9).
- Tests (`codes/tests/test_timing.py`, red first where feasible): warm-up arithmetic at n_batches ∈ {1, 3, 20}; sync callable invoked exactly twice when a monkeypatched cuda device is presented, zero times on cpu; `validate()` no longer emits a timing value.
- Verify: full suite + ruff green; smoke green.

Commit: `fix(timing): drop unsynced per-epoch timing; warm-up-discarded benchmark timing (UB-09, T2.1)`

## Step 3 — Commit 3: T2.2 / UB-11 — one metric authority

- New `codes/metrics.py`: `torchmetrics` on **argmax** — `JaccardIndex(average='none')` + hard Dice; macro excludes classes absent from the target; background reported separately, never folded into the macro (M2). Check the **pinned** torchmetrics API for the right Dice class (or F1 equivalence) and document the choice in the module docstring — the hand-computed numeric check is the real guarantee either way.
- Both `unified_training.py` and `benchmark_models.py` consume it (R5). Delete the soft `calculate_dice_score` (this item's mechanism); leave `calculate_iou` to T2.6 unless it falls out naturally. Benchmark "loss" becomes the training `CombinedLoss` via one shared construction path; relabel outputs: "Dice (hard, macro, excl. absent)", "Loss (CE+Dice)".
- Tests (`codes/tests/test_metrics.py`, red first): crafted logits/target with hand-computed Dice/IoU (exact match); absent-class exclusion; background-separation shape; trainer and benchmark paths produce identical values on one fixture tensor pair.
- **Numbers will move** (soft→hard). That is the point — say so in the commit body. Smoke asserts structure, not values; it stays green.

Commit: `fix(metrics): single hard-metric authority shared by trainer and benchmark; benchmark loss = CombinedLoss (UB-11, T2.2)`

## Step 4 — Deliverable for the human: `docs/phase1_realdata_checklist.md`

§2 says it plainly: the real-data run is **unverified** — the outstanding half of Phase-1's DoD, and only the human's GPU box can close it. Write the checklist: fresh clone; `git lfs pull`; venv per §3.1 GPU path (note UB-20b caveats until T3.5); dry run `python codes/main_pipeline.py --models unet --epochs 2`; then full `./run.sh`; mid-run Ctrl-C + `--resume <run_id>` drill; expected artifacts (single `logs/<ts>` dir, `outputs/latest` symlink, 3-row `benchmark_comparison.csv`); capture protocol — any deviation becomes a new UB row with logs attached, timing/VRAM numbers are provisional until UB-10/T2.3 lands. Reference it from §2.

Commit: `docs(checklist): Phase-1 real-data validation protocol for the GPU box`

## Step 5 — Docs & prompt record

Ledger flips (UB-09, UB-11 → FIXED@sha; UB-25 added); §2 frontier advanced (remaining Phase-2: UB-10, T2.4, T2.5, T2.6+UB-24/25; real-data validation pending human run); save THIS file over `prompts/phase2_session6_prompt.md` with a `docs(prompts):` commit noting the deltas from the self-draft.

## Step 6 — Suite budget + push & CI

Measure post-session full-suite wall time. If it exceeds the §7.1 five-minute cap: split CI into `fast` (unit) and `smoke` jobs in a `chore(ci):` commit and amend §6.4/§7.1 in the same commit (R9). Measure, don't guess. Push; paste Actions status/URL for both jobs or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails

Excluded temptations: UB-10 (T2.3 — you will be inside `benchmark_models.py`; resist), UB-12 (T3.1), UB-15 beyond what metric unification directly supersedes (T2.6), UB-24 (parked for T2.6/T3.1), best-checkpoint-by-mIoU (T3.3), per-family recipes (T3.3). Tool-generated edits (graphify etc.) → separate `chore(tooling):` or discard. R1/R3/R4/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:

7. **Frontier:** Phase-2 remainder (UB-10, T2.4, T2.5, T2.6+UB-24/25) and real-data validation status.
8. Ledger diff hunks (UB-09/UB-11 flips, UB-25 addition, §2 update) + the Step 0.6 catch-up digest.
9. **Prompt for Session 7** covering **T2.3 (UB-10)** — memory probe at one fixed batch size for all models, report labeled "VRAM @ batch=N (fixed)", CPU-safe no-op path — and **T2.4 (M1)** — config-driven `test_subjects` excluded from all CV with CV and TEST report sections. **Warning to encode:** the synthetic fixture has subjects S1–S5 only; `test_subjects: [S9, S10]` from the plan must be config/fixture-aware (e.g., default from config, fixture sets its own), or the smoke test breaks on nonexistent subjects. Two items, two commits, Session Entry Protocol first.

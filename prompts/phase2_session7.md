# Session 7 — T2.3 (UB-10) + T2.4 (M1): fixed-batch VRAM probe & held-out test subjects

Two ledger/plan items, two atomic commits, plus docs. Continues **Phase 2
(trustworthy numbers)**. Read CLAUDE.md §5 (UB-10), §8 M1/M3/M9, R4, R5, and
§9 T2.3/T2.4 before acting.

Branch: `phase-2/ub10-testsubjects` from `main` **after** main is fast-forwarded
to the `phase-2/ub09-ub11-metrics` tip (`ce7922f`). This repo merges by FF push,
linear history. **If main still lacks the Session-6 commits, halt after Step 0
and ask the human to FF first. Do not stack.**

Disk pre-flight: full-suite runs churn ~16 GB of pytest temp; Session 6 set
`tmp_path_retention_policy = "failed"` so green runs self-clean, but clear
`/tmp/pytest-of-doga` before long runs if the box is near capacity (UB-25).

## Step 0 — SESSION ENTRY PROTOCOL (§10)

Verify Session 6's end-state empirically before any work:
1. `test -f codes/metrics.py && test -f codes/tests/test_timing.py && test -f codes/tests/test_metrics.py`.
2. `pytest codes/tests/test_timing.py codes/tests/test_metrics.py -q` → green (11 tests).
3. Ledger: UB-09 `FIXED@bc01d07`, UB-11 `FIXED@3260638`, UB-25 present (OPEN, mitigated).
4. `grep -n "timed_inference" codes/benchmark_models.py` present; `grep -n "SegmentationMetrics" codes/unified_training.py` present.
5. Full suite `pytest codes/tests -q` → green (52 tests); paste count + wall time.

**If any check fails:** stop, report; if Session 6 was never executed, run
`prompts/phase2_session6_prompt.md` first. Do not build on an unverified base.

## Step 1 — Commit 1: T2.3 / UB-10 — fixed-batch-size VRAM probe

- **Defect (ledger):** `benchmark_models.py` compares peak VRAM across models at
  *different* batch sizes (each model's hardware-selected batch), so the numbers
  are not comparable (M3: "never compare peak VRAM across models at different
  batch sizes").
- **Fix shape (§9 T2.3, M3):** measure peak allocated memory in a **separate
  probe at one fixed batch size** shared by all models (e.g. a config/const
  `MEMORY_PROBE_BATCH_SIZE = 4`): `torch.cuda.reset_peak_memory_stats()` →
  one forward on a fixed-size batch → `torch.cuda.max_memory_allocated()`. Make
  it a small unit-testable helper alongside `timed_inference`. **CPU-safe no-op
  path:** on CPU (no CUDA) return `None`/0 and label it "n/a (CPU)" — do not
  fabricate a number. Keep the existing per-pass `peak_memory_mb` out of the
  headline or relabel it; the comparable figure is the fixed-batch probe.
- **Report/CSV:** label the column/line **"VRAM @ batch=4 (fixed)"** (M9). State
  the probe batch size wherever peak memory is reported.
- **Tests (`codes/tests/test_memory.py`, red-first where feasible):** the probe
  uses the fixed batch size regardless of the model's training batch; CPU path
  returns the no-op sentinel and is labeled; a monkeypatched CUDA path calls
  `reset_peak_memory_stats`/`max_memory_allocated` once each.
- Verify: full suite + ruff green; smoke green.

Commit: `fix(benchmark): fixed-batch-size VRAM probe; CPU no-op labeled (UB-10, T2.3)`

## Step 2 — Commit 2: T2.4 / M1 — held-out test subjects, CV vs TEST report

- **Goal (§9 T2.4, M1):** reserve **held-out test subjects** excluded from all
  CV folds; fold metrics *select*, test metrics *headline*. Never tune on test
  subjects.
- **Fix shape:** a config key `test_subjects` (list of subject/dataset ids). In
  `unified_data.py`, drop those subjects from the CV pool **before**
  `GroupKFold` (so no fold ever sees them), and expose a loader for the held-out
  set. The benchmark evaluates the test set through the same
  `SegmentationMetrics` authority and the report grows a **CV section** (per-fold
  mean ± std) and a **TEST section** (held-out subjects). Document the
  weight/ensemble choice for producing test predictions (e.g. best-fold model or
  fold ensemble) — state it, don't hide it.
- **⚠ Fixture warning (critical):** the synthetic fixture has subjects **S1–S5
  only**; the plan's `test_subjects: [S9, S10]` do not exist there. Make
  `test_subjects` **config/fixture-aware**: default from config, but the smoke
  fixture must set its own (e.g. reserve S5) or default to empty so the smoke
  does not break on nonexistent subjects. Prove the smoke still produces the
  3-row CSV.
- **Tests (`codes/tests/test_test_subjects.py`, red-first):** a configured test
  subject appears in **no** fold's train or val split (exclusivity); with test
  subjects set, the benchmark emits both CV and TEST numbers; with none set,
  behavior is unchanged (backward-compatible default).
- Verify: full suite + ruff green; smoke green.

Commit: `feat(splits): held-out test subjects excluded from CV; CV+TEST report sections (M1, T2.4)`

## Step 3 — Docs
Ledger: UB-10 → `FIXED@<sha>`; §2 frontier advanced (remaining Phase-2: T2.5
seeding, T2.6 dead-code/UB-24/UB-25); §4.5 splits note the held-out test set
(the CLAUDE.md §4.5 line "A held-out test-subject set arrives in T2.4" becomes
"landed"); §8 M1 reconciled. Save this prompt file. Update README split protocol
if T2.4 changes user-facing behavior (R9).

## Step 4 — Suite budget + push & CI
Measure full-suite wall time (was 4:25 at Session 6 with 52 tests). If it now
exceeds the §7.1 five-minute cap, split CI into `fast`/`smoke` jobs in a
`chore(ci):` commit and amend §6.4/§7.1 (R9) — measure, don't guess. Push; paste
Actions status/URL or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails
Excluded temptations: UB-12 config consolidation (T3.1 — but T2.4 *reads* a
config key; do not rework the loader), best-checkpoint-by-mIoU (T3.3), per-family
recipes (T3.3), UB-15 dead code (T2.6), UB-24 top-level outputs clutter (T2.6).
Tool-generated edits (graphify) → separate `chore(tooling):`. R1/R3/R4/R9/R10 in
force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Frontier:** Phase-2 remainder (T2.5 seeding/determinism, T2.6 dead-code +
   UB-24/UB-25) and real-data validation status.
8. Ledger diff hunks (UB-10 flip, §2 update).
9. **Prompt for Session 8** covering **T2.5 (UB-20a/M6)** — seeded DataLoader
   `generator` + `worker_init_fn`, single `cudnn.benchmark` owner, per-run
   metadata JSON (git SHA, lockfile hash, torch/CUDA, GPU, effective config);
   AC = two identical seeded 1-epoch CPU runs produce identical loss curves —
   and **T2.6 (UB-15/22 + UB-24/UB-25)** — delete dead code & scratch scripts
   (port `verify_regex` into a test), wire-or-delete `inference_comparison.py`,
   fix the 2×6 grid, `normalize_thermal` min==max → zeros, fold in the top-level
   `outputs/` clutter (UB-24) and checkpoint slimming (UB-25). Two items, two
   commits, Session Entry Protocol first.

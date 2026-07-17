# Session 7 — T2.3 (UB-10) + T2.4 (M1): fixed-batch VRAM probe & held-out test subjects

*Supersedes the self-drafted `prompts/phase2_session7.md` — same scope, with the FF target corrected and seven methodology/ordering fixes folded in (marked ▲).* Two ledger/plan items, two atomic commits, plus docs. Continues **Phase 2 (trustworthy numbers)**. Read CLAUDE.md §5 (UB-10), §8 M1/M3/M9, R4, R5, and §9 T2.3/T2.4 before acting.

Branch: `phase-2/ub10-testsubjects` from `main` **after** main is fast-forwarded to the Session-6 tip **`7ad9434`** ▲ (the current tip — *not* `ce7922f`; the FF target moved when the graphify and Session-7-prompt commits landed). Linear FF history. **If main lacks the Session-6 commits, halt after Step 0 and ask the human to FF. Do not stack.**

Disk pre-flight: green suite runs self-clean since Session 6 set `tmp_path_retention_policy="failed"`; still clear `/tmp/pytest-of-*` before long runs if the box is near capacity (UB-25).

## Step 0 — SESSION ENTRY PROTOCOL (§10)

Verify Session 6's end-state empirically before any work:
1. `test -f codes/metrics.py && test -f codes/tests/test_timing.py && test -f codes/tests/test_metrics.py`.
2. `pytest codes/tests/test_timing.py codes/tests/test_metrics.py -q` → green (11 tests).
3. Ledger: UB-09 `FIXED@bc01d07`, UB-11 `FIXED@3260638`, UB-25 present (OPEN, mitigated).
4. `grep -n "timed_inference" codes/benchmark_models.py` and `grep -n "SegmentationMetrics" codes/unified_training.py` both present.
5. Full suite `pytest codes/tests -q` → green (52 tests); paste count + wall time.

**If any check fails:** stop, report; if Session 6 was never executed, run `prompts/phase2_session6_prompt.md` first. Do not build on an unverified base.

## Step 1 — Commit 1: T2.3 / UB-10 — fixed-batch-size VRAM probe

**Defect (ledger):** `benchmark_models.py` compares peak VRAM across models at *different* (hardware-selected) batch sizes → not comparable (M3). Note Session 6 already reads a per-pass `peak_memory_mb` before the timing pass; that is not the comparable figure.

**Probe design — be precise (M3):**
- One shared constant `MEMORY_PROBE_BATCH_SIZE = 4` (config or module const), used for all models.
- ▲ **Synthetic input, not a data batch:** build a tensor of shape `(MEMORY_PROBE_BATCH_SIZE, in_ch, H, W)` (1-channel thermal, config image size) directly on the device. No dependence on the loader or batch contents → the number reflects the *model*, deterministically, and the CPU path and test become trivial.
- ▲ **Inference-mode, isolated:** `model.eval()` + `torch.no_grad()` (same mode as `timed_inference`, so the two stay comparable). `torch.cuda.reset_peak_memory_stats()` **immediately before** the single probe forward, then read `torch.cuda.max_memory_allocated()`. This is **one forward on one fixed batch** — not a loader pass — run as its own isolated step so it neither perturbs nor is perturbed by the timing/metrics passes. Label it inference-mode; training-mode memory, if ever wanted, is a separate future line, not this one.
- **CPU no-op:** no CUDA → return `None`, render **"n/a (CPU)"**; never fabricate a number (R10).

**Report/CSV:** one comparable figure labeled **"VRAM @ batch=4 (fixed, inference)"** (M9); state the probe batch size + mode wherever peak memory appears. Relabel or drop the old per-pass `peak_memory_mb` so there is exactly one memory column.

**Tests (`codes/tests/test_memory.py`, red-first, unit-level — no subprocess):** probe uses the fixed batch regardless of the model's training batch; synthetic-input shape correct; CPU path returns the sentinel and is labeled; a monkeypatched CUDA path calls `reset_peak_memory_stats`/`max_memory_allocated` once each.

Verify: full suite + ruff green; smoke green (its VRAM shows "n/a (CPU)").

Commit: `fix(benchmark): fixed-batch inference VRAM probe; CPU no-op labeled (UB-10, T2.3)`

## Step 2 — Commit 2: T2.4 / M1 — held-out test subjects, CV vs TEST report

**Goal (M1):** reserve held-out test subjects excluded from **all** CV folds; folds *select*, test *headlines*; never tune on test.

**Split mechanics — mind the ordering (real bug risk):**
- Config key `test_subjects: []` — a list, **default empty** in shipped `codes/config.yaml` (opt-in; no silent behavior change on real data).
- ▲ In `unified_data.py`, remove `test_subjects` from the subject pool **before** `resolve_fold_count` **and** `GroupKFold`, so (a) no fold can see them and (b) the T1.4 fold-count guard now operates on the **CV (training) pool**. A `test_subjects` list that leaves **<2 CV subjects** must raise the same actionable "≥2 subjects" error — verify this interaction explicitly, it is easy to get wrong.
- ▲ **Validate config (R4):** a `test_subject` id absent from the discovered data raises (or at minimum loudly warns) — silently holding out *nothing* is the exact failure M1 exists to prevent.
- Preprocessing is unchanged: `metadata.csv` still covers all subjects (test subjects are preprocessed like any other); they are merely filtered out of the CV pool and routed to a held-out loader containing **exactly** the test subjects.

**Producing test metrics — recommended strategy (state it, M9):** ▲ evaluate **each of the K fold-models on the held-out test set** and report **mean ± std**, parallel to the CV section. No retraining, and no "best" model is picked (there is nothing to select on — test is held out), so it is the most honest default. If you instead ensemble folds or retrain-on-all-CV, document why. Both CV (per-fold mean ± std) and TEST (held-out) sections go through the one `SegmentationMetrics` authority; the report grows a **CV section** and a **TEST section**.

**⚠ Fixture (critical) — the synthetic fixture has S1–S5 only; the plan's `[S9,S10]` don't exist there.** ▲ Cover both paths:
- The **smoke fixture reserves one real fixture subject** (e.g. `test_subjects: ["S5"]`) so the CV+TEST path is exercised end-to-end: assert the CV section covers the S1–S4 folds, the TEST section covers S5, S5 appears in no fold, and the 3-row CSV survives.
- A **DataFrame-level unit test** covers the empty default (backward-compatible: no TEST section, unchanged behavior).

**Tests (`codes/tests/test_test_subjects.py`, red-first; DataFrame/unit-level — ▲ reuse the existing smoke for the integration assertion, do NOT add a second full-pipeline subprocess run; protects the 5-min budget and UB-25 disk):** two-directional exclusivity — configured test subjects appear in **no** fold's train or val, AND the held-out loader is **exactly** the test subjects; the <2-CV-subject config raises; an absent test_subject id raises; the empty default is unchanged.

Verify: full suite + ruff green; smoke green.

Commit: `feat(splits): held-out test subjects excluded from CV; CV+TEST report sections (M1, T2.4)`

## Step 3 — Docs

- Ledger: UB-10 → `FIXED@<sha>`.
- §2 frontier advanced (remaining Phase-2: T2.5 seeding, T2.6 dead-code + UB-24/UB-25).
- §4.5: the line "A held-out test-subject set arrives in T2.4" → landed (opt-in via `test_subjects`, default empty; per-fold-on-test reporting).
- §8 M1 reconciled to the shipped mechanism and the recommended strategy.
- ▲ `docs/phase1_realdata_checklist.md`: add that a real/citable run **must** set `test_subjects` (M1) — 1–2 subjects held out from CV — and that with the default empty there is no TEST section.
- Save this prompt file (superseding the self-draft). Update README split protocol if user-facing behavior changed (R9).

## Step 4 — Suite budget + push & CI

Measure full-suite wall time (52 tests → 4:25 at Session 6). If it now exceeds the §7.1 five-minute cap, split CI into `fast`/`smoke` jobs in a `chore(ci):` commit and amend §6.4/§7.1 (R9) — measure, don't guess. Push; paste Actions status/URL or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails

Excluded temptations: UB-12 config consolidation (T3.1 — T2.4 only *reads* a config key; do **not** rework the loader, and the memory probe must not import training-only state), best-checkpoint-by-mIoU (T3.3), per-family recipes (T3.3), UB-15 dead code (T2.6), UB-24 top-level outputs clutter (T2.6). Tool-generated edits (graphify) → separate `chore(tooling):`. R1/R3/R4/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:

7. **Frontier:** Phase-2 remainder (T2.5 seeding/determinism, T2.6 dead-code + UB-24/UB-25) and real-data validation status.
8. Ledger diff hunks (UB-10 flip; §2/§4.5/§8 updates).
9. **Prompt for Session 8** covering **T2.5 (UB-20a/M6)** — seeded DataLoader `generator` + `worker_init_fn`, a single `cudnn.benchmark` owner, per-run metadata JSON (git SHA, lockfile hash, torch/CUDA, GPU, effective config); AC = two identical seeded 1-epoch CPU runs produce identical loss curves — and **T2.6 (UB-15/22 + UB-24/UB-25)** — delete dead code & scratch scripts (port `verify_regex` into a test), wire-or-delete `inference_comparison.py`, fix the 2×6 grid, `normalize_thermal` min==max → zeros, fold in the top-level `outputs/` clutter (UB-24) and checkpoint slimming (UB-25). Two items, two commits, Session Entry Protocol first.

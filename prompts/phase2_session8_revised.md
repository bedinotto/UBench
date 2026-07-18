# Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep

*Supersedes the self-drafted `prompts/phase2_session8.md` — same scope, with the FF target corrected and seven fixes folded in (marked ▲). The most important, ▲A, fixes a determinism test that would ship a broken `worker_init_fn` green.* Closes out **Phase 2 (trustworthy numbers)**. Read CLAUDE.md §5 (UB-15, UB-20, UB-22, UB-24, UB-25), §8 M6, R3, R4, R6, R8, and §9 T2.5/T2.6 before acting.

Branch: `phase-2/determinism-cleanup` from `main` **after** main is fast-forwarded to the Session-7 tip **`55f3456`** ▲ (the current tip — *not* `8156bc8`; the FF target moved when the graphify and Session-8-prompt commits landed). Linear FF history. **If main lacks the Session-7 commits, halt after Step 0 and ask the human to FF. Do not stack.**

Disk pre-flight: green suites self-clean (`tmp_path_retention_policy=failed`); clear `/tmp/pytest-of-*` before long runs if near capacity (UB-25).

## Step 0 — SESSION ENTRY PROTOCOL (§10)

1. `test -f codes/tests/test_memory.py && test -f codes/tests/test_test_subjects.py`.
2. `pytest codes/tests/test_memory.py codes/tests/test_test_subjects.py -q` → green (~10 tests).
3. Ledger: UB-10 `FIXED@3bbe8df`; UB-09 `FIXED@bc01d07`; UB-11 `FIXED@3260638`.
4. `grep -n "test_subjects\|VRAM @ batch" codes/*.py` present (probe + held-out split landed).
5. Full suite `pytest codes/tests -q` → green (62 tests); paste count + wall time.

**If any check fails:** stop, report; if Session 7 was never executed, run `prompts/phase2_session7.md` first. Do not build on an unverified base.

## Step 1 — Commit 1: T2.5 / UB-20a + M6 — seeded determinism + per-run metadata

**Defect (ledger UB-20):** DataLoaders have no `generator`/`worker_init_fn`; a `cudnn.benchmark` contradiction with no single owner. Runs are not reproducible.

**Fix (M6):**
- Every DataLoader (`create_single_fold_loader`, `create_kfold_data_loaders`, `create_test_loader`) gets `generator=torch.Generator().manual_seed(config.RANDOM_SEED)` and a module-level `worker_init_fn(worker_id)` that seeds numpy **and** python `random` per worker (derive from base seed + worker_id). Do **not** touch the augmentation transform definitions (T3.4) — seeding the global numpy/random RNGs is exactly what makes the existing albumentations pipeline reproducible, and that is M6, not augmentation redesign.
- ▲ **Resolve the `cudnn.benchmark` contradiction with a value, not just an owner.** Add a config flag `deterministic: true` (default true — reproducibility is the whole point of this task). When true, one owner (`seed_everything`) sets `cudnn.deterministic = True` and `cudnn.benchmark = False`; when false, the hardware profile may enable `benchmark` for speed. Either way a single code path decides it from the one flag, so the old contradiction is gone. Do **not** call `torch.use_deterministic_algorithms(True)` — it raises on ops lacking deterministic kernels and would destabilize the pipeline. Note in the commit body that this branch is GPU-only and thus validated by review + the flag's existence, not by the CPU tests.
- **Per-run metadata JSON** in each run dir: git SHA + dirty flag, torch/CUDA versions, GPU name, effective-config dump, ▲ and the `RANDOM_SEED` + `deterministic` flag actually used (reproducibility provenance). Lockfile hash is a `TODO` until T3.5 — note it, don't fabricate. Write it in `Pipeline.__init__`/`run()` beside existing per-run artifacts.

**Tests — ▲ this is the upgrade that matters:**
- ▲ **`worker_init_fn` is never called at `NUM_WORKERS=0`**, so a 0-worker determinism test does not exercise it and would pass even if the seeding were broken. Two tests instead:
  1. **Direct unit test** of `worker_init_fn(worker_id)`: after calling it, assert numpy's and `random`'s next draws are the deterministic expected values for that `worker_id`, and differ across worker ids. Fast, reliable, no data.
  2. **`codes/tests/test_determinism.py`** loss-curve test run at **`NUM_WORKERS=2`** with augmentation on — this is red-first and meaningful: before the fix, two seeded 1-epoch runs diverge (worker RNGs unseeded); after, per-epoch train/val losses match exactly (or within a tight tolerance). If a 2-worker run proves flaky on the CI runner, fall back to the direct unit test as the guarantee plus a 0-worker loss-curve test, and say so. Do **not** add a second full-pipeline subprocess run (budget/UB-25).
- Verify: full suite + ruff green; smoke green.

Commit: `fix(repro): seeded DataLoader generator + per-worker init; single cudnn owner via deterministic flag; per-run metadata (UB-20a, M6, T2.5)`

## Step 2 — T2.6 / UB-15, UB-22, UB-24, UB-25 — dead-code & scratch sweep

▲ **One ledger item per commit (R6) — not one bundled "cleanup" commit.** These four are independent; keep the ledger flip atomic with each change so history stays bisectable. Surgical deletions only (R3); prove no references remain (ruff + grep) before each commit.

- **Commit 2a — UB-15 (dead code / correctness bugs):** remove `train_iou_metric` (never updated), `calculate_iou` (unused after T2.2), the no-op `load_shared_data`; fix `normalize_thermal` so `min == max` returns **zeros** (not the un-normalized image); fix the "11-panels-in-10-axes"/2×6 grid. AC: unit test pins `normalize_thermal(min==max) → zeros`; grep proves the removed symbols have no references; suite green.
  `fix(cleanup): remove dead code; normalize_thermal min==max→zeros; grid panel count (UB-15, T2.6)`
- **Commit 2b — UB-22 (scratch/orphan removal):** wire-or-**delete** `inference_comparison.py` (orphaned — deletion is fine, it was never in the pipeline); delete the gitignored scratch debug scripts and drop their `collect_ignore_glob`; port `verify_regex`'s intent into a real test only if signal remains, else delete. AC: grep/ruff show no dangling references or ignore globs; suite green.
  `chore(cleanup): remove orphaned inference_comparison + scratch scripts (UB-22, T2.6)`
- **Commit 2c — UB-24 (stray top-level `outputs/` dirs):** ▲ first `grep -rn "outputs/models\|outputs/plots\|outputs/predictions"` to find everything that *reads* those paths, so removal doesn't strand a consumer (don't delete-and-hope — Config is constructed in many places). Then stop `Config.__init__` from mkdir-ing top-level `outputs/{models,plots,predictions}`; create those only under `outputs/<run_id>/` when a run starts. AC: a fresh-clone-style run (and the smoke) leave **no** stray top-level `outputs/` subdirs; nothing that read them breaks.
  `fix(io): create output subdirs only under the run dir, not top-level (UB-24, T2.6)`
- **Commit 2d — UB-25 (checkpoint disk footprint):** ▲ recommended root-cause lever is **retain only the latest resume checkpoint** (delete the prior epoch's when writing the next — resume needs only the most recent, so this is a legitimate production improvement, not smoke-only). **But re-run `test_resume.py` and confirm UB-06 resume semantics still hold** — if rolling retention risks that behavior, keep it smoke-config-scoped or leave mitigated and document why. AC: `test_resume` green; basetemp footprint after a full suite run materially reduced (paste `du -sh`).
  `perf(checkpoint): retain only latest resume checkpoint; UB-06 resume re-verified (UB-25, T2.6)`

## Step 3 — Docs & Phase-2 close-out

- Ledger: UB-15, UB-22 → `FIXED@<sha>`; UB-24, UB-25 → `FIXED@<sha>` (or note residual for UB-25 if left mitigated); **UB-20 → partial** — the M6/loader/cudnn part (a) is done; the deps/lockfile/CUDA-index part (b) remains for T3.5 — annotate, do **not** close.
- ▲ **Phase-2 exit criteria — confirm the phase goal, not just flip rows:** state explicitly that "trustworthy numbers" now holds — synchronized warm-up timing (UB-09), one hard-metric authority (UB-11), fixed-batch inference VRAM (UB-10), held-out test subjects (T2.4), seeded reproducibility (T2.5) — and that the one outstanding cross-phase item is the **real-data validation on the human GPU box** (`docs/phase1_realdata_checklist.md`), which a citable run must pair with `test_subjects`.
- §2 frontier → **Phase 2 complete**; frontier becomes Phase 3 (T3.1 config consolidation first). §4.8 reconciled once scratch scripts are gone. Save this prompt file. README/quickstart updated if behavior changed (R9).

## Step 4 — Suite budget + push & CI

Measure full-suite wall time (62 tests → 4:18 at Session 7; T2.5 adds a determinism test, T2.6 removes collection overhead). If it exceeds the §7.1 five-minute cap, split CI into `fast`/`smoke` jobs (`chore(ci):`) and amend §6.4/§7.1 (R9) — measure, don't guess. Push; paste Actions status/URL or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails

Excluded temptations: UB-12 config consolidation (T3.1), pretrained encoders (T3.2), per-family recipes (T3.3), thermal-domain preprocessing / augmentation redesign (T3.4), lockfiles + CUDA index (T3.5 — UB-20b), import restructure (T3.6). Tool-generated edits (graphify) → separate `chore(tooling):`. R1/R3/R4/R6/R8/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:

7. **Frontier:** Phase 2 done → Phase 3 opener (T3.1 UB-12 config consolidation), plus real-data validation status (still pending the human GPU box). Include the Phase-2 exit-criteria confirmation.
8. Ledger diff hunks (UB-15/22/24/25 flips, UB-20 partial-annotation; §2 update).
9. **Prompt for Session 9** covering **T3.1 (UB-12)** — delete root `config.yaml`; a pydantic(-settings) schema over `codes/config.yaml` that fails on unknown keys; wire the scheduler/optimizer/augmentation/`class_weights` keys (class weights from train-fold frequencies) or delete the dead ones. AC: a bogus key raises; every remaining key demonstrably alters behavior; the smoke still runs. Note the Phase-3 shift — this is the first "make the *science* credible" session — and that T3.1 must not silently change current numeric behavior (a key that was hardcoded must keep its current effective value unless deliberately changed). One item; Session Entry Protocol first.

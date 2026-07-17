# Session 8 — T2.5 (UB-20a/M6) + T2.6 (UB-15/22, UB-24/UB-25): determinism & dead-code sweep

Two items, two atomic commits, plus docs. Closes out **Phase 2 (trustworthy
numbers)**. Read CLAUDE.md §5 (UB-15, UB-20, UB-22, UB-24, UB-25), §8 M6, R3,
R4, R8, and §9 T2.5/T2.6 before acting.

Branch: `phase-2/determinism-cleanup` from `main` **after** main is
fast-forwarded to the Session-7 tip (`git log -1 phase-2/ub10-testsubjects` —
currently `8156bc8`). Linear FF history. **If main lacks the Session-7 commits,
halt after Step 0 and ask the human to FF. Do not stack.**

Disk pre-flight: green suites self-clean (`tmp_path_retention_policy=failed`);
clear `/tmp/pytest-of-*` before long runs if near capacity (UB-25).

## Step 0 — SESSION ENTRY PROTOCOL (§10)

1. `test -f codes/tests/test_memory.py && test -f codes/tests/test_test_subjects.py`.
2. `pytest codes/tests/test_memory.py codes/tests/test_test_subjects.py -q` → green (10 tests).
3. Ledger: UB-10 `FIXED@3bbe8df`; UB-09 `FIXED@bc01d07`; UB-11 `FIXED@3260638`.
4. `grep -n "probe_peak_memory\|test_subjects\|create_test_loader" codes/*.py` present.
5. Full suite `pytest codes/tests -q` → green (62 tests); paste count + wall time.

**If any check fails:** stop, report; if Session 7 was never executed, run
`prompts/phase2_session7.md` first. Do not build on an unverified base.

## Step 1 — Commit 1: T2.5 / UB-20a + M6 — seeded determinism + per-run metadata

- **Defect (ledger UB-20):** DataLoaders have no `generator` / `worker_init_fn`;
  there is a `cudnn.benchmark` contradiction with no single owner. Results are
  not bit-reproducible run to run.
- **Fix (M6):** give every DataLoader (`create_single_fold_loader`,
  `create_kfold_data_loaders`, `create_test_loader`) a seeded
  `generator=torch.Generator().manual_seed(config.RANDOM_SEED)` plus a
  `worker_init_fn` that seeds numpy/random per worker. One owner for
  `cudnn.benchmark` (the hardware profile), documented determinism trade-off.
  Confirm a single `seed_everything` owner (already imported in
  `main_pipeline`). **Do not** widen scope into the augmentation RNG (T3.4).
- **Per-run metadata JSON:** each run dir logs git SHA + dirty flag, torch/CUDA
  versions, GPU name, and an effective-config dump (lockfile hash is optional
  until T3.5 lands lockfiles — note it as TODO, don't fabricate). Write it in
  `Pipeline.__init__` or `run()` alongside the existing per-run artifacts.
- **Test (`codes/tests/test_determinism.py`):** AC = two seeded 1-epoch CPU runs
  produce **identical loss curves** — drive `train_epoch`/`validate` twice with
  `NUM_WORKERS=0`, fixed seed, and assert the per-epoch train/val losses match
  exactly (or within a tight float tolerance). Keep it unit/integration-level;
  do not add a second full-pipeline subprocess run (budget/UB-25).
- Verify: full suite + ruff green; smoke green.

Commit: `fix(repro): seeded DataLoader generator + worker_init_fn; cudnn owner; per-run metadata (UB-20a, M6, T2.5)`

## Step 2 — Commit 2: T2.6 / UB-15 + UB-22 (+ UB-24, UB-25) — dead-code & scratch sweep

Surgical deletions only (R3); prove no references remain (ruff + grep).
- **UB-15 dead code:** remove `train_iou_metric` (never updated), `calculate_iou`
  (unused after T2.2), the no-op `load_shared_data`, and fix `normalize_thermal`
  so `min == max` returns **zeros** (not the un-normalized image). Fix the
  "11-panels-in-10-axes"/2×6 grid bug.
- **UB-22 hygiene:** wire-or-**delete** `inference_comparison.py` (orphaned);
  delete the gitignored scratch debug scripts (`codes/tests/test_unet_nan*.py`,
  `test_suite.py`) and drop their `collect_ignore_glob`; port `verify_regex`'s
  intent into a real test if any signal remains, else delete.
- **UB-24:** stop `preprocess_all_data()`'s default `Config()` from mkdir-ing
  top-level `outputs/{models,plots,predictions}` — construct with the run dirs,
  or make `_create_output_dirs` not create the stray top-level dirs.
- **UB-25:** if cheap, slim per-epoch resume checkpoints under smoke configs
  (root cause of the disk churn); otherwise leave mitigated and note why.
- **Test/AC:** ruff clean; `grep` proves the removed symbols have no references;
  a small unit test pins `normalize_thermal(min==max) -> zeros`; full suite green;
  a fresh-clone-style run leaves no stray top-level `outputs/` subdirs.

Commit: `refactor(cleanup): remove dead code & scratch scripts; normalize_thermal zeros; no stray outputs dirs (UB-15/22/24/25, T2.6)`

## Step 3 — Docs
Ledger: UB-15, UB-22 → `FIXED@<sha>`; UB-20 → partial (a-part done, b-part/T3.5
remains — annotate, don't close); UB-24, UB-25 → `FIXED@<sha>` (or note residual).
§2 frontier → **Phase 2 complete**; frontier becomes Phase 3 (T3.1 config
consolidation first). §4.8 (dev/debug scripts) reconciled once the scratch
scripts are gone. Save this prompt file. README/quickstart updated if behavior
changed (R9).

## Step 4 — Suite budget + push & CI
Measure full-suite wall time (was 4:18 at Session 7 with 62 tests; deleting the
gitignored scratch scripts does not change the collected count, but T2.6 may
remove some collection overhead). If it exceeds the §7.1 five-minute cap, split
CI into `fast`/`smoke` jobs (`chore(ci):`) and amend §6.4/§7.1 (R9). Push; paste
Actions status/URL or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails
Excluded temptations: UB-12 config consolidation (T3.1), pretrained encoders
(T3.2), per-family recipes (T3.3), thermal-domain preprocessing / augmentation
RNG (T3.4), lockfiles (T3.5), import restructure (T3.6). Tool-generated edits
(graphify) → separate `chore(tooling):`. R1/R3/R4/R8/R9/R10 in force. Blocked or
contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Frontier:** Phase 2 done → Phase 3 opener (T3.1 UB-12 config consolidation),
   and real-data validation status (still pending the human GPU box).
8. Ledger diff hunks (UB-15/22/24/25 flips, UB-20 partial; §2 update).
9. **Prompt for Session 9** covering **T3.1 (UB-12)** — delete root `config.yaml`;
   pydantic schema over `codes/config.yaml` failing on unknown keys; wire
   scheduler/optimizer/augmentation/`class_weights` (from train-fold frequencies)
   or delete the dead keys. AC: a bogus key raises; every remaining key
   demonstrably alters behavior. One item; Session Entry Protocol first.

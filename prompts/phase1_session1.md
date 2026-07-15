# Session 1 — Ledger reconciliation + CPU/dependency enablers + T1.1 (UB-01)

Phase 0 landed: the smoke test XFAILs on UB-01 via strict xfail, CI is green. This session starts Phase 1. Read `CLAUDE.md` in full again before acting — then note that some of what the Phase 0 report *described* was never written into the file; Step 1 fixes that first.

Branch: `phase-1/ub01-and-enablers`, based on `main` if the Phase 0 PR is merged, otherwise on `phase-0/safety-net`. One commit per ledger item, in the order below — every commit must leave `ruff check codes/` clean and `pytest codes/tests -q` green (smoke stays XFAIL throughout; its *reason* moves at the end).

**Session outcome (read carefully):** the smoke test will NOT go green this session, by design. Fixing UB-01 lets training run, which advances the failure to UB-02 (benchmark finds only TransUNet's weights). Your deliverable is moving the frontier exactly one defect forward, with proof. UB-02 is next session — do not touch it.

## Step 0 — Environment & branch
Recreate/activate the venv per CLAUDE.md §3.1, paste `python -V` and the torch import check, create the branch, confirm `git status` is clean.

## Step 1 — Commit 1 (docs only): make CLAUDE.md true again
The Phase 0 session put required doc changes in prose and commit messages instead of the file (violates R9's letter). Edit `CLAUDE.md` itself:
- §9 T0.2 AC → the strict-xfail formulation actually adopted ("suite green; smoke test xfail(strict=True) enforcing the current frontier defect").
- §5 ledger: **add row UB-23** — Sev: Blocker — `codes/hardware_detector.py` — "`detect()` calls `sys.exit(1)` when CUDA is unavailable; contradicts §3.1 CPU-first doctrine; blocks CI E2E; Phase 0 worked around it with a test-only runner that patches detection (to be removed by the UB-23 fix)." Status: OPEN.
- §5 UB-20: append note — "albumentations 2.0.8 rejects `ShiftScaleRotate(value=…)` (Phase 0/D2); pinned `<2.0` in Session 1; API migration lands with T3.4."
- §3.1: amend the Python note — "3.13 verified for CPU-only dev work (Phase 0/D1); GPU training boxes remain on 3.10/3.11 until T3.5 verifies CUDA wheels."
- §7.1: update the test-layout listing to match reality (root `conftest.py` with `collect_ignore_glob`; `_smoke_runner.py` will be deleted this session — reflect the end-state, i.e. don't list it).

Commit: `docs(claude): reconcile ledger and ACs with Phase 0 discoveries (UB-23, UB-20 note, D1)`. Paste the diff hunks of the ledger edits in your report — the ledger must change **in the file**, not in prose.

## Step 2 — Commit 2: UB-23 — real CPU mode; delete the test-only bypass
- In `hardware_detector.py`: when CUDA is unavailable **and** env `UBENCH_ALLOW_CPU=1` is set, return a complete CPU profile instead of exiting — `device='cpu'`, AMP off, `pin_memory=False`, `num_workers=0`, small batch sizes, plus a loud warning banner that CPU mode is for testing/CI, not benchmarking. Without the env var, current behavior stands. **Use the existing batch-size key scheme `{unet, transunet, swin}` with small values (e.g. 4/2/2) — do NOT rename keys; that is UB-05/T1.3's job.**
- Populate every field the code reads: grep all attribute accesses on the hardware profile across `main_pipeline.py`, `unified_data.py`, `unified_training.py`, `benchmark_models.py` and satisfy them.
- Same grep pass for unguarded `torch.cuda.*` calls reachable on the train/benchmark path (`synchronize`, `reset_peak_memory_stats`, `memory_allocated`, `.cuda()`, `empty_cache`…). Guard each with a device check. These guards are in UB-23's scope — list every one in the commit body. Anything CUDA-related you find but do **not** need to touch this session gets a ledger note instead.
- **Delete `codes/tests/_smoke_runner.py`.** The smoke test now invokes `codes/main_pipeline.py` directly as the subprocess, with `UBENCH_ALLOW_CPU=1` in its env. The gate must exercise the real entry point.
- Verify: full suite green; smoke still XFAIL and the captured subprocess output still shows the UB-01 `FileNotFoundError` — now reached through real hardware detection. Paste it.

Commit: `fix(hardware): CPU execution profile behind UBENCH_ALLOW_CPU; drop test-only runner (UB-23)`

## Step 3 — Commit 3: UB-20 (partial) — pin albumentations
`requirements/requirements.txt`: pin `albumentations>=1.4,<2.0`. Reinstall into the venv; paste the resolved version. Rationale in commit body: full lockfile is T3.5, API migration to `A.Affine` is T3.4; this pin merely keeps 1.x behavior until then.

Commit: `fix(deps): pin albumentations<2.0 pending T3.4 migration (UB-20)`

## Step 4 — Commit 4: T1.1 / UB-01 — wire preprocessing into the pipeline
- In `main_pipeline.py`: as an explicit pipeline step before data loading, if `data/processed/metadata.csv` is absent, invoke `preprocess_all_data()` (import from `codes.preprocess_data`); add `--force-preprocess` to rebuild unconditionally. Log it as a named stage. README quickstart gains the step (state that it is automatic, and how to force it).
- Expect fixture-vs-preprocess friction (synthetic 64×64 frames, bbox padding, resize to 256, `.npy`/`.png` outputs, `image_path` resolution from CWD). If the fixture proves structurally unfaithful, fix the **fixture** — change nothing in pipeline code beyond the wiring itself (R3). The border-bbox sample's mask **will be silently mis-shifted — that is UB-08, expected, owned by T1.5. Do not touch it.**
- Run the smoke test with the marker bypassed (`pytest --runxfail` or temporarily strip it) to observe the new frontier. Required evidence, pasted:
  (a) preprocessing stage log lines;
  (b) each of the three models completes epoch 1 on both folds (per-model loss lines);
  (c) `ls outputs/<run>/models/` showing `best_unet_fold_*`, `best_transunet_fold_*`, `best_swin_unet_plus_plus_fold_*` — registry names, pre-verifying T1.2's premise;
  (d) the benchmark stage warning that U-Net and Swin-UNet++ weights were not found, and the 3-row assertion failing with only TransUNet present — i.e. the frontier is exactly UB-02.
- Update the xfail marker: `reason="UB-02: checkpoint filename contract mismatch (frontier after T1.1)"`, `strict=True` stays. Re-run the suite normally: green, smoke XFAIL under the new reason.
- Mark UB-01 `FIXED@<sha>` and UB-23 `FIXED@<sha>` in the ledger (in this commit or a trailing docs commit).

Commit: `fix(pipeline): run preprocessing automatically when metadata absent (UB-01, T1.1)`

## Hard scope exclusions this session
Do **not** fix UB-02 (even though it is now staring at you from the logs — next session), UB-05 (no batch-size key renames), UB-13, UB-08 (border mask shift stays), UB-09/UB-11 (you will see the bogus timing and soft-Dice numbers scroll by — leave them). No reformatting, no other dependency changes, no README rewrites beyond the preprocessing step.

## End-of-session report — same format as Phase 0, plus:
7. **Frontier statement:** the exact defect now blocking the smoke test, with the log lines proving it (evidence d above).
8. **Ledger-in-file confirmation:** paste the CLAUDE.md diff hunks for UB-23 (added), UB-20 (note), UB-01/UB-23 (status flips). Prose claims about the ledger without a diff are not accepted.
9. The exact prompt for Session 2 (T1.2: `codes/naming.py`, `test_filenames.py`, xfail marker removal — the smoke-goes-green session).

Rules in force: R1 (no claim without executed output), R3 (surgical diffs), R4 (no silencing — the CPU profile is an explicit opt-in, not a hidden fallback), R9 (docs change in the file, same PR), R10 (all CPU-run numbers are smoke artifacts, never results). If anything blocks you or contradicts CLAUDE.md, stop, report, propose options, and wait.

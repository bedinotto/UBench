# Session 4 — T1.5 (UB-08) + T1.6 (UB-07): mask-offset clamp & failure honesty

Two ledger items, two atomic commits, plus docs. Read CLAUDE.md §5 (UB-08, UB-07), R4, R1, and §9 T1.5/T1.6 before acting.

Branch: `phase-1/ub08-ub07-honesty` — from `main` if the pending PR stack is merged (it was 4 deep at Session 3's end: `phase-0/safety-net` → `phase-1/ub01-and-enablers` → `phase-1/ub02-naming` → `phase-1/ub05-ub03-splits`; the human was asked to merge before this session); otherwise stacked on `phase-1/ub05-ub03-splits`. If still stacking, halt after Step 0 and ask the human to merge first — 5 deep is past the limit.

## Step 0 — SESSION ENTRY PROTOCOL (mandatory, per CLAUDE.md §10)

Verify Session 3's claimed end-state empirically before any work:

1. `test -f codes/tests/test_batch_size_keys.py && test -f codes/tests/test_splits.py` → both exist.
2. `pytest codes/tests/test_batch_size_keys.py codes/tests/test_splits.py -q` → green (includes the 3-subject partial-corpus pipeline run).
3. `grep -n "UB-03\|UB-04\|UB-05" CLAUDE.md | head -5` → ledger rows show `FIXED@73967c8` / `FIXED@73967c8` / `FIXED@cd3afa4`.
4. `grep -rn "\.get(model_name" codes/main_pipeline.py` → no hit (hard lookup in place); `grep -n "Session Entry Protocol" CLAUDE.md` → present in §10.
5. Full suite: `pytest codes/tests -q` → green (~2 min, includes smoke + partial-corpus run).

**If ANY check fails:** stop, report which and how. If Session 3 was never executed, run `prompts/phase1_session3_prompt.md` to completion first. Do not build on an unverified base; do not "quickly fix" gaps in passing.

## Step 1 — Commit 1: T1.5 / UB-08 — clamped crop origin, aligned mask offsets

- **Defect (verified in ledger):** `preprocess_data.py:66-67` computes the polygon offset as `bbox.min − 10` **unclamped**, while the crop origin is `max(0, bbox.min − 10)`. For bboxes closer than 10 px to the border (the fixture's `min_x=3` sample), masks shift by up to 10 px — silent label corruption.
- **Fix shape (per §9 T1.5):** compute the clamped origin **once**; `crop_to_bbox` returns `(img, origin)` — the origin it *actually used* — and the mask/polygon offset consumes that returned origin. One authority, no recomputation (R5). Check every caller of `crop_to_bbox` and every place the `−10` padding appears; grep for other unclamped `min_x`/`min_y` arithmetic in the file.
- **`codes/tests/test_preprocess_offsets.py`** (red first — the current code must fail it):
  - Drive the real preprocessing path on a synthetic sample whose bbox has `min_x = 3` (< 10 px padding; the fixture's UB-08 tripwire sample exists for exactly this). Construct a polygon at a known position; after preprocessing, assert the mask's foreground centroid is within 1 px of the expected position in the cropped/resized frame (T1.5's AC).
  - Include a control: a non-border bbox (`min_x = 10`) must produce the same alignment — proving the fix didn't shift the normal path.
  - Unit-level: `crop_to_bbox` on a border bbox returns origin `(0, y)` (clamped), not `(-7, y)`.
- Re-run preprocessing in the test via `--force-preprocess` or a fresh tmp dir — stale `data/processed/` from other tests must not mask the red.
- Verify: new test red on unfixed code (paste it), green after; full suite + ruff green. **Note:** the smoke fixture's masks change meaning after this fix — if the smoke or other tests assert on mask content, re-verify they still pass for the right reason, not by accident.

Commit: `fix(preprocess): clamp crop origin once and align polygon offsets to it (UB-08, T1.5)`

## Step 2 — Commit 2: T1.6 / UB-07 — failure registry, honest exit codes

- **Defect:** per-model/fold `try/except` in `train_all_models` prints one line and continues; a SUCCESS banner + exit 0 is possible with zero trained models; `main()`'s error-log writer is unreachable because `Pipeline()` is constructed outside the `try`.
- **Fix (per §9 T1.6):**
  - Failure registry on the `Pipeline` instance: every caught per-model/fold exception is recorded (model, fold, exception repr, traceback), not just printed.
  - End-of-run summary lists failures explicitly; the SUCCESS banner prints **only** when zero failures were recorded; otherwise a FAILURE summary + **non-zero exit** from `main()`.
  - `--fail-fast` CLI flag: first failure aborts the run immediately (non-zero exit).
  - Move `Pipeline()` construction inside `main()`'s `try` so its error-log writer (`error_log_*.txt`) is reachable; a constructor crash must produce the error log, not a bare traceback.
  - R4 alert: do **not** widen any `except`; the registry records what is already caught. The per-model `except Exception` stays narrow in scope but becomes honest.
- **Tests** (red first where feasible):
  - Injected failure: monkeypatch/register a model whose `forward` (or trainer entry) raises → pipeline subprocess exits **1**, output contains the failure summary naming model+fold, `error_log_*.txt` exists (T1.6's AC). An env-var hook (e.g. `UBENCH_INJECT_FAIL=<model_key>`) honored only in the trainer's test path is acceptable if a monkeypatch can't cross the subprocess boundary — keep it inert without the var.
  - `--fail-fast`: same injection → run aborts after first failure (no later folds in output), exit non-zero.
  - Smoke test tightened: assert rc==0 **and** no failure summary in output (rc==0 now actually means "all models trained" — close the loophole noted in Session 3's test_splits docstring and the smoke scope note).
- Update the smoke-test scope note and §2 (UB-07 no longer masks failures) in the same commit or the docs commit.

Commit: `fix(pipeline): failure registry with honest exit codes and --fail-fast (UB-07, T1.6)`

## Step 3 — Docs commit: ledger flips (UB-08, UB-07 → FIXED@<sha>); §2 current-state rewrite (remaining Phase-1: UB-06, UB-13/14); §3.2 run.sh warning note removed; commit this session's prompt file.

## Step 4 — Push & remote CI
Push; paste the GitHub Actions status/URL for this branch. `UNVERIFIED: remote CI` + instruction to the human if unreachable.

## Standing guardrails
- Temptations, all excluded: UB-09/10/11 metric noise, UB-06 resume (next session), UB-13/14 flags, UB-19's `np.vectorize` sitting right next to the UB-08 code you'll touch, dead `calculate_iou`/`train_iou_metric` (T2.6). Preprocessing performance is NOT in scope — only correctness.
- Tool-generated edits (graphify etc.) → separate `chore(tooling):` commit or discard.
- R1/R3/R4/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format:
1–6 as before (gate results, per-commit summary with pasted verification runs, suite status, CI status, stack state, memory update).
7. **Frontier statement:** expected remaining Phase-1 items: UB-06 (T1.7), UB-13/14 (T1.8); any new discoveries ledgered in-file.
8. Ledger diff hunks (UB-08, UB-07 flips; §2 rewrite).
9. **Prompt for Session 5** covering **T1.7 (UB-06)** — `--resume <run_id>` reusing dirs/checkpoints, `outputs/latest` symlink, kill-after-epoch-1-resume-completes-epoch-2 test with metric history length 2 — **and T1.8 (UB-13/14)** — `--epochs default=None` honored at exactly 100, `run.sh` exports `UBENCH_RUN_ID` and Python reuses it, single `logs/<ts>` per run. Phase-1 exit criteria: all §9 Phase-1 boxes checked, smoke green, `./run.sh` honest end-to-end.

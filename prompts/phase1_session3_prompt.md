# Session 3 — T1.3 (UB-05) + T1.4 (UB-03/04): batch-size contract & fold-count guard

Two ledger items, two atomic commits, plus docs. Pace increases; one item per commit stays. Read CLAUDE.md §5 (UB-05, UB-03, UB-04), R4, R5, and §9 T1.3/T1.4 before acting.

Branch: `phase-1/ub05-ub03-splits` — from `main` if the pending PR stack (phase-0, session 1, session 2) is merged; otherwise stacked on the Session 2 branch. If still stacking, list the full stack in the report: it is now deep enough that the human must merge before Session 4.

## Step 0 — PREDECESSOR VERIFICATION GATE (mandatory, before any work)

Session reports are claims; the tree is the truth. Verify Session 2's end-state empirically:

1. `test -f codes/naming.py` and `pytest codes/tests/test_filenames.py -q` → passes.
2. `grep -rn "xfail" codes/tests/test_pipeline_smoke.py` → no marker remains.
3. `grep -n "UB-02" CLAUDE.md` → ledger row shows `FIXED@<sha>`; `git log --oneline -5` shows the naming commit.
4. Full suite: `pytest codes/tests -q` → green, smoke test **passes** (executes the real pipeline, ~70 s), zero xfail.

**If ANY check fails:** stop immediately. Report which check failed and in what way. If the cause is "Session 2 was never executed," run the Session 2 prompt (`phase1_session2_prompt.md`) to completion first, then re-enter this session from Step 0. Do not attempt T1.3/T1.4 on top of an unverified base, and do not "quickly fix" whatever gap you find — the gap itself is the finding.

**Institutionalize the gate:** in this session's docs commit, add to CLAUDE.md §10 (Verification Gates) a standing **Session Entry Protocol**: *every session begins by verifying the predecessor's claimed end-state from the working tree (tests, ledger, key files) before new work; a failed check halts the session.* This exists because trusting reports over trees is how drift starts.

## Step 1 — Commit 1: T1.3 / UB-05 — canonical batch-size keys, hard lookup

- **Rename** the batch-size dict keys in `hardware_detector.py` to registry names — `unet`, `transunet`, `swin_unet_plus_plus` — in **every** tier branch, including the CPU profile added by UB-23 (it deliberately used the old scheme; migrating it now is in scope). Update the detector's log labels to match.
- **Grep every consumer** of `batch_sizes` (training path in `main_pipeline._get_fold_loaders`, benchmark loader creation, anything else) and replace `.get(key, default)` with hard `[key]` lookup — unknown key must raise `KeyError` (R4). Delete any surviving display-name→short-key mapping remnants; after Session 2, the registry key is the only model identifier that touches dicts and I/O (R5).
- **`codes/tests/test_batch_size_keys.py`** (red first where feasible):
  - Simulate tiers by driving `_calculate_batch_sizes` (or the detector) with monkeypatched VRAM values — no GPU needed.
  - Assert the 6 GB tier yields `swin_unet_plus_plus == 6` and the <5.5 GB tier `== 3` (the ledger's reference values). **Derive expected numbers from the code's actual tier table; if the code disagrees with the ledger's 6/3, report the discrepancy — do not silently adjust either side.**
  - **Parity invariant:** for every tier (CPU included), `set(batch_sizes.keys()) == set(registered model keys)` from `model_registry`. This is the test that makes UB-05 structurally unrepeatable — when T3.2 registers new models, it will force their batch sizes to exist.
  - Unknown-key access raises `KeyError`.
- Update `test_hardware_cpu.py` assertions for the renamed keys (same ledger item, same commit).
- Verify: full suite green including the smoke (which now exercises the hard lookup end-to-end on the CPU profile). Paste the run.

Commit: `fix(hardware): registry-named batch-size keys with hard lookup and parity test (UB-05, T1.3)`

## Step 2 — Commit 2: T1.4 / UB-03+UB-04 — fold-count guard & split-semantics docs

- **One shared helper** in `unified_data.py` — e.g. `resolve_fold_count(requested_k: int, n_groups: int) -> int` — used by **both** GroupKFold call sites (no duplicated logic): returns `min(requested_k, n_groups)` with a loud warning naming both numbers when reduced; raises a clear, actionable error when `n_groups < 2` ("leave-subjects-out CV requires ≥2 subject datasets; found N: […]"). Error and warning text must use the *leave-subjects-out* vocabulary.
- **`codes/tests/test_splits.py`** — unit-test at the DataFrame level (fast, no files): build tiny metadata frames with variable subject counts and assert:
  - 3 subjects, K=5 → runs with `effective_k == 3`, warning captured (`caplog`/`pytest.warns`);
  - 2 subjects, K=5 → runs with `effective_k == 2`;
  - 1 subject → raises the clear error (not sklearn's cryptic `ValueError`);
  - **subject exclusivity:** for every produced fold, train-subjects ∩ val-subjects == ∅.
  Keep one cheap integration touch via the existing fixture only if it adds signal; the dataframe tests are the substance.
- **AC precision fix (R9):** CLAUDE.md §9 T1.4's AC says "runs with 1–3 subject dirs" — internally inconsistent with "error if <2". Rewrite the AC in the same commit: *2–3 subjects → runs with reduced K + warning; 1 subject → clear actionable error.*
- **README (scoped):** rewrite the split-semantics section to leave-subjects-out CV and delete the impossible per-fold example output. Nothing else in the README moves — full reconciliation is T3.7.
- Ledger: UB-03 → `FIXED@<sha>`; UB-04 → `FIXED@<sha>` (its fix *is* docs-aligned-with-code plus the guard).

Commit: `fix(data): fold-count guard with leave-subjects-out semantics; docs aligned (UB-03, UB-04, T1.4)`

## Step 3 — Docs commit (or fold into Step 2's): ledger flips, Session Entry Protocol added to §10, §2 "Current state" updated (partial-corpus crash resolved; remaining Phase-1 items: UB-08, UB-07, UB-06, UB-13/14).

## Step 4 — Push & remote CI
Push; paste the GitHub Actions status/URL for this branch. `UNVERIFIED: remote CI` + instruction to the human if unreachable.

## Standing guardrails
- Temptations in these files, all excluded: `create_kfold_data_loaders`' vestigial return (UB-15/T2.6), seeding/`worker_init_fn` (UB-20/T2.5), `np.vectorize` (UB-19/T3.4), UB-08's offset (you will pass the crop code — T1.5 next session), UB-07's swallowed exceptions, UB-09/10/11 metric noise in the smoke logs.
- Tool-generated edits (graphify etc.) never fold into fix commits — separate `chore(tooling):` or discard.
- R1/R3/R4/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Frontier statement:** what now stands between the current tree and Phase-1 completion (expected: UB-08, UB-07, UB-06, UB-13/14), with any new discoveries ledgered in-file.
8. Ledger diff hunks (UB-05, UB-03, UB-04 flips; §10 Session Entry Protocol addition; T1.4 AC rewrite).
9. **Prompt for Session 4** covering **T1.5 (UB-08)** — clamped crop origin, `crop_to_bbox` returning the origin it used, `test_preprocess_offsets.py` proving the border-bbox mask is pixel-aligned (the fixture's `min_x=3` sample finally earns its keep) — **and T1.6 (UB-07)** — failure registry, non-zero exit, `--fail-fast`, `Pipeline()` inside `main()`'s try, with an injected-failure test asserting exit 1 and the error log. Two items, two commits, Session Entry Protocol at the top.

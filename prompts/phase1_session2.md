# Session 2 — T1.2 (UB-02): unify the checkpoint filename contract — the smoke-goes-green session

Session 1 moved the frontier to UB-02 with full evidence: training writes registry-named checkpoints; the benchmark searches display-name derivations (`best_u_net_…`, `best_swin_unetplusplus_…`) and silently drops 2 of 3 models while exiting 0. This session fixes exactly that, removes the xfail marker, and produces the first fully green end-to-end smoke run. **One ledger item: UB-02.** Read CLAUDE.md §5 (UB-02), R4, R5, and §9 T1.2 before acting.

Branch: `phase-1/ub02-naming` — from `main` if the pending PRs (phase-0, phase-1 session 1) have been merged, otherwise stacked on `phase-1/ub01-and-enablers`. If still stacking, say so in the report: the stack is now three deep and the human should merge before Session 3.

## Step 0 — Environment & gates baseline
Venv active; paste `python -V`, torch version, `git status` clean, and a baseline `pytest codes/tests -q` (expect: green with 1 XFAIL under the UB-02 reason).

## Step 1 — RED: `codes/tests/test_filenames.py`
Write the contract test before the module exists. It must assert, as pure-function properties (no training, no I/O):
- For every registry key (`unet`, `transunet`, `swin_unet_plus_plus`) × fold (1, 2) × kind, `naming.checkpoint_path(...)` embeds the registry key **verbatim** in the filename.
- **Negative regression assertions:** no produced path ever contains `u_net` or `swin_unetplusplus` — the two historical bad derivations. This is UB-02's tombstone.
- The path for a given (key, fold, kind) is deterministic and unique across keys and folds.

Expected red state: collection/ImportError on `codes.naming` — that is the correct "module not built yet" red; do not gold-plate around it. Paste the red run.

## Step 2 — GREEN: `codes/naming.py` + wiring both sides
- Create `codes/naming.py`: `checkpoint_path(output_dir: Path, model_key: str, fold: int, kind: ...) -> Path`, pure, no globals. Grep every `.pth`-writing and `.pth`-searching call site (trainer best-model save, per-epoch resume checkpoints, benchmark search) and route **all** of them through this one authority; enumerate the call sites in the commit body. Optionally validate `model_key` against `model_registry` (cheap typo guard, R4-friendly) if it imports without a cycle.
- `unified_training.py` (save side) and `benchmark_models.py` (load side) both import it. Thread `model_key` through `run_benchmark` alongside the display name — display names remain for logs, plots, and the CSV's `Model` column (the smoke asserts display names in the CSV; do not change them — R5: display for humans, registry for I/O).
- **Delete the alternate/fallback filename search block** in the benchmark. A dead fallback is how contract drift hides; one canonical path, and the existing warn-and-skip on a genuinely missing file stays (hard-failure semantics are T1.6's job, not yours).
- Audit remaining `_safe_filename` call sites: any that produce *paths* must route through `naming.py`; pure label-sanitizing uses may stay. Report the audit result.
- Make `weights_only=True` explicit on any `torch.load` you touch (torch ≥2.6 already defaults to it; explicit is the R8 standard).
- Do **not** touch `hardware_profile.batch_sizes` lookups while you are in `main_pipeline.py` threading `model_key` — the `.get(key, 8)` you will be staring at is UB-05, next session.

## Step 3 — Marker removal is atomic with the fix
`strict=True` means the suite goes RED on a surprise pass — therefore the xfail marker removal and the naming fix must land **in the same commit**, or an intermediate tree violates every-commit-green. In that commit:
- Full suite run: paste the tail — expected `4 passed` (or current count), zero xfail.
- Paste the 3-row `benchmark_comparison.csv` content ({U-Net, TransUNet, Swin-UNet++}) and the smoke wall time (Session 1 baseline: ~67 s; it now executes as a full PASS on every CI run — if the total suite creeps past the 5-min budget, note it for a future slow-marker decision, do not act now).
- Confirm rc==0 and that all artifacts landed under the tmp CWD (hygiene line).

Commit: `fix(naming): single checkpoint-path authority; benchmark finds all models; smoke green (UB-02, T1.2)`

## Step 4 — Docs & ledger (same commit or trailing docs commit)
- Ledger: UB-02 → `FIXED@<sha>`.
- §2 "Current state" rewrite — precise, not triumphant: *smoke-level E2E is green on synthetic CPU data; a full real-data run (10 subjects, K=5) is plausible but **unverified**; partial-corpus runs still crash on UB-03; UB-07 still masks per-model failures.* Smoke-green ≠ real-data-green — say exactly that.
- §3.2 `run.sh` annotation updated to the same effect; §9 T1.2 checkbox ticked.

## Step 5 — Push & verify CI remotely (first time)
Every green so far has been local. Push the branch; confirm the GitHub Actions run for it is green (gh CLI or the web UI) and paste the run status/URL. If you cannot reach Actions from this environment, write `UNVERIFIED: remote CI` and instruct the human to check before merge.

## Standing guardrails (new + repeated)
- Tool-generated edits (graphify or any other local skill touching files) are never folded into fix/ledger commits — separate `chore(tooling):` commit or discard. Diffs must stay auditable (R3).
- Hard exclusions this session: UB-05, UB-03/04, UB-07 (warn-and-skip stays), UB-08, UB-09/10/11, UB-13. You will see all of them scroll by in a passing smoke run; leave them.
- R1/R3/R4/R9/R10 in force as always. If blocked or CLAUDE.md proves wrong: stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Milestone statement** replacing the frontier statement: smoke green, with the CSV and suite evidence; then the *next* frontier per plan — T1.3 (UB-05) — and the standing real-data risks (UB-03 partial-corpus, UB-07 masking) that smoke-green does not cover.
8. Ledger-in-file diff hunks (UB-02 flip, §2 rewrite excerpt).
9. **Prompt for Session 3**, drafted to cover **two ledger items as two separate commits/PR-able units in one session** — T1.3 (UB-05: canonical batch-size keys + hard `[k]` lookup + `test_batch_size_keys.py` with a simulated 6 GB profile asserting swin=6) and T1.4 (UB-03/04: `effective_k = min(K, n_groups)` guard + error below 2 + `test_splits.py` subject-exclusivity + README split-semantics rewrite deleting the impossible example). Discipline is proven; pace increases, one item per commit stays.

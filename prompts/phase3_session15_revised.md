# Session 15 — T3.7 (R9): README end-to-end reconciliation — closes Phase 3

*Supersedes the self-drafted Session-15 prompt from Session 14's report — same intent, but scoped to the **specific drift the tree actually shows** rather than a blanket rewrite. The config table, model-choices table, and preprocessing note were already reconciled in stride across T3.1–T3.4 and are CORRECT — do not churn them (R3). The stale surface is the **Prerequisites + Quickstart + Expected-Output** top half, which still predates T3.5's `uv` locks and the CV/recipes work. Six targeted fixes below (▲).* One ledger item, docs-only. Read CLAUDE.md R3/R9, §9 T3.7. **Note: the ledger currently has ZERO open UB rows** — this session is the last Phase-3 task; after it, only the human GPU-box validation and Phase 4 remain.

Branch: `phase-3/readme` from current `main` (**`5644728`**). Linear FF; no stacking. Push: gh-credentialed HTTPS.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. `ruff check codes/` clean; full suite green (~142, 2 net-skips) — paste count.
2. Ledger: UB-21 `FIXED@1800abf`; **grep confirms no `OPEN` rows remain**; §2 says next T3.7.
3. 5 models register; `gh run list --branch main` green for `5644728`.

## The verified drift (grep the whole README against the code — do not trust this list alone; Session 14's lesson) — known-stale sections:
- ▲ **§ Python Version (README:38-42):** claims "PyTorch CUDA wheels only for 3.8–3.12, 3.13 not supported." Reconcile to the real §3.1 state: CPU dev works on 3.13 (verified); the **CPU lock is pinned to the 3.10 floor** and CI runs 3.11; GPU/CUDA path stays 3.10/3.11. State the lock/`requires-python` reality, not the old blanket "3.13 unsupported."
- ▲ **§ Python Packages / auto-installed by setup (README:65-76):** says "PyTorch installed by `setup.py` via index URL; all other packages from `requirements.txt`." **This is now wrong** (T3.5): deps are single-sourced in `pyproject.toml [project]`; `setup.py` installs from the committed **`requirements/requirements.cpu.lock`** via `uv pip sync`, no `--force-reinstall --no-deps`, no plain-`requirements.txt` install. Rewrite to the lock-based install.
- ▲ **§ Quickstart Step 2 (README:110-131):** `./run.sh` + `--force-preprocess` are fine, but there is **no `uv` environment-setup step shown before it**. Add the actual reproducible-install quickstart from §3.1: `uv venv` → `uv pip sync requirements/requirements.cpu.lock --torch-backend=cpu` (CPU/dev/CI), and the CUDA-lock path pointer for the GPU box. This is the command the AC will run verbatim.
- ▲ **§ Expected Console Output (README:146-179):** predates the split/recipe/normalization work. Reconcile to current behavior: **leave-subjects-out GroupKFold** (not the old stratified description, if present), the **CV vs TEST** sections when `test_subjects` is set, **per-family recipes** (AdamW+warmup for transformers) in the log, `fixed_range` normalization, best-checkpoint-by-**val mIoU**. Keep it illustrative; do not fabricate specific metric numbers (R10) — show shape/labels, not invented values.
- ▲ **§ How the Pipeline Works / Execution Flow (README:277-330):** verify the diagram includes the **preprocessing stage** (UB-01) and reflects the current module set (no orphaned `inference_comparison.py` — deleted in T2.6; no `swin`-only choices — UB-26). Fix any step that names removed code.
- ▲ **Execution style everywhere:** any `python codes/foo.py` bare-script invocation → `python -m codes.foo` (T3.6); grep the whole README for `python codes/` and fix each.

## Already-correct — DO NOT rewrite (R3):
The `codes/config.yaml` config-key table (README:370-389), the 5-model choices table with the honest UB-16/17 baseline caveats (README:357-368), and the T3.4 preprocessing/manifest note (README:390) all match the code. Leave them. If a cross-reference to them needs a wording touch, minimal only.

## AC — the real proof (R9)
Every command in the reconciled README must run **verbatim in a fresh environment** during this session, and no claim may describe unimplemented behavior. Concretely, run and paste:
- `uv venv && uv pip sync requirements/requirements.cpu.lock --torch-backend=cpu` → success;
- the quickstart smoke path (fast config / `LIMIT_SAMPLES`+`NUM_EPOCHS=1`+`K_FOLDS=2`) → exit 0, 3-model comparison produced;
- `python -m codes.transunet` → forward shape;
- a grep proving no `python codes/` bare-script invocations and no references to deleted modules (`inference_comparison`, the old `swin`-only choice list) remain.
If any README command fails, the README is wrong — fix the doc, not the command (unless the command reveals a genuine code regression, in which case stop and report per the guardrails).

## Docs / ledger / close-out
Ledger: UB-04's documentation tail and any residual doc-drift rows → note reconciled; there is no new UB row unless the run surfaces a real code bug. **§2 "Current state": mark Phase 3 COMPLETE** — enumerate what "credible science" now means (single validated config, pretrained encoders, per-family recipes + mIoU selection, thermal-domain preprocessing, reproducible locks, package-clean imports) and state the **one remaining cross-phase item: real-data GPU-box validation** (`docs/phase1_realdata_checklist.md` — CUDA lock, `--force-preprocess`, pretrained/recipes). §9: tick T3.7; frontier → **Phase 4 (T4.1 Wilcoxon + enhancements)**. Save this prompt (R9).

## Standing guardrails
Docs-only session — **no code changes** (if reconciling the README seems to require a code edit, that is a real bug: stop, report, propose a follow-up UB row, do not silently fix in a docs session). Do not touch the already-correct tables (R3). No fabricated metric numbers in the expected-output example (R10). Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Frontier: Phase 3 COMPLETE → Phase 4 opener (T4.1).** Real-data GPU-box validation still pending (the only thing a sandbox cannot close).
8. Ledger/§2 hunk marking Phase 3 complete; the pasted fresh-venv install + quickstart-smoke transcript; the grep proving no stale invocations/module refs; the list of README sections changed vs left-untouched (proving R3 restraint).
9. **Prompt for Session 16 — Phase 4 opener, T4.1 (M8): statistical rigor.** The report generator must, before any "model A outperforms B" phrasing, compute across the K folds (and the held-out TEST set) **mean ± std** and a **paired Wilcoxon signed-rank** test (report the p-value); absent significance, the report says "comparable within noise" (M8). AC: a unit test feeds two known per-fold metric vectors and asserts the reported p-value + the wording branch (significant vs within-noise); the benchmark report gains a significance column/section; no fabricated data (the real comparison runs on the GPU box). Then survey remaining Phase-4 items (T4.2 TensorBoard, T4.3 `pyproject` entry points `ubench train|benchmark`, T4.4 modern AMP, T4.5 ONNX export, T4.6 mypy in CI) and propose the Phase-4 sequencing. One item; Session Entry Protocol first.

## One meta-note for the human (not a task)
After Session 15, the automated remediation has taken the repo as far as a sandbox can: 26 ledger defects closed, Phases 1–3 done, CI green. The **real-data validation run on your GPU box** is now the highest-value action — it is what converts "the pipeline is correct and reproducible" into "the benchmark produces trustworthy numbers," and several fixes (pretrained-vs-scratch, per-family recipes, fixed_range normalization, the CUDA lock) have only ever been exercised on synthetic CPU data. The checklist is ready.

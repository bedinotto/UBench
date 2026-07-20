# Session 14 — T3.6 (UB-21): package-clean imports

*Supersedes the self-drafted Session-14 prompt from Session 13's report — adopt its shape and its ▲ registration-drop audit. Six fixes folded in (▲), grounded in the actual import inventory below; ▲A is a real trap: the self-draft's own acceptance criterion (`python -m codes.transunet`) cannot pass as the code stands, so "fix imports" and "prove via `-m`" need reconciling or the session ends red.* One ledger item (UB-21), surgical (R3). Read CLAUDE.md §5 (UB-21), §11, R3/R4/R9, §9 T3.6.

Branch: `phase-3/imports` from current `main` (**`578c442`** — Session 13 merged + pushed; CI green). Linear FF; no stacking. Push: gh-credentialed HTTPS.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. Ledger: UB-20 fully closed (20a `@c85a5c5`, 20b `@0cc7724`); UB-21 `OPEN`; §2 says next T3.6.
2. `UBENCH_PRETRAINED=0 python -c "import codes.main_pipeline"` → **5** models register; `python -c "from codes.config_schema import load_config; load_config('codes/config.yaml')"` ok.
3. Full suite green (~141, 2 net-skips) — paste count + wall time; `ruff check codes/` clean.
4. Verify CI green for main@`578c442` (`gh run list --branch main`).

## The verified inventory (this is the whole job — no more, no less; R3)
- **Relative imports** (`from .`): `unet_v2.py:31`, `transunet.py:168`, `swin_unet_plus_plus.py:213` (top-level); `swin_pretrained.py:28-29`, `transunet_pretrained.py:31-32` (inside functions). All import `model_registry` / `pretrained_stem`.
- **`sys.path.insert` hacks:** `main_pipeline.py:21-22`, `preprocess_data.py:15-16`, `setup.py:14`.
- **`try/except` dual-import fallbacks:** `benchmark_models.py:20+`, `unified_training.py:20+` (try `from codes.x` … except relative/bare).
- **Registration by import-for-side-effect:** `main_pipeline.py:42-46` imports the 5 model modules purely so `@register_model` runs; `get_registered_models()` feeds the CLI choices (UB-26). **This is the fragile seam** (▲D).
- **`__init__.py`:** present at `codes/` and `codes/tests/` (so `codes` is already a real package — good; the job is consistency, not package creation).
- **9 modules have `__main__` blocks** (`main_pipeline`, `preprocess_data`, `unified_data`, `unified_training`, `benchmark_models`, `hardware_detector`, `extract_data`, `generate_boxes_polygons`, `setup`).

## ▲A — Reconcile "absolute imports" with "runnable via `-m`" BEFORE coding
Standard: **absolute imports rooted at `codes.`** everywhere (R7). But absolute imports mean a module run as a **bare script** (`python codes/transunet.py`) breaks (no `codes` parent on path) — while `python -m codes.transunet` works. So the two must move together:
- Convert every `from .x`/bare import to `from codes.x`.
- The acceptance criterion becomes **`python -m codes.<model>`** (module execution), **not** `python codes/<model>.py` (bare script). State this in §4.8/README — it is the correct invocation for a package and it is what removes the need for `sys.path` hacks. ▲ If any `__main__` block or doc/`run.sh` currently calls a bare `python codes/foo.py` that relies on the removed `sys.path` insert, convert that call site to `python -m codes.foo` in the same commit, or it breaks silently. Grep `run.sh`/`run.bat`/README for `python codes/` and fix every occurrence.

## ▲B — Model modules need a `__main__` self-test to satisfy the AC (they may not have one)
The inventory shows `unet_v2/transunet/swin_*` are **not** in the `__main__` list. The AC "`python -m codes.<model>` prints a forward shape" requires each model module to *have* a `__main__` block that builds the model (with `pretrained=False`/`UBENCH_PRETRAINED=0` — no downloads, R10/offline) and prints `model(torch.zeros(2,1,256,256)).shape`. Add a minimal, uniform self-test block to each of the 5 model modules as part of this session (it is the AC's mechanism, and a genuine smoke for the import refactor). Keep it offline and CPU-only.

## ▲C — The try/except fallbacks: delete, don't "improve"
`benchmark_models.py` / `unified_training.py` wrap imports in `try: from codes.x … except: <relative/bare>`. Once imports are uniformly `from codes.x` and always run as a package, the fallback is dead — **delete the except branch entirely** (R4: no silent fallback paths). Do not keep a "just in case" bare-import arm; that is the very ambiguity UB-21 exists to remove.

## ▲D — Registration-drop guard (the self-draft's audit, sharpened into a test)
`@register_model` fires only when the module is imported; `main_pipeline.py:42-46` does that for its side effect. After the refactor a reordered/dropped import would **silently** deregister a model and shrink the CLI choices — no error, wrong benchmark. Guard it with a test, not just an audit: **`test_all_models_registered`** asserts `set(get_registered_models())` equals the expected 5 keys (`unet, transunet, swin_unet_plus_plus, swin_pretrained, transunet_pretrained`) after importing `codes.main_pipeline`. Red-first is impossible here (it passes now), so add it as a **regression pin** and confirm it stays green through the refactor. If registration currently depends on import order in `main_pipeline`, consider a single explicit `codes/models/__init__.py`-style aggregation import, but only if it does not balloon scope (R3) — otherwise keep the explicit imports and rely on the pin.

## Tests / budget / push
Red where practical (the `-m` self-tests are new and demonstrable; the registration pin is a regression guard). No new subprocess pipeline runs; smoke unchanged. `ruff check codes/` must stay clean (watch for F401 "imported but unused" on the registration side-effect imports — mark them `# noqa: F401` with a comment, since they are imported *for* the side effect; do not delete them). Measure wall time; split CI only if the cap breaks. Push via gh HTTPS; paste `gh run list` conclusion.

## Docs
Ledger UB-21 → `FIXED@<sha>`. §4.8 (dev/scripts) + §5 UB-21 row + README: state absolute-imports-rooted-at-`codes`, package execution via `python -m codes.<module>` (not bare scripts), `sys.path` hacks removed, dual-import fallbacks removed. ▲ Note the `run.sh` invocation style if any call site changed. Save this prompt (R9). §2 frontier → **T3.7 (README full reconciliation)** — the last Phase-3 task, which will finally fold in every doc-vs-code drift accumulated across T3.1–T3.6 (config keys, model table incl. pretrained, normalization, recipes, lock install, `-m` execution).

## Standing guardrails
Excluded: README *full* reconciliation (T3.7 — this session touches only the import-related doc lines), Wilcoxon/T4.x, any behavior change (imports are a pure refactor — the smoke's numbers must be **bit-identical**; if anything numeric moves, you changed more than imports, stop). Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: **T3.7 (README reconciliation) — the final Phase-3 session**; real-data GPU-box validation still pending (CUDA lock + `--force-preprocess` + pretrained/recipes).
8. Ledger hunk (UB-21), a paste of `python -m codes.transunet` (and the other four) printing forward shapes with no ImportError, the `test_all_models_registered` result, and confirmation the smoke numbers are unchanged (pure refactor).
9. Prompt for Session 15 (T3.7, UB-04-doc-tail + all accumulated drift): reconcile README end-to-end to the code as it now stands — quickstart with `--force-preprocess` and `uv` lock install; the 5-model table (from-scratch + pretrained, with the honest UB-16/17 baseline caveats); leave-subjects-out CV + held-out `test_subjects`; per-family recipes + selection-by-mIoU; `fixed_range` normalization default; `python -m codes.<x>` execution; config-key table matching `config_schema.py`. AC: every command in the README executed verbatim in a fresh venv during review; no claim describes unimplemented behavior (R9). One item; Session Entry Protocol first. Note this likely closes Phase 3 → next is Phase 4 (T4.1 Wilcoxon + enhancements).

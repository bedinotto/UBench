# Session 12 — T3.4 (UB-19, M7): thermal-domain preprocessing

*Fourth Phase-3 session. One ledger item (UB-19). Makes the preprocessing/augmentation physically honest for thermal data: absolute-temperature-preserving normalization, physically-plausible augmentation, and a vectorized raw→°C conversion. Read CLAUDE.md §4.2 (config), §8 M7/M9, §5 (UB-19), §11, R1–R10, §9 T3.4.*

Branch: `phase-3/thermal-preprocessing` from current `main` (**`f498ef2`** — Session 11 was merged AND pushed this session; CI ran green on main). Linear FF; no stacking.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. Ledger: UB-18 `FIXED@b74ef7d`; UB-19 OPEN; §2 says next T3.4. UB-16/17/12 FIXED.
2. `UBENCH_PRETRAINED=0 python -c "import codes.main_pipeline"` registers 5 models; `python -c "from codes.config_schema import load_config; load_config('codes/config.yaml')"` ok.
3. Full suite green (**117 collected: 115 pass + 2 network-skip, ~4:44**) — paste count + wall time. `ruff check codes/` clean.
4. Verify CI for main@`f498ef2` (Session-11 push) is green (`gh run list --branch main`); if red, stop and report.
5. Carry-over: pipeline not bit-reproducible on CPU (deterministic mini-trajectory for numbers proofs); config is the single validated authority (add keys to `config_schema.py`, never a second file); `UBENCH_PRETRAINED` gates downloads; **the pretrained stems were trained on ImageNet statistics — a normalization change alters what they see, so record it (M9)**.

## The defects (UB-19, verified in `utils.py`/`unified_data.py`)
- **Per-image min–max normalization destroys absolute temperature** — the modality's core signal (`get_stats_info` itself reports the °C range). Two frames at different absolute temperatures map to the same [0,1].
- **`RandomBrightnessContrast` is physically dubious on thermal** (multiplicative contrast on a temperature field has no sensor analogue).
- **`np.vectorize` raw→°C is a per-pixel Python loop** — slow on 640×480.

## The work (M7) — per-concern commits (R6)

**Commit 1 — normalization switch (config-driven):**
- Add a `preprocessing` (or extend an existing) section to `codes/config.yaml` behind the pydantic schema: `normalization: per_image_minmax | fixed_range` with `fixed_range_celsius: [20.0, 40.0]` (recommended default per M7 — but ⚠ **decide the default deliberately**: `fixed_range` preserves absolute temperature and is correct, yet it *changes the numbers vs every prior run and vs the pretrained stems' expected input*. Options: (a) ship `fixed_range` as the new default and treat it as an intentional, documented numbers-move — the "credible science" phase is where this belongs; or (b) keep `per_image_minmax` default + `fixed_range` opt-in to avoid moving numbers this session. **Recommend (a) with the deterministic-trajectory diff pasted to show the exact change is only the normalization**, but state the choice and rationale.)
- Implement both modes where normalization currently happens; `fixed_range` maps °C linearly from `[lo,hi]` to `[0,1]` (clip outside). Unit-test both modes on crafted inputs (a hot vs cold frame map differently under fixed_range, identically under per_image_minmax).
- ▲ Record the effective normalization mode + range in `run_metadata.json` and the benchmark report (M9) — it changes what the pretrained stems see.

**Commit 2 — physical augmentation:**
- Replace `RandomBrightnessContrast` with a physically-plausible **additive sensor offset (±0.5 °C drift)** + **Gaussian noise**; keep flip/affine. Migrate the deprecated `ShiftScaleRotate(...)` → `A.Affine(...)` **within the pinned `albumentations<2.0`** (verify the exact API on the installed version; do not bump the pin — that is T3.5). The additive offset must be applied in the same units the pipeline normalizes in (apply before/after normalization consistently — decide and document).
- Unit-test: augmentation preserves the mask; the offset is additive (mean shift) not multiplicative; determinism holds under the seeded pipeline (T2.5).

**Commit 3 — vectorized raw→°C:**
- Replace `np.vectorize(_raw_to_celsius)` (`Config.RAW_TO_CELSIUS`) with `(raw.astype(np.float32) / 100.0) - 273.15` applied array-wise. Keep `_raw_to_celsius` for scalars if still referenced. AC: identical output to the old path on a range of raw values **and ≥100× faster on a 640×480 array** (timed, generous bound). Unit-test both.

## Tests (red-first) — both normalization modes; physical-augmentation properties; conversion equivalence + the ≥100× timing bound; smoke stays green (trio unchanged structurally, but now normalizes via the configured mode — assert the mode is recorded).

## Suite budget + push & CI — measure (was 4:44). Push via the working method this session: `git checkout main && git merge --ff-only <branch> && git -c credential.helper='!gh auth git-credential' push https://github.com/bedinotto/UBench.git main` (gh token has repo scope; SSH remote does NOT work in the sandbox). Paste the `gh run list` conclusion.

## Docs — ledger UB-19 → `FIXED@<sha>`. §4.x + §8 M7 reconciled (normalization modes, physical augmentation, vectorized conversion). §2 frontier → T3.5 (UB-20b: lockfiles + `setup.py` venv/CUDA index). README preprocessing note + config table gains the normalization keys. Save this prompt (R9).

## Standing guardrails
Excluded: lockfiles/CUDA index (T3.5); import restructure (T3.6); README full reconciliation (T3.7); Wilcoxon (T4.1). Also **out of scope but FLAG loudly** (found in T3.3): `main_pipeline.py` argparse `choices=['unet','transunet','swin']` rejects `swin_pretrained`/`transunet_pretrained` on the CLI though the README advertises them — a ~1-line T3.2 gap; propose fixing it in T3.7 (README reconciliation) or as a tiny standalone `fix(cli):`. No fabricated numbers (R10). Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.5 next; real-data GPU-box validation still pending (now covering pretrained + per-family recipes + the chosen normalization mode).
8. Ledger hunk (UB-19), the normalization-mode decision + its deterministic before/after diff, and the conversion speedup factor measured.
9. Prompt for Session 13 (T3.5, UB-20b): `uv` lockfiles (CPU lock for CI, CUDA lock for training); `setup.py` creates a venv, drops `--force-reinstall --no-deps`, resolves the current CUDA index; CI installs from the lockfile. AC: clean-container install from lockfile → smoke green; record lockfile hash in `run_metadata.json` (the T2.5 TODO). One item; Session Entry Protocol first.

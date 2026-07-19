# Session 12 — T3.4 (UB-19, M7): thermal-domain preprocessing

*Supersedes the self-drafted `prompts/phase3_session12.md` — right intent, but it misses the architectural fact that decides the whole design: **normalization is baked into `data/processed/*.npy` at preprocess time** (`preprocess_data.py:72` → `utils.preprocess_thermal_image` → `normalize_thermal`), so a config switch "where normalization happens" silently does nothing to already-processed data. Five fixes folded in (▲).* Read CLAUDE.md §4.2, §8 M7/M9, §5 (UB-19), R4, R5, R6, §9 T3.4.

Branch: `phase-3/thermal-preprocessing` from current `main` (**`43c2c02`** ▲ — the draft's `f498ef2` is stale; the Session-12-prompt commit moved the tip). Linear FF; no stacking. Push method that works from this sandbox: gh-credentialed HTTPS (the draft's one-liner), not SSH.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
Adopt the draft's five checks (ledger states, 5-model registration, config gate, full suite ~117 @ ~4:44 pasted, CI green for the Session-11 push via `gh run list`). Add:
6. ▲ Confirm the report-vs-tree discrepancy: Session 11's report said the CLI gap was "flagged in the ledger" — **no UB-26 row exists**. Claims about the ledger require the row (R1/R9); Commit 0 repairs this.

## Commit 0 ▲ — `fix(cli)`: registry-derived model choices (UB-26)
`main_pipeline.py:596` hardcodes `choices=['unet','transunet','swin']`, rejecting `swin_pretrained`/`transunet_pretrained` that the README and the GPU-box checklist advertise — it blocks the pending human validation run, which is why it gets pulled into this session rather than waiting for T3.7. Fix by **deriving choices from the model registry** (delete the hand-maintained list — the same second-authority bug class as UB-02/UB-05; keep any short-alias mapping like `swin` as an explicit alias table next to the registry, not a parallel truth). Add ledger row **UB-26** and flip it `FIXED@<sha>` in the same commit; update §3.2's model-choices line. Test: argparse accepts every registered key; an unknown key still errors.

## Commit 1 — normalization: pick the architecture first ▲A
Two designs; **recommended = (i)**, and the choice binds Commit 2:

**(i) Load-time normalization (recommended).** Preprocessing stores **Celsius float32** in the `.npy` (no normalization baked); the Dataset normalizes per config at `__getitem__` via one function. The mode switch becomes runtime-pure — no stale-data class of bug — and augmentation can act in physical units (▲B). Cost: the processed-data contract changes → bump a **preprocess version** and require one rebuild.

**(ii) Baked normalization + staleness guard (fallback only if (i) is blocked).** Keep normalizing at preprocess time, but the mode then lives in the data, so it MUST be recorded and enforced.

**Either way — `preprocess_manifest.json` is mandatory (R4):** written next to `metadata.csv` with `{preprocess_version, normalization_mode, fixed_range_celsius, created_at}`; at load, a mismatch with the active config (or a missing manifest = legacy data) **hard-errors** with the exact `--force-preprocess` remediation in the message. A silent config/processed-data mismatch is this session's UB-02-equivalent — the test for it is non-negotiable: build with mode A, flip config to mode B, assert the load raises.

Config (schema-gated): `preprocessing.normalization: per_image_minmax | fixed_range`, `preprocessing.fixed_range_celsius: [20.0, 40.0]`. **Default = `fixed_range`** per the draft's option (a) — Phase 3 is where numbers move deliberately; disclose it. (The synthetic fixture's raw values span exactly 20–40 °C, so the smoke fits the default natively.) ▲ **Unify the duplicate normalizer:** `unified_training.py:896` (`ThermalFaceDetector.normalize_thermal`) is a second inference-path copy — route it through the same config-driven function (R5), else inference normalizes differently than training. Record mode + range in `run_metadata.json` and the benchmark report (M9 — it changes what the pretrained stems see).

**Proof sequencing ▲ (the draft's single diff conflates two changes):** after Commit 1 *only*, run the deterministic mini-trajectory main-vs-branch **under `per_image_minmax`** → bit-identical (proves the refactor is behavior-preserving); then flip to `fixed_range` → numbers move (the intended, disclosed change). Paste both.

## Commit 2 — physical augmentation (units follow ▲A)
Replace `RandomBrightnessContrast` with **additive sensor drift (±0.5 °C) + Gaussian noise (σ≈0.1 °C)**; keep flip; migrate `ShiftScaleRotate` → `A.Affine` within the pinned `albumentations<2.0` (verify the installed API; do not bump the pin — T3.5). ▲B **Apply drift/noise in Celsius space, before normalization** — under design (i) this is exact (augment the Celsius array, then normalize). "±0.5 °C" has no honest meaning after per-image min-max (the physical scale differs per image); if fallback (ii) was chosen, physical augmentation is exact only in `fixed_range` (0.5 °C = 0.025 in [0,1]) and must be documented as approximate or disabled under minmax. Config keys under `preprocessing.augmentation` (drift, noise, affine params) with the new pipeline as defaults. Tests: mask untouched by intensity ops; drift is additive (pure mean shift); seeded determinism holds — same seed → identical augmented batch twice (T2.5 infra). After this commit the invariant is **determinism**, not comparability to main — augmentation changes trajectories by design.

## Commit 3 — vectorized raw→°C (as drafted)
Replace `Config.RAW_TO_CELSIUS = np.vectorize(...)` with the array expression `(raw.astype(np.float32)/100.0) - 273.15`; the call site is `load_thermal_image_from_tiff` (`unified_data.py:332`). Tests: exact equality vs the scalar function across a raw-value range; **≥100× on a 640×480 array** (generous timed bound).

## Tests / budget / push / docs
Red-first throughout; no new subprocess runs; the smoke trio is unchanged structurally (it re-preprocesses fresh in tmp dirs, so design (i)'s rebuild is automatic there). Measure suite wall time; split CI only if the cap breaks. Push via gh HTTPS; paste `gh run list` conclusion. Docs: ledger UB-19 → `FIXED@<sha>` (+ UB-26 from Commit 0); §4.x/§8 M7 reconciled (normalization architecture, manifest guard, physical augmentation, vectorized conversion); §2 frontier → T3.5; **`docs/phase1_realdata_checklist.md` gains `--force-preprocess` as a required step** ▲ (real processed data predates the manifest and will hard-error by design — the checklist must say so, or the guard will read as a regression on the GPU box). README config table updated. Save this prompt (R9).

## Standing guardrails
Excluded: lockfiles/CUDA index (T3.5/UB-20b), import restructure (T3.6), full README reconciliation (T3.7), Wilcoxon (T4.1). No fabricated numbers (R10). Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.5 next; real-data GPU-box validation pending — now requiring `--force-preprocess` and covering pretrained + recipes + the fixed_range default.
8. Ledger hunks (UB-19, UB-26), the two-stage normalization proof (bit-identical refactor, then the disclosed fixed_range move), the staleness-guard test name, and the measured conversion speedup.
9. Prompt for Session 13 (T3.5, UB-20b) as the draft specced — `uv` lockfiles (CPU for CI, CUDA for training), `setup.py` venv + current CUDA index, CI installs from the lockfile, lockfile hash into `run_metadata.json` (closing the T2.5 TODO) — plus one addition ▲: the albumentations `<2.0` pin and the `A.Affine` migration from this session set up the 2.x unpin decision; Session 13 must decide it explicitly (stay pinned in the lock, or migrate-and-unpin) rather than inherit it silently.

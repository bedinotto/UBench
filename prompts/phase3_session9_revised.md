# Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener)

*Supersedes the self-drafted `prompts/phase3_session9.md` — same scope, FF target corrected and seven fixes folded in (▲). Two of them fix scope errors that would otherwise **move the numbers** or **collide with T3.4**, which is exactly what this Phase-3 session must avoid.* The first "make the **science** credible" session. One ledger item, a small set of focused commits (plus docs). Read CLAUDE.md §4.2 (config truth), §5 (UB-12), §8, §11 (anti-patterns), R3, R4, R6, R9, and §9 T3.1 before acting.

Branch: `phase-3/config-consolidation` from `main` **after** main is fast-forwarded to the Session-8 tip **`f3f7a1c`** ▲ (the current tip — *not* `e50920f`; it moved when the graphify and Session-9-prompt commits landed). Linear FF history. **If main lacks the Session-8 commits, halt after Step 0 and ask the human to FF. Do not stack.**

Disk pre-flight: green suites self-clean (`tmp_path_retention_policy=failed`); clear `/tmp/pytest-of-*` before long runs if near capacity.

## Step 0 — SESSION ENTRY PROTOCOL (§10)

1. `grep -rln "test_determinism\|normalize_thermal" codes/tests/` → the determinism + normalize tests exist (adapt to actual filenames).
2. Run those test files → green.
3. Ledger: UB-15/22/24/25 `FIXED`; UB-20 shows **20a FIXED / 20b OPEN**; UB-09/10/11 `FIXED`; §2 says "Phase 2 complete".
4. `test -f codes/inference_comparison.py` → absent; `grep -rn "calculate_iou\|train_iou_metric" codes/` → nothing.
5. Full suite `pytest codes/tests -q` → green (~68 tests); paste count + wall time.

**If any check fails:** stop, report; if Session 8 was never executed, run `prompts/phase2_session8.md` first. Do not build on an unverified base.

## The exact current behavior T3.1 must preserve (do not move these numbers)

Verified in `unified_training.py`: `CombinedLoss(ce_weight=0.5, dice_weight=0.5)`; `Adam(lr=config.LEARNING_RATE)` with torch-default `betas=(0.9, 0.999)`, `weight_decay=0`; `ReduceLROnPlateau(mode='min', patience=5, factor=0.5)`; `class_weights` permanently `None` (uniform — the `hasattr(config,'CLASS_WEIGHTS')` branch has no producer). These hardcoded values become the schema **defaults** so a run's results are bit-identical before and after.

## Step 1 — the work (schema + wire/delete). Keep commits per-concern (R6); suggested split below.

**Defect (ledger/§4.2):** root `config.yaml` loaded by nothing (dead); `codes/config.yaml` is the real one but read with unchecked `.get(...)` defaults, so a typo'd key is silently ignored; `training.batch_sizes` unconsumed; `CLASS_WEIGHTS` has no producer; scheduler/optimizer hardcoded.

**Commit 1a — schema + delete dead root file:**
- **Delete the root `config.yaml`** (dead per §4.2). Do **not** reintroduce a second config file (§11).
- ▲ **Add `pydantic>=2` to `requirements/requirements.txt`** (new dep; note v2 API — `model_config = ConfigDict(extra="forbid")`). Put a typed schema over `codes/config.yaml` that **fails on unknown keys and wrong types**; `Config` loads through it so a bogus key raises at startup instead of being silently ignored. Preserve the existing env-var overrides (`NUM_EPOCHS`, `K_FOLDS`, `TEST_SUBJECTS`, `UBENCH_DETERMINISTIC`).

**Commit 1b — resolve `codes/config.yaml`'s own dead `batch_sizes`:**
- ▲ The shipped `training.batch_sizes` keys are `unet/transunet/swin` — the **old** short keys (T1.3 renamed the hardware_detector keys to registry names; `swin` ≠ `swin_unet_plus_plus`) — and they are unconsumed (the code reads `hardware_profile.batch_sizes[key]`, never config's). **Delete them.** `hardware_detector` is the single batch-size authority (UB-05's whole point); re-adding a config override would reintroduce the dual-authority that caused UB-05. If a manual override is genuinely wanted it is a separate feature, not this dead-key cleanup.

**Commit 1c — wire the currently-hardcoded knobs (defaults = current values):**
- Add `loss` (`ce_weight: 0.5`, `dice_weight: 0.5`), `optimizer` (`name: adam`, `weight_decay: 0.0`, `betas: [0.9, 0.999]`), and `scheduler` (`name: reduce_on_plateau`, `patience: 5`, `factor: 0.5`) sections to `codes/config.yaml`, and read them in `unified_training.py` instead of the hardcoded literals. ▲ **Shape these keys so T3.3's per-family recipes can extend them cleanly** (e.g. an optional per-model override map later), but ship only the current single global recipe now — do **not** change any recipe (T3.3). Note: resume restores `scheduler_state_dict`/`optimizer_state_dict`, so scheduler/optimizer config changes apply to **fresh** runs, not resumed ones (expected).
- ▲ **`class_weights` — wire the mechanism but default OFF.** Add `loss.class_weights: null`. `null` → current uniform behavior (numbers unchanged). Support `"balanced"` (computed from **train-fold** class frequencies, recomputed per fold from the training split only — never the val/test split) and an explicit list. **Default stays `null`** so T3.1 does not move the numbers; enabling balanced weighting is an opt-in a later experiment/T3.3 uses. (The self-draft's "compute class_weights from frequencies" — if enabled by default — would change the loss and violate the no-numbers-moved rule; defaulting off resolves that.)
- ▲ **Do NOT wire augmentation or a hardware section.** The dead augmentation/hardware keys lived only in the now-deleted root file, so they vanish with it. The hardcoded albumentations pipeline stays as-is until **T3.4** redesigns it (physical augmentation, normalization switch, `A.Affine` migration); adding augmentation config now would be ripped out next phase. `hardware_detector` remains the owner of device/batch/worker/AMP.

**Rule across 1a–1c (R4):** no key may remain that does nothing — every key in the final `codes/config.yaml` is either consumed or gone. That is the UB-12 failure mode being closed.

**Tests (`codes/tests/test_config.py`, red-first):** an unknown key raises; a wrong-typed value raises; the shipped `codes/config.yaml` validates; each wired key demonstrably changes behavior (e.g. `scheduler.patience` alters the constructed scheduler; `loss.ce_weight` alters the loss value; `loss.class_weights: "balanced"` produces non-uniform weights while `null` produces `None`). Unit-level; the smoke covers end-to-end.

▲ **Prove the numbers did not move (concrete, using T2.5 determinism):** run the smoke on `main` (pre-branch) and on this branch with the **same fixed seed**, and diff the per-epoch train/val losses in the metrics JSON — they must be **identical**, not merely "similar." Paste the comparison. This is the Phase-3 regression guard.

Verify each commit: full suite + ruff green; smoke green.

## Step 2 — Docs
Ledger UB-12 → `FIXED@<sha>`. §4.2 rewritten to the single validated authority (no dead root file; unknown keys raise; list which keys are now wired: loss/optimizer/scheduler/class_weights; note augmentation + hardware remain code-owned pending T3.4). §2 frontier → Phase 3 underway (next: T3.2 pretrained encoders). README config section reconciled to `codes/config.yaml`. Save this prompt file (R9).

## Step 3 — Suite budget + push & CI
Measure full-suite wall time (was 4:31 at Session 8; +pydantic import is negligible). If it exceeds the §7.1 five-minute cap, split CI into `fast`/`smoke` (`chore(ci):`) and amend §6.4/§7.1. Push; paste Actions status/URL or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails
Excluded temptations: pretrained encoders (T3.2); per-family recipes (T3.3 — T3.1 *exposes* optimizer/scheduler as config but must **not** change the recipes or their values); thermal preprocessing / augmentation redesign (T3.4 — do not wire augmentation now); lockfiles + CUDA index (T3.5, UB-20b); import restructure (T3.6). Tool-generated edits (graphify) → separate `chore(tooling):`. R1/R3/R4/R6/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Frontier:** Phase 3 progress (T3.2 pretrained encoders next) + real-data validation status (still pending the human GPU box).
8. Ledger diff hunks (UB-12 flip; §2/§4.2 updates) + the pasted identical-loss before/after comparison.
9. **Prompt for Session 10** covering **T3.2 (UB-16/17, M5)** — pretrained encoders via timm/MONAI (TransUNet R50+ViT-B/16; Swin via `SwinUNETR`), 1-channel stem adaptation by **summing pretrained RGB kernels**, register `transunet_pretrained` / `swinunetr` as new model keys (do not delete the from-scratch variants — the benchmark can compare pretrained vs scratch). AC: forward-shape tests; param counts logged; batch-size parity test (§UB-05) forces the new keys to have batch sizes; smoke covers the new keys. ▲ Flag the network/dependency reality: timm/MONAI weight downloads need connectivity, and CI/smoke on CPU must either use a tiny/random-weight path or skip the download — decide and encode. One item; Session Entry Protocol first.

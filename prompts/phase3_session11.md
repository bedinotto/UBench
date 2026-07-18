# Session 11 — T3.3 (UB-18, M4): per-family recipes + selection by val mIoU

*Third Phase-3 session. One ledger item (UB-18). T3.1 already **exposes** `optimizer`/`scheduler` as validated config sections (single global recipe); T3.2 added the two pretrained transformer keys. T3.3 makes the recipe **per model family** and switches checkpoint selection to the headline metric. Read CLAUDE.md §4.2 (config), §4.6 (models), §8 M4/M9, §5 (UB-18), R1–R10, §9 T3.3.*

Branch: `phase-3/per-family-recipes` from current `main`. Confirm main contains the T3.2 commits (tip `chore(tooling): graphify update after T3.2`, `828ce41`, or later). **If not merged/pushed, halt and hand the FF+push to the human** — Session 10 could not push (sandbox has no GitHub SSH key; PyPI/HF hub reachable, GitHub SSH not). Linear FF; no stacking.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. Ledger: UB-12 `FIXED`, UB-16/17 `FIXED@faa9a2d`; §2 says next T3.3. Root `config.yaml` absent.
2. `python -c "import codes.main_pipeline"` registers **5** models: unet, transunet, swin_unet_plus_plus, swin_pretrained, transunet_pretrained (`UBENCH_PRETRAINED=0` to stay offline).
3. Full suite green (**100 collected: 98 pass + 2 network-skipped, ~4:44 wall**) — paste count + time. `ruff check codes/` clean.
4. Carry-over facts: pipeline NOT bit-reproducible on CPU (use the deterministic mini-trajectory technique for any numbers-unchanged proof); trainer construction needs `config.LOSS/OPTIMIZER/SCHEDULER`; pretrained models gated by `UBENCH_PRETRAINED` (tests set it 0); `timm>=1.0` is a dep; the network-gated `@pytest.mark.pretrained` test needs `UBENCH_ALLOW_DOWNLOADS=1`.

## The defect (UB-18)
`unified_training.py` applies **one** recipe to every architecture (Adam lr=1e-4 + plateau; T3.1 made these config-driven but still a single global recipe), and selects the "best" checkpoint by **val loss** while the headline metric is **mIoU** (M4). Identical hyperparameters are not fair across families (a from-scratch/pretrained ViT-B wants AdamW + warmup + cosine, not plain Adam).

## The work (M4) — per-concern commits (R6)
**Commit 1 — per-family recipe resolution in config:**
- Extend the `optimizer`/`scheduler` schema (`codes/config_schema.py`) with an **optional per-family/per-model override map** on top of the existing global default (shape it so a model key or a family name selects a recipe; keep the current global as the default so unset = today's behavior). Decide family assignment explicitly: `unet` → **cnn**; `transunet`, `swin_unet_plus_plus`, `swin_pretrained`, `transunet_pretrained` → **transformer**. Encode the mapping where it is discoverable (not hard-coded in the trainer) — e.g. a `family:` field per model or a `families:` table in config, validated.
- Wire **AdamW** (add to the accepted optimizer names — T3.1 hard-raises on unknown, so extend the allow-list) and a **warmup+cosine** scheduler (add to accepted scheduler names). CNN family keeps Adam + plateau (numbers unchanged for unet). Transformer family: AdamW wd≈0.05 + linear warmup ~5% of steps + cosine, grad-clip 1.0 (grad-clip already exists). Fully disclosed in the run metadata + report (M9).
- Warmup needs total step count (len(train_loader)*num_epochs) — compute in the trainer where both are known.

**Commit 2 — selection by val mIoU (M4):**
- The trainer currently checkpoints "best" by val loss; switch the **headline** selection to **val mIoU** (higher is better). `validate()` already returns (loss, mIoU, dice) since T2.1. Keep saving loss for logging, but the "best" checkpoint + the reported best is mIoU. `benchmark_models.py` selects the best fold-model consistently. Update any test that assumes best-by-loss.
- ⚠ Numbers WILL move for the reported "best" (by design — M4). Prove the *training dynamics* of the CNN recipe are unchanged (unet loss trajectory identical via the deterministic mini-trajectory) so the change is *only* selection + transformer recipe, not an accidental CNN regression.

**Commit 3 — resume compatibility:**
- Resume restores `optimizer_state_dict`/`scheduler_state_dict` (T3.1 note). A recipe change (Adam→AdamW, plateau→cosine) is incompatible with a checkpoint trained under the old recipe. Decide + encode: either (a) resume keeps the checkpoint's optimizer/scheduler and warns that a recipe change only applies to fresh runs, or (b) detect a recipe mismatch on resume and hard-error (R4). Do NOT silently load an AdamW state into an Adam optimizer. Add a `test_resume`-style guard.

## Tests (red-first)
`test_config.py`/new: per-family map validates; unknown family raises; AdamW + cosine constructible; a transformer-family model gets AdamW+warmup while unet gets Adam+plateau (assert on the constructed optimizer/scheduler types). Selection-by-mIoU: a crafted two-epoch history where loss and mIoU disagree picks the mIoU-best checkpoint. Warmup: LR rises then falls across steps (unit-level on the scheduler). Resume guard per commit 3. Transformer logs show warmup (assert in the single-step or a scheduler unit test). Full suite + ruff green; smoke green (canonical trio unchanged — but unet now logs its recipe).

## Suite budget + push & CI
Measure wall time (was 4:44). If over 5 min, split CI fast/smoke (`chore(ci):`) + amend §7.1. Push or `UNVERIFIED: remote CI` + FF/push one-liner.

## Docs
Ledger UB-18 → `FIXED@<sha>`. §4.6: per-family recipes + selection by val mIoU (not loss). §8 M4 note satisfied. §2 frontier → T3.4 (thermal preprocessing/augmentation, UB-19/M7). README config table gains optimizer/scheduler family overrides. Save this prompt (R9).

## Standing guardrails
Excluded: thermal preprocessing/augmentation (T3.4); lockfiles/CUDA index (T3.5); import restructure (T3.6); Wilcoxon (T4.1). Do NOT fabricate benchmark numbers (R10) — real per-family comparison is the GPU-box run. Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.4 next; real-data GPU-box validation still pending (now covering pretrained + per-family recipes).
8. Ledger hunk (UB-18) + the deterministic proof that the CNN/unet recipe is unchanged + a transformer-recipe log excerpt showing AdamW/warmup.
9. Prompt for Session 12 (T3.4, UB-19/M7): normalization switch (`per_image_minmax | fixed_range [20,40]°C`), physically-plausible thermal augmentation (additive ±0.5°C drift + Gaussian noise; migrate deprecated `ShiftScaleRotate`→`A.Affine` per pinned albumentations <2.0), vectorized raw→°C. AC: unit tests both normalization modes; conversion ≥100× faster on 640×480; augmentation is physical. One item; Session Entry Protocol first.

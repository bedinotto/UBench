# Session 10 — T3.2 (UB-16/17, M5): pretrained encoders via timm/MONAI

*Second Phase-3 session ("make the science credible"). One ledger item spanning UB-16 (ViT-B from scratch is known-degenerate) and UB-17 (hand-rolled Swin attention is broken); both are closed by adopting library-provided, pretrained encoders instead of re-implementing attention (M5). Read CLAUDE.md §4.6 (shared loop / adding a model), §8 M5 (pretraining policy) + M9, §5 (UB-16/17), R1–R10, and §9 T3.2 before acting.*

Branch: `phase-3/config-consolidation-t32` from `main` **after** main is fast-forwarded to the Session-9 tip. **Session 9 (T3.1) was NOT pushed** — the sandbox could not reach `git@github.com:bedinotto/UBench.git` (SSH timed out). Before starting, confirm main contains the five T3.1 commits (tip `6a42d27` "chore(tooling): graphify update after T3.1"). **If main lacks them, halt and FF+push first:** `git checkout main && git merge --ff-only phase-3/config-consolidation && git push origin main`. Do not stack on an unmerged base.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. Ledger: UB-12 `FIXED@72c59f4`; §2 says "Phase 3 underway, next T3.2". `test -f config.yaml` at repo root → **absent** (deleted in T3.1); `codes/config_schema.py` exists.
2. `python -c "from codes.config_schema import load_config; load_config('codes/config.yaml')"` → no error; a bogus key raises (spot-check).
3. Full suite `pytest codes/tests -q` → green (**86 tests**, ~4:24 wall). Paste count + time.
4. `ruff check codes/` clean.
**If any check fails:** stop, report; do not build on an unverified base.

## Context you must carry from Session 9
- **The pipeline is NOT bit-reproducible on CPU** (`use_deterministic_algorithms` is deliberately off, T2.5). Full-pipeline losses vary run-to-run by ~1e-2. To prove "numbers didn't move" for any change, use the **deterministic mini-trajectory** technique from S9 (Conv2d-only model + `torch.use_deterministic_algorithms(True)` + `set_num_threads(1)`), not a full-pipeline loss diff.
- **Trainer/loader construction now requires `config.LOSS`/`OPTIMIZER`/`SCHEDULER`** (schema sub-objects). Any new test double that builds a `UnifiedTrainer` must supply them (see `test_timing.py`/`test_config.py` for the pattern).
- `hardware_detector.batch_sizes` is the **single** batch-size authority; per-tier key parity with the model registry is test-enforced (`test_batch_size_keys.py`).

## Step 1 — the work (M5). Keep commits per-concern (R6).
**Defect:** `transunet.py` trains a ~100M ViT-B from scratch on ~1.8k images (UB-16 — guaranteed underperformance, invalidates fairness); `swin_unet_plus_plus.py` hand-rolls shifted-window attention **without** the attention mask / relative position bias, so the shift is a no-op and windows are permutation-invariant (UB-17). Do **not** patch the hand-rolled attention — replace with library encoders (M5, §11 forbids hand-rolling attention timm/MONAI already provide).

- **New model keys, do NOT delete the from-scratch variants.** Register `transunet_pretrained` (timm **R50+ViT-B/16**, `vit_base_r50_s16_224` or equivalent hybrid) and `swinunetr` (MONAI `SwinUNETR`). The benchmark then compares pretrained vs from-scratch head-to-head (that comparison is itself a result).
- **1-channel stem adaptation by summing pretrained RGB kernels** (M5): load the 3-channel pretrained stem, set `weight_1ch = weight_rgb.sum(dim=1, keepdim=True)` so the pretrained filters transfer to thermal single-channel input. Add a unit test asserting the summed-kernel weight equals the RGB sum (not re-initialized).
- Integrate through `UnifiedTrainer` (§4.6) — no parallel training loop. Import the new modules in `main_pipeline.py` so `@register_model` runs.
- **Batch sizes (UB-05):** add `transunet_pretrained` and `swinunetr` keys to **every tier** in `hardware_detector` (ViT-B + SwinUNETR are heavier — size conservatively for the 6 GB baseline). `test_batch_size_keys.py`'s registry-parity assertion will force this; update it if it enumerates keys.
- **Dependencies:** add `timm` and `monai` to `requirements/requirements.txt` (pin compatibly with pinned torch/torchmetrics). Note MONAI pulls extras — keep the CPU-install path working.

## Step 2 — the network/dependency reality (decide + encode) ▲
timm/MONAI **download pretrained weights** on first use → needs connectivity, which **CI and the CPU smoke do not have** (this sandbox couldn't even reach GitHub). Decide and encode ONE of:
- (preferred) a **`pretrained: bool` construction flag** (or `UBENCH_PRETRAINED=0`) so tests/smoke build the **same architecture with random weights** (no download), while real runs default to `pretrained=True`. The forward-shape and stem-sum tests then run offline; a separate, `@pytest.mark.skipif(no network)` test exercises the real download.
- or gate the pretrained path behind a cached-weights dir and skip when absent.
Whichever: **the smoke must cover the new keys on CPU without network** (random-weight path), and this must be documented. Do not let CI depend on an external download.

## Step 3 — tests (red-first where practical)
`codes/tests/test_models_forward.py` (planned in §7.1): each registered model incl. the two new keys maps `(2,1,256,256) → (2,10,256,256)`. Add: stem-sum correctness; param counts **logged** (M9) and sane (pretrained ViT-B ≫ from-scratch unet); batch-size parity forces the new keys. Extend the smoke to include `transunet_pretrained` and `swinunetr` (random-weight path). Full suite + ruff green; smoke green.

## Step 4 — Docs
Ledger UB-16 & UB-17 → `FIXED@<sha>` (or `PARTIAL` if scoped). §2 frontier → next T3.3 (per-family recipes). §4.6 note the new keys + that from-scratch variants remain for comparison. README model list. `requirements` note the new deps + the offline/random-weight test path. Save this prompt (R9).

## Standing guardrails
Excluded: per-family optimizer recipes (T3.3 — but note pretrained transformers will *want* AdamW+warmup; that's T3.3, do not change recipes here); thermal preprocessing/augmentation (T3.4); lockfiles/CUDA index (T3.5); import restructure (T3.6). Do **not** fabricate benchmark numbers (R10) — a real pretrained-vs-scratch comparison needs the GPU box. Tool edits (graphify) → separate `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report
Same format; include param-count log lines, the offline-test decision, the stem-sum test, and the Session-11 prompt (T3.3 — per-family optimizer/scheduler recipes from config, selection by val mIoU; the T3.1 config now *exposes* optimizer/scheduler, so T3.3 wires per-family overrides on top).

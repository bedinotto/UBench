# Session 10 — T3.2 (UB-16/17, M5): pretrained encoders — timm route

*Supersedes the self-drafted `prompts/phase3_session10.md` — same intent, with the stale push/FF premise removed and six fixes folded in (▲). The two that matter: ▲A the MONAI `SwinUNETR` route cannot satisfy M5 (its published pretrained checkpoints are 3D medical; a 2D instance would be random-init in the session whose purpose is pretraining — R10-adjacent), so the route is **timm-only**; ▲B the canonical smoke is **not** extended to 5 models (it asserts the exact trio, and the suite is at 86 tests / 4:24 against a 5-min cap).* Second Phase-3 session. Read CLAUDE.md §4.6, §8 M5/M9, §5 (UB-16/17), §11, R1–R10, §9 T3.2.

Branch: `phase-3/pretrained-encoders` from current `main` (**`64aa6ce`** — Session 9 IS merged and pushed; the self-draft's "T3.1 was not pushed, FF first" instruction is stale ▲). Linear FF history; no stacking.

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. Ledger: UB-12 `FIXED@72c59f4`; §2 says Phase 3 underway, next T3.2. Root `config.yaml` absent; `codes/config_schema.py` present.
2. `python -c "from codes.config_schema import load_config; load_config('codes/config.yaml')"` → ok; a bogus key raises (spot-check).
3. Full suite green (**~86 tests, ~4:24**) — paste count + wall time. `ruff check codes/` clean.
4. Carry-over facts from S9 (in the self-draft, verified): the pipeline is not bit-reproducible on CPU — use the **deterministic mini-trajectory** technique for any "numbers didn't move" proof, not full-pipeline loss diffs; trainer construction now requires `config.LOSS/OPTIMIZER/SCHEDULER` sub-objects (copy the `test_timing.py` double pattern); `hardware_detector.batch_sizes` is the single authority with registry-parity test-enforced.

**If any check fails:** stop, report; do not build on an unverified base.

## The work (M5) — per-concern commits (R6)

**Defect:** `transunet.py` trains ~100M ViT-B from scratch on ~1.8k images (UB-16); `swin_unet_plus_plus.py` hand-rolls shifted-window attention with no mask and no position bias — the shift is a no-op (UB-17). Do **not** patch hand-rolled attention (§11); adopt library encoders.

**Commit 1 — `swin_pretrained`:** timm **`swinv2_tiny_window8_256`** (native 256 input, window 8, ImageNet-1k) via `timm.create_model(..., pretrained=<flag>, in_chans=1, features_only=True)` + a light UNet-style conv decoder over the 4 feature stages to `(B,10,256,256)`. ▲ Verify availability at session start with `timm.list_models('swinv2*256*', pretrained=True)`; if the tiny variant is missing pick the smallest available 256-input swinv2 and record it. timm's `in_chans=1` adaptation **is** the M5 RGB-kernel summation — do not hand-roll it. Register key `swin_pretrained` (display: "SwinV2-UNet (ImageNet)").

**Commit 2 — `transunet_pretrained`:** two acceptable constructions — state which you shipped and why (M9):
  (a) *minimum:* timm hybrid `vit_base_r50_s16_224` (`img_size=256`, `in_chans=1`, pretrained) + progressive-upsampling decoder from the 1/16 token map (no CNN skips — label honestly);
  (b) *preferred if time allows:* timm `resnet50` `features_only` (pretrained) as CNN encoder + timm ViT-B blocks (pretrained, pos-embed resized) consuming the 1/16 stage via a 1×1 projection, with UNet skips from the ResNet stages — closest to original TransUNet using only library-pretrained parts.
Register key `transunet_pretrained` (display: "TransUNet (ImageNet)"). Either way: **no hand-written attention/transformer blocks.**

**Both commits:** integrate through `UnifiedTrainer` (§4.6), import in `main_pipeline.py` so `@register_model` runs; **do not delete the from-scratch variants** (pretrained-vs-scratch is itself a result); add both new keys to **every** `hardware_detector` tier incl. CPU (size conservatively for the 6 GB baseline — ViT-B-hybrid is the heaviest model in the repo); parity test will force this. `requirements`: add **`timm>=1.0`** only ▲ — no MONAI (heavy, unneeded under the timm route).

**Commit 3 — network gating (the draft's Step 2, sharpened):** constructor flag `pretrained: bool` + env `UBENCH_PRETRAINED=0/1` (default 1 for real runs, forced 0 in tests/smoke). CI and the smoke must never depend on a download. One `@pytest.mark.pretrained` test, skipped unless `UBENCH_ALLOW_DOWNLOADS=1`, does the real load with **proof, not absence-of-error** ▲: (i) the adapted stem weight equals `rgb_w.sum(dim=1, keepdim=True)` computed from the hub checkpoint; (ii) a named deep encoder tensor equals the hub tensor exactly; (iii) log the fraction of encoder params sourced from the checkpoint and the total param count (M9). **Run it once locally with downloads enabled and paste the output (R1)** — if the sandbox has no route to the HF hub, mark `UNVERIFIED: pretrained load` and hand the one-liner to the human; do not fake it.

## Tests (red-first where practical) — ▲ smoke is NOT extended
- `test_models_forward.py`: every registered key incl. the two new ones maps `(2,1,256,256) → (2,10,256,256)` with `pretrained=False` (offline, random init).
- One **single-optimizer-step** test per new key (batch ≤2, CPU, seconds) through `UnifiedTrainer`'s criterion/optimizer path — proves train-loop integration without a subprocess run.
- Stem-sum unit test at the adaptation seam (offline: construct with `pretrained=False`, apply the adaptation function to a synthetic 3-ch weight, assert the sum).
- The canonical trio smoke and its exact set assertion stay **unchanged**; new keys are opt-in via `--models`. No new subprocess pipeline runs (budget, UB-25).

## Docs
Ledger ▲ with honest semantics: **UB-16 → `FIXED@<sha>`** (pretrained TransUNet exists with verified load path; from-scratch retained as an explicitly-scoped baseline — annotate the row). **UB-17 → `FIXED@<sha> (superseded)`**: the fix is supersession by `swin_pretrained`, not repair; the legacy `swin_unet_plus_plus` module gains a docstring warning ("known-defective attention — UB-17; retained only as historical baseline") and §4.6 + README say the same. §2 frontier → next T3.3 (per-family recipes — pretrained transformers will want AdamW+warmup; that is T3.3, do not change recipes here). README model table gains the two keys + the offline/random-weight test policy. Save this prompt (R9).

## Suite budget + push & CI
Measure full-suite wall time (~86 tests + ~6 new unit tests; forward passes of ViT-B-hybrid on CPU are the cost driver — keep batch 2). If it exceeds the 5-min cap, split CI `fast`/`smoke` (`chore(ci):`) and amend §6.4/§7.1 (R9). Push; paste Actions status/URL or `UNVERIFIED: remote CI` + the FF/push one-liner for the human.

## Standing guardrails
Excluded: per-family recipes + selection-by-mIoU (T3.3); thermal preprocessing/augmentation (T3.4 — ImageNet-vs-thermal normalization mismatch is noted, owned by T3.4, do not redesign now); lockfiles/CUDA index (T3.5/UB-20b); import restructure (T3.6). No fabricated pretrained-vs-scratch numbers (R10) — the real comparison happens on the human GPU box; add it to `docs/phase1_realdata_checklist.md` as an optional `--models` superset run. Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.3 next; real-data validation still pending the human GPU box (now including the optional pretrained-vs-scratch run).
8. Ledger hunks (UB-16/17 with the supersession wording) + param-count log lines + the pretrained proof-of-load output (or its `UNVERIFIED` handoff).
9. **Prompt for Session 11 (T3.3):** per-family optimizer/scheduler recipes as config overrides on the T3.1 sections (CNN: current Adam/plateau; transformer family incl. the two new keys: AdamW wd≈0.05 + linear warmup ~5% + cosine, grad-clip 1.0), checkpoint selection by **val mIoU**, both fully disclosed (M4/M9); AC: config toggles recipe per family, transformer logs show warmup, selection metric changed and test-pinned; resume-compat note (optimizer/scheduler state restore vs recipe change). One item; Session Entry Protocol first.

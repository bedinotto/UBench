# Session 11 — T3.3 (UB-18, M4): per-family recipes + selection by val mIoU

*Supersedes the self-drafted `prompts/phase3_session11.md` — the strongest self-draft yet; adopt its structure wholesale (per-family recipe map on the T3.1 config, AdamW/warmup+cosine allow-list, selection by val mIoU, resume-mismatch guard, CNN-unchanged proof via the deterministic mini-trajectory). Five fixes folded in (▲); ▲A prevents a silent LR-schedule corruption the draft would ship.* Read CLAUDE.md §4.2/§4.6, §8 M4/M9, §5 (UB-18), R1–R10, §9 T3.3.

Branch: `phase-3/per-family-recipes` from current `main` (**`7be0bd1`** — the human's FF+push landed after Session 10's report; the draft's "halt if not merged" contingency is satisfied). Linear FF; no stacking.

## Step 0 — SESSION ENTRY PROTOCOL (§10) + first timm-era CI check ▲
1. Ledger: UB-16/17 `FIXED@faa9a2d`; UB-18 OPEN; §2 says next T3.3. Root `config.yaml` absent.
2. `UBENCH_PRETRAINED=0 python -c "import codes.main_pipeline"` registers **5** models.
3. Full suite green (**100 collected: 98 + 2 network-skips, ~4:44**) — paste count + wall time. `ruff check codes/` clean.
4. ▲ **Verify the remote CI run for the Session-10 push** (main@`7be0bd1`) — it is the **first CI ever to install timm and run the 100-test suite on GitHub's 2-vCPU runner**, and Session 10 left it `UNVERIFIED`. Paste its conclusion (gh CLI or ask the human for the URL). **If it failed or timed out on runtime:** the pre-authorized `fast`/`smoke` CI split (`chore(ci):` + §6.4/§7.1 amendment) happens NOW as commit 0, before any T3.3 work. Do not build on a red or unknown CI.
5. Carry-over facts (from the draft, verified): no CPU bit-reproducibility (use the deterministic mini-trajectory for numbers-unchanged proofs); trainer needs `config.LOSS/OPTIMIZER/SCHEDULER`; `UBENCH_PRETRAINED` gates downloads.

## The work — adopt the draft's three commits, with these amendments

**Commit 1 — per-family recipes (draft's design + ▲):**
- Family assignment, schema override map, AdamW + warmup-cosine allow-list, transformer recipe (AdamW wd≈0.05, warmup ~5%, cosine), CNN unchanged — as drafted.
- ▲ **Grad-clip into the recipe schema:** `self.grad_clip_norm` already exists (trainer ~L512/545); expose it per-family (`transformer: 1.0`; CNN keeps its current value) so M4's "grad-clip 1.0" is config-declared, not hardcoded.
- ▲ **Degenerate warmup guard — the smoke WILL exercise this:** on tiny runs (smoke: ~3 batches × 1 epoch) `int(0.05 × total_steps)` = 0. Use `warmup_steps = max(1, int(0.05 * total_steps))` and require `total_steps ≥ 2`; a zero-length `LinearLR`/`SequentialLR` milestone is an edge-crash. Unit-test the arithmetic at total_steps ∈ {2, 3, 1000}.

**Commit 2 — scheduler cadence + selection (draft's commit 2 + ▲A, the important one):**
- ▲A **The stepping contract differs by scheduler kind, and getting it wrong is silent.** Today the trainer calls `self.scheduler.step(val_loss)` once per epoch (~L646) — correct for `ReduceLROnPlateau` only. For warmup+cosine, `step()` is called **per optimizer step, with NO argument**: `_LRScheduler.step()` treats a positional argument as the *epoch number*, so `step(val_loss)` would silently jump the LR to "epoch ≈1.6" of the schedule every epoch — corrupted LR curve, zero errors, green tests. Implement an explicit branch (plateau → per-epoch `step(metric)`; warmup-cosine → per-batch `step()` after `optimizer.step()`/`scaler.update()`, never with a metric) and **pin the contract with a unit test**: a plateau double receives the metric; a cosine double is stepped per-batch and asserts it never receives an argument. Compute `total_steps = len(train_loader) × num_epochs` in the trainer where both are known (draft ✓).
- Selection by val mIoU: ▲ the fields already exist — `best_val_iou` is tracked and checkpointed (~L287/402/454/660); the change is the **save condition** (~L651) from loss-improves to mIoU-improves (keep loss logged). Old checkpoints already load via `.get(...)` defaults, so schema continuity is free — but `test_resume` asserts best-by-loss semantics and must be updated. ▲ `benchmark_models.py` needs **no code change** (it loads `best_*.pth` per fold regardless of what selected them) — docs/report labels only ("best = val mIoU"); don't invent a benchmark refactor.
- Prove the CNN/unet trajectory is bit-identical via the deterministic mini-trajectory (draft ✓) — the diff must show *only* selection + transformer recipe changed.

**Commit 3 — resume mismatch guard:** as drafted; prefer the hard-error on optimizer/scheduler class mismatch (R4) with the message naming both recipes; warn-and-keep-checkpoint-recipe acceptable only if justified. Never load AdamW state into Adam. Record the effective recipe in `run_metadata.json` and the checkpoint.

## Tests (red-first) — draft's list + the cadence-contract test, warmup arithmetic, and updated `test_resume`. No new subprocess runs; smoke trio unchanged (transformers inside it now run AdamW+warmup — structure-asserting, stays green, and the degenerate-warmup guard is what keeps it so).

## Suite budget + push & CI — measure (was 4:44 local; the runner is slower — Step 0.4 may already have split CI). Push; paste Actions conclusion or `UNVERIFIED` + the FF/push one-liner.

## Docs — ledger UB-18 → `FIXED@<sha>` (all three sub-claims now closed: per-family recipes, selection by headline metric, and the held-out test set landed in T2.4 — say so in the row). §4.6 + §8 M4 reconciled; §2 frontier → T3.4 (UB-19/M7). README config table gains the family overrides. Save this prompt (R9).

## Standing guardrails — excluded: thermal preprocessing/augmentation (T3.4), lockfiles/CUDA index (T3.5/UB-20b), import restructure (T3.6), Wilcoxon (T4.1). No fabricated comparisons (R10) — the per-family + pretrained-vs-scratch numbers come from the human GPU box. Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.4 next; real-data GPU-box validation still pending (now covering pretrained + per-family recipes — update `docs/phase1_realdata_checklist.md` accordingly).
8. Ledger hunk (UB-18), the CNN-unchanged deterministic proof, a transformer log excerpt showing AdamW + warmup LR rise, and the cadence-contract test names.
9. Prompt for Session 12 (T3.4, UB-19/M7) as the draft specced — normalization switch (`per_image_minmax | fixed_range [20,40]°C`), physical thermal augmentation (±0.5 °C drift + Gaussian noise; `ShiftScaleRotate`→`A.Affine` within the pinned albumentations <2.0), vectorized raw→°C with the ≥100× timing bound — plus one addition: normalization choice must be recorded in `run_metadata.json` and the report (M9), since it changes what the pretrained stems see.

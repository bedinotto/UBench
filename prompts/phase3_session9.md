# Session 9 — T3.1 (UB-12): one validated config authority (Phase-3 opener)

*This is the first "make the **science** credible" session (Phase 3).* One
ledger item, one focused commit (plus docs). Read CLAUDE.md §4.2 (config truth),
§5 (UB-12), §8, R3, R4, R9, and §9 T3.1 before acting.

Branch: `phase-3/config-consolidation` from `main` **after** main is
fast-forwarded to the Session-8 tip (`git log -1 phase-2/determinism-cleanup` —
currently `e50920f`). Linear FF history. **If main lacks the Session-8 commits,
halt after Step 0 and ask the human to FF. Do not stack.**

Disk pre-flight: green suites self-clean (`tmp_path_retention_policy=failed`);
clear `/tmp/pytest-of-*` before long runs if near capacity.

## Step 0 — SESSION ENTRY PROTOCOL (§10)

1. `test -f codes/tests/test_determinism.py && test -f codes/tests/test_utils.py`.
2. `pytest codes/tests/test_determinism.py codes/tests/test_utils.py -q` → green (~6 tests).
3. Ledger: UB-15/UB-22/UB-24/UB-25 `FIXED`; UB-20 shows **20a FIXED / 20b OPEN**;
   UB-09/10/11 `FIXED`. §2 says "Phase 2 complete".
4. `grep -rn "inference_comparison\|calculate_iou\|train_iou_metric" codes/` → nothing;
   `test -f codes/inference_comparison.py` → absent.
5. Full suite `pytest codes/tests -q` → green (~68 tests); paste count + wall time.

**If any check fails:** stop, report; if Session 8 was never executed, run
`prompts/phase2_session8.md` first. Do not build on an unverified base.

## Step 1 — Commit 1: T3.1 / UB-12 — single validated config

**Defect (ledger/§4.2):** the root `config.yaml` is loaded by nothing (dead);
`codes/config.yaml` is the real one but is read with unchecked `.get(...)`
defaults, so a typo'd key is silently ignored. `training.batch_sizes` is
unconsumed; the `CLASS_WEIGHTS` branch in `unified_training` has no producer;
scheduler/optimizer are hardcoded.

**Fix (§9 T3.1, R4):**
- **Delete the root `config.yaml`** (dead — verified by §4.2). Do **not**
  reintroduce a second config file (anti-pattern §11).
- Put a **schema** over `codes/config.yaml` — pydantic or pydantic-settings —
  that **fails on unknown keys** (`extra="forbid"`) and validates types. `Config`
  loads through it. A bogus key must raise at startup, not be ignored.
- **⚠ Do not silently change current numeric behavior.** Every key that was
  hardcoded (scheduler patience/factor, optimizer type/betas, augmentation
  probabilities, etc.) must keep its **current effective value** as the schema
  default unless a change is deliberate and called out. Diff the resulting
  effective config against today's behavior before/after — the smoke's loss on
  a fixed seed should be unchanged.
- For each currently-dead key: **either wire it** (scheduler/optimizer read from
  config; augmentation params read from config; `class_weights` computed from
  **train-fold class frequencies** and passed to `CombinedLoss`) **or delete it**.
  No key may remain that does nothing (that is the UB-12 failure mode). Prefer
  wiring for scheduler/optimizer/augmentation; `training.batch_sizes` is either
  wired as an override of the hardware profile or deleted.

**Tests (`codes/tests/test_config.py`, red-first):** a bogus/unknown key raises;
a wrong-typed value raises; each remaining key demonstrably changes behavior
(e.g. setting `scheduler.patience` changes the scheduler state; a `class_weights`
toggle changes the loss); the shipped `codes/config.yaml` validates. Keep it
unit-level; the smoke covers end-to-end.

Verify: full suite + ruff green; **smoke green with unchanged fixed-seed loss**
(prove T3.1 did not move the numbers).

Commit: `refactor(config): single pydantic-validated config; delete dead root yaml; wire or drop every key (UB-12, T3.1)`

## Step 2 — Docs
Ledger UB-12 → `FIXED@<sha>`. §4.2 rewritten to the single validated authority
(no dead root file; unknown keys raise; list which keys are wired). §2 frontier →
Phase 3 underway (next: T3.2 pretrained encoders). README config section
reconciled to `codes/config.yaml` (the root file is gone). Save this prompt file.

## Step 3 — Suite budget + push & CI
Measure full-suite wall time (was 4:31 at Session 8). If it exceeds the §7.1
five-minute cap, split CI into `fast`/`smoke` (`chore(ci):`) and amend §6.4/§7.1.
Push; paste Actions status/URL or `UNVERIFIED: remote CI` + human instruction.

## Standing guardrails
Excluded temptations: pretrained encoders (T3.2), per-family recipes (T3.3 — but
T3.1 *exposes* optimizer/scheduler as config; do not yet change the recipes),
thermal preprocessing/augmentation redesign (T3.4), lockfiles (T3.5), imports
(T3.6). Tool-generated edits (graphify) → separate `chore(tooling):`.
R1/R3/R4/R9/R10 in force. Blocked or contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. **Frontier:** Phase 3 progress (T3.2 pretrained encoders next) + real-data
   validation status (still pending the human GPU box).
8. Ledger diff hunks (UB-12 flip; §2/§4.2 updates).
9. **Prompt for Session 10** covering **T3.2 (UB-16/17, M5)** — pretrained
   encoders via timm/MONAI (TransUNet R50+ViT-B/16; Swin via SwinUNETR), 1-channel
   stem adaptation by summing pretrained RGB kernels, register `transunet_pretrained`
   / `swinunetr`. AC: forward-shape tests; param counts logged; smoke covers the
   new keys. One item; Session Entry Protocol first.

# Session 13 — T3.5 (UB-20b): lockfiles + reproducible install

*Fifth Phase-3 session. One ledger item (UB-20b — the remaining half of UB-20; 20a landed in T2.5). Closes the dependency/reproducibility gap: pinned lockfiles, a `setup.py` that creates a venv and resolves the current CUDA index without `--force-reinstall --no-deps`, CI installing from the lockfile, and the lockfile hash recorded in `run_metadata.json` (the T2.5 TODO). Read CLAUDE.md §3.1, §5 (UB-20), §8 M6, §11, R6–R9, §9 T3.5.*

Branch: `phase-3/lockfiles` from current `main` (**`67d3733`** — Session 12 merged + pushed via gh HTTPS; CI ran green). Linear FF; no stacking. Push: `git -c credential.helper='!gh auth git-credential' push https://github.com/bedinotto/UBench.git main` (SSH remote does not work in the sandbox; gh token has repo scope).

## Step 0 — SESSION ENTRY PROTOCOL (§10)
1. Ledger: UB-19 `FIXED@5f8e15b`, UB-26 `FIXED@dc3fe29`; UB-20 shows **20a FIXED / 20b OPEN**; §2 says next T3.5. Root `config.yaml` absent.
2. `UBENCH_PRETRAINED=0 python -c "import codes.main_pipeline"` registers 5 models; config loads.
3. Full suite green (**~141 collected: ~139 pass + 2 network-skip, ~4:50**) — paste count + wall time. `ruff check codes/` clean.
4. Verify CI for main@`67d3733` (Session-12 push) is green (`gh run list --branch main`); if red, stop and report.
5. Carry-over: `timm`, `pydantic`, `albumentations<2.0` are deps; PyPI + HF hub reachable from the sandbox, GitHub SSH is not (use gh HTTPS). The pipeline is not bit-reproducible on CPU.

## The defect (UB-20b, verified)
`codes/setup.py` installs torch with `--force-reinstall --no-deps` against a hardcoded aging `cu121` index; deps are unpinned `>=` in `requirements/*.txt` (no lockfile), so a fresh clone resolves *whatever* is current — non-reproducible, and `--no-deps` can leave torch's own deps unmet. No CPU vs CUDA lock split. `run_metadata.json` has `lockfile_hash: None` (the T2.5 placeholder).

## The work (M6) — per-concern commits (R6)
**Decision to make explicit first ▲ (inherited from T3.4):** the `albumentations<2.0` pin + the `A.Affine` migration set up the 2.x unpin. **Decide it now, in the lockfile:** either (a) stay pinned `<2.0` and lock that, documenting why (the 2.x API churn is out of scope), or (b) migrate-and-unpin (verify the `A.Affine`/`ImageOnlyTransform` API on 2.x, re-run `test_augmentation`). **Recommend (a)** — the lockfile pins it deterministically and the unpin is its own future task; do not let it ride silently.

**Commit 1 — lockfiles (`uv`):**
- Adopt `uv` (fast, standard). Produce **two locks**: a **CPU lock** (CI/dev — the CPU torch index) and a **CUDA lock** (training box — the CUDA index). Generate from the existing `requirements/*.txt` inputs + `pydantic`/`timm`/`albumentations<2.0`. Commit both lockfiles.
- ▲ Do not invent versions (R10) — generate the locks with `uv` against the real indexes and commit exactly what it resolves. If the sandbox cannot reach an index for one of them (e.g. the CUDA wheel index), generate the CPU lock (which the sandbox + CI use) and mark the CUDA lock `UNVERIFIED: generate on a CUDA box` with the exact command, rather than hand-writing it.

**Commit 2 — `setup.py` reproducible install:**
- Create a venv (§3.1 doctrine — never system Python); install from the **lockfile**, not loose `>=`. Drop `--force-reinstall --no-deps`. Resolve the CUDA index from the actual current stable (parameterize cu121/cu118 or detect), no hardcoded aging index. Keep it CPU-first (the CPU lock is the default; CUDA lock is the training-box path).
- AC: a clean-container (or fresh-venv) install **from the CPU lockfile** → smoke green. Paste it.

**Commit 3 — lockfile hash in run metadata:**
- `collect_run_metadata` fills `lockfile_hash` (sha256 of the active lockfile) instead of `None` — closing the T2.5 TODO (M6/M9). Which lock (CPU/CUDA) is recorded should reflect what's installed; a simple, documented heuristic (e.g. hash the CPU lock unless CUDA torch is active) is fine. Test: the field is a hex digest, not None.

## Tests / CI / budget
- CI (`.github/workflows/ci.yml`): install from the **CPU lockfile** (`uv sync`/`uv pip sync`) instead of the current loose `pip install`. This is the real proof the lock works — paste the green run.
- Determinism/metadata test updated for `lockfile_hash`. No new subprocess runs; smoke unchanged. Measure wall time; the CI-cap guidance stands (local ~4:50; split only if broken).

## Docs
Ledger UB-20b → `FIXED@<sha>` (UB-20 fully closed). §3.1 rewritten to the lockfile install (CPU lock for CI/dev, CUDA lock for training; `uv`; no `--no-deps`). §2 frontier → T3.6 (UB-21 imports) or T3.7 (README reconciliation) — pick per §9 ordering. §7.4 CI snippet updated to `uv`. Record the albumentations pin decision. Save this prompt (R9).

## Standing guardrails
Excluded: import restructure (T3.6), README full reconciliation (T3.7), Wilcoxon (T4.1), the albumentations 2.x *migration* (only the pin *decision* is in scope). No fabricated version numbers (R10) — locks are generated, not written. Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.6/T3.7 next; real-data GPU-box validation pending — now also needs the CUDA lockfile verified on the box.
8. Ledger hunk (UB-20b), the clean-install-from-lock → smoke-green transcript, the CI-installs-from-lock green run, and the albumentations pin decision.
9. Prompt for Session 14 (T3.6, UB-21): absolute imports rooted at `codes.`, add `__init__.py` where missing, models runnable as `python -m codes.<model>` self-test (forward-shape print), remove `sys.path` hacks and the try/except dual-import fallbacks now that the package is importable. AC: `python -m codes.transunet` prints a forward shape with no ImportError; ruff clean; smoke green. One item; Session Entry Protocol first.

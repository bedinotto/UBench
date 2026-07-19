# Session 13 — T3.5 (UB-20b): lockfiles + reproducible install

*Supersedes the self-drafted `prompts/phase3_session13.md` — adopt its three-commit shape and the "generate, never hand-write versions" rule (R10) wholesale. Six fixes folded in (▲); ▲A settles a design fork the draft leaves implicit, and ▲B corrects a lockfile mechanic that would otherwise produce a lock CI can't install from.* Closes UB-20 (20a landed in T2.5). Read CLAUDE.md §3.1, §5 (UB-20), §8 M6, §11, R6–R10, §9 T3.5.

Branch: `phase-3/lockfiles` from current `main` (**`92ebaa8`** ▲ — the draft's `67d3733` is stale; the Session-13-prompt commit moved the tip). Linear FF; no stacking. Push: gh-credentialed HTTPS (the draft's one-liner), not SSH.

Sandbox facts (verified this session, use them): **`uv 0.11.7` is installed**; PyPI reachable; the PyTorch CPU wheel index is reachable; the CUDA wheel index and GitHub SSH are **not**. So the CPU lock can be generated and proven here end-to-end; the CUDA lock cannot (▲C).

## Step 0 — SESSION ENTRY PROTOCOL (§10)
Adopt the draft's five checks (ledger: UB-19 `FIXED@5f8e15b`, UB-26 `FIXED@dc3fe29`, UB-20 = 20a FIXED/20b OPEN; 5 models register; config loads; full suite ~141 @ ~4:50 pasted; `gh run list --branch main` green for `92ebaa8`; ruff clean).

## ▲A — Decide the dependency source of truth FIRST (this shapes every commit)
`pyproject.toml` currently holds **only** ruff + pytest config — **no `[project]` table, no dependencies**. The deps live in `requirements/requirements.txt` (loose `>=`), and torch is installed separately by `setup.py`. Two coherent ways to lock, pick one and state why:
- **(a) uv-native (recommended):** add a minimal `[project]` table to `pyproject.toml` declaring the runtime deps (mirror `requirements.txt` exactly, same `>=` floors + the `albumentations<2.0` ceiling), then `uv lock` → `uv.lock`. `requirements.txt` becomes a generated/derived artifact or a pointer, so there is **one** dependency source (avoids the requirements-vs-pyproject dual-authority — the UB-02/UB-05 bug class). Torch's CPU/CUDA index split is expressed via uv index configuration, not a second requirements file.
- **(b) requirements-in/-out:** keep `requirements.txt` as the input and `uv pip compile` it to a pinned `requirements.lock`. Lighter diff, but leaves `pyproject.toml` non-authoritative for deps and needs a separate mechanism for the torch index.
Recommend (a): it makes `pyproject.toml` the single authority CLAUDE.md §3.1 can point to. Whichever you pick, **there must be exactly one place that declares a dependency** by the end (R5).

## ▲B/▲C — The torch dual-index reality (the draft under-specifies this)
torch is the whole reason UB-20 exists, so the lock must cover it — but the CPU and CUDA wheels come from **different indexes** (`download.pytorch.org/whl/cpu` vs `.../cu121`). A single lock cannot pin both; that is why the draft's "two locks" is right, but the mechanism matters:
- **CPU lock** (CI + dev + sandbox): resolve torch/torchvision from the **CPU index**. ▲ Generate it **here** with `uv` against the real CPU index and commit exactly what resolves — then prove it (Commit 2 AC). This is the lock CI installs from, so it must be real, not `UNVERIFIED`.
- **CUDA lock** (training box): resolve from the CUDA index. ▲C The sandbox **cannot reach the CUDA index**, so do **not** fabricate it (R10). Generate the CPU lock here; emit the exact `uv` command + pinned inputs for the CUDA lock and mark it `UNVERIFIED: run on a CUDA box`, to be produced during the GPU-box validation run. Record this in the checklist, not as a committed guessed lock.
- ▲ **Pin the Python version in the lock inputs.** `uv` resolves per interpreter; CI is 3.11, the sandbox is 3.12, real GPU boxes are 3.10/3.11 (§3.1). Set `requires-python = ">=3.10,<3.14"` (or the range §3.1 endorses) so the lock is valid across them, and note that torch CUDA wheels currently gate the GPU box to 3.10/3.11 — the lock must not silently assume 3.12/3.13 for the CUDA path.

## ▲ Resolve the albumentations pin decision (inherited from T3.4) — explicitly, in the lock
The draft's recommendation stands: **stay `<2.0` and lock it deterministically**; the 2.x `A.Affine`/`ImageOnlyTransform` migration is its own future task, out of scope. State it in the commit body and the ledger note so it does not ride silently.

## The work — per-concern commits (R6)
**Commit 1 — dependency source + CPU lock:** implement ▲A's chosen design; generate and commit the **CPU lock** (proven-here); add the CUDA-lock command as an `UNVERIFIED` doc/checklist entry per ▲C. Do not invent versions.
**Commit 2 — `setup.py` reproducible install:** create a venv (§3.1 — never system Python); install from the **lockfile** (`uv sync`/`uv pip sync`), not loose `>=`; **drop `--force-reinstall --no-deps`**; parameterize the CUDA index (detect cu121/cu118 from the driver as today, but no hardcoded *aging* default — resolve the current stable tag) with CPU as the default path. ▲ **Two caveats:** (1) `setup.py` is invoked by `run.sh` inside an already-active environment — creating a *nested* venv from within it can orphan the running interpreter; make venv-creation opt-in / idempotent (skip if already in a venv, or gate behind a flag) rather than unconditional, and keep `--skip-setup` working. (2) `run.bat` has the same install logic — mirror the change or explicitly defer Windows to a ledger note (`UNVERIFIED: Windows`, consistent with prior sessions). **AC: fresh-venv install from the CPU lock → smoke green, pasted.**
**Commit 3 — lockfile hash in run metadata:** `collect_run_metadata` fills `lockfile_hash` (sha256 of the active lock) instead of `None`, closing the T2.5 TODO (M6/M9); documented heuristic for which lock is active (hash the CPU lock unless CUDA torch is imported). Test: the field is a hex digest, not None.

## Tests / CI / budget
CI (`.github/workflows/ci.yml`) currently does `pip install torch --index-url .../cpu` then `pip install -r requirements/requirements.txt`. ▲ Replace **both** steps with a single lock-driven install (`uv sync --locked` or `uv pip sync <cpu-lock>`) — the `--locked`/frozen flag is the actual proof the lock is complete and current (it fails if the lock is stale vs the inputs). Paste the green run. Determinism/metadata test updated for `lockfile_hash`. No new subprocess runs; smoke unchanged. Measure wall time; split CI only if the 5-min cap breaks.

## Docs
Ledger UB-20b → `FIXED@<sha>` (UB-20 fully closed; note the CUDA lock is `UNVERIFIED` pending the GPU box, and the albumentations `<2.0` pin decision). §3.1 rewritten to the lock install (source of truth per ▲A; CPU lock for CI/dev, CUDA lock for training; `uv`; no `--no-deps`; venv caveat). §2 frontier → T3.6 (UB-21 imports). §7.4 CI snippet → `uv`. **`docs/phase1_realdata_checklist.md`:** add "generate + commit the CUDA lock on the box" and "install from the CUDA lock" as steps. Save this prompt (R9).

## Standing guardrails
Excluded: import restructure (T3.6), full README reconciliation (T3.7), Wilcoxon (T4.1), the albumentations 2.x *migration* (only the pin decision is in scope). No fabricated versions — locks are generated (R10); the CUDA lock is `UNVERIFIED`, not guessed. Tool edits → `chore(tooling):`. Blocked/contradicted → stop, report, propose, wait.

## End-of-session report — same format, plus:
7. Frontier: T3.6 next (imports); real-data GPU-box validation pending — now also needs the CUDA lock generated + verified on the box.
8. Ledger hunk (UB-20b), the fresh-venv-from-CPU-lock → smoke-green transcript, the CI-installs-from-lock (`--locked`) green run, the `pyproject.toml`/single-source decision, and the albumentations pin decision.
9. Prompt for Session 14 (T3.6, UB-21) as the draft specced — absolute imports rooted at `codes.`, `__init__.py` where missing, `python -m codes.<model>` self-test, remove `sys.path` hacks and try/except dual-import fallbacks; AC: `python -m codes.transunet` prints a forward shape, no ImportError, ruff clean, smoke green — plus one addition ▲: audit for and remove any import-time side effects that only worked under the old loose-import scheme (e.g. registration relying on import order), so the package-clean imports don't silently drop a model registration. One item; Session Entry Protocol first.

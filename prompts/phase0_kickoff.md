# Session 0 — Build the safety net (Phase 0 of CLAUDE.md)

You are starting the UBench remediation program. Before anything else, read `CLAUDE.md` at the repository root **in full**. It is the authoritative operating guide: §5 is the verified defect ledger (ground truth — do not re-audit it), §6 are binding rules, §7 specifies exactly what you will build this session, §9 defines the plan and acceptance criteria. Work on a branch named `phase-0/safety-net`; everything in this session is one PR.

## Scope lock — Phase 0 only (T0.1 + T0.2)

Build the testing safety net. Do **not** fix any UB-xx ledger defect in this session, no matter how trivial — not the one-line ones (UB-05, UB-13 will tempt you), not "while you're there" improvements (rules R3, R4). The entire point of Phase 0 is that every later fix lands against a net that already exists. The only repository changes allowed are the ones T0.1 and T0.2 name.

## Step 0 — Environment & reconnaissance

1. Create a venv and install CPU-only torch, the project requirements, pytest, and ruff (CLAUDE.md §3.1). Verify and paste: `python -V` and `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
2. Read these files before writing any test, so the fixture is structurally faithful: `codes/unified_data.py` (Config path resolution, `REGION_NAMES`, `get_tiff_path`, `create_single_fold_loader`, `LIMIT_SAMPLES` handling), `codes/preprocess_data.py`, `codes/main_pipeline.py` (CLI flags + env vars), `codes/config.yaml`.
3. List whatever already exists in `codes/tests/` (gitignored ad-hoc debug scripts may live there). Do not delete or modify anything found; your new files must coexist. Deletion decisions belong to T2.6.
4. Produce a short todo list mapping your actions to the T0.1/T0.2 acceptance criteria, then execute it without waiting.

## Step 1 — T0.1: synthetic fixture + smoke test

Implement `codes/tests/conftest.py` and `codes/tests/test_pipeline_smoke.py` exactly per CLAUDE.md §7.2–§7.3. Operational details that matter:

- **Subprocess invocation:** `sys.executable <repo>/codes/main_pipeline.py --models unet transunet swin`, with `cwd=<fixture dataset root>` — `Config` resolves `data/`, `outputs/`, `logs/` against CWD while `codes/config.yaml` is found via `__file__`, so CWD isolation is safe. Set env `LIMIT_SAMPLES`, `NUM_EPOCHS=1`, `K_FOLDS=2`, `NUM_WORKERS=0`, and pass a hard `timeout=` so CI can never hang. Capture stdout+stderr and include their tails in assertion messages.
- **Group-count trap:** `LIMIT_SAMPLES` truncates with `df.head(n)`. After truncation, the number of distinct subjects must remain ≥ `K_FOLDS`, or GroupKFold fails for the *wrong* reason. With 5 subjects × 4 frames, `LIMIT_SAMPLES=20` keeps all 5; if you shrink for speed, verify the surviving group count first.
- **CPU runtime:** three models at 256×256 (TransUNet is ViT-B-sized) — keep sample counts minimal and report the measured wall time; the test must finish comfortably inside the CI timeout.
- **Expected-failure handling:** today the smoke test must fail with UB-01's `FileNotFoundError: Preprocessed data not found at ... metadata.csv`. Mark it `@pytest.mark.xfail(reason="UB-01: preprocessing not wired into pipeline", strict=True)` so the suite is green while the failure stays enforced — strict xfail will force removal of the marker the moment T1.1 fixes it. CLAUDE.md §9 T0.2's AC currently reads "red only on the smoke test"; update that AC to this strict-xfail formulation **in the same PR**, citing R9 (living document).
- **Verification (R1):** run `pytest codes/tests -x -q -rx` and paste the output. If the smoke test fails for ANY reason other than the UB-01 error — fixture bug, import error, region-name mismatch, group-count crash — fix the fixture/test until the failure is exactly the expected one. *Getting the test to fail for the right reason is the deliverable.* (If dependency drift surfaces here, e.g. albumentations 2.x API changes, record it under UB-20 in the ledger notes and work around it in the test env only — do not patch pipeline code.)

## Step 2 — T0.2: repo plumbing + CI

- `.gitignore`: remove the `codes/tests/*` and `CLAUDE.md` lines. `git add` CLAUDE.md and the new tests; then prove with `git status --porcelain` that nothing from `data/`, `outputs/`, `logs/`, or `*.pth` is staged.
- Add empty `codes/__init__.py` and `codes/tests/__init__.py`. Do not convert existing modules' import styles — that is T3.6.
- `pyproject.toml`: configure ruff with a minimal critical ruleset (`E9`, `F63`, `F7`, `F82`) and `target-version = "py310"` so the *existing* code passes without reformatting — tightening the ruleset is later work, and a mass-reformat would violate R3. Verify `ruff check codes/` exits 0 and paste the output.
- `.github/workflows/ci.yml` per §7.4 (CPU torch, ruff, pytest — green with the strict-xfail smoke test).

## End-of-session report (required format)

1. AC checklist for T0.1 and T0.2, each item with pasted command output as evidence.
2. `git log --oneline` of your commits — conventional commits referencing tasks, e.g. `test(smoke): add synthetic dataset fixture and E2E smoke gate (T0.1)` and `chore(ci): commit tests, add ruff config and GitHub Actions workflow (T0.2)`.
3. List of files created/modified.
4. Ledger confirmation: every UB row still `OPEN` (nothing fixed this session); note the §9 doc-AC update you made.
5. Anything you discovered that contradicts CLAUDE.md, with either the doc fix you made (R9) or an `UNVERIFIED:` note.
6. The exact command/prompt to begin the next session (Phase 1, T1.1).

Rules in force for the whole session: R1 (no claim without an executed command and its output), R3 (surgical diffs only), R4 (never silence an error to make progress), R10 (smoke artifacts are labeled as smoke, never as results). If anything blocks you — a missing dependency, unexpected repository state, an assumption in CLAUDE.md that proves false — stop, report what you found, propose options, and wait rather than improvising around it.

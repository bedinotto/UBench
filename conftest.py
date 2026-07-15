# Root-level conftest: exclude untracked ad-hoc debug scripts that would fail
# collection on a CPU-only install (module-level CUDA code, no pytest functions).
# These files are not committed to git and won't exist in CI.
# Deletion is deferred to T2.6.
collect_ignore_glob = [
    "*test_unet_nan*.py",
    "*/test_suite.py",
]

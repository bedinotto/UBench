"""
Console Logger (Tee)
====================
Redirects sys.stdout and sys.stderr so that every print() call and every
exception traceback is written to BOTH the console AND a log file at the
same time.

Usage
-----
    from codes.logger import TeeLogger

    logger = TeeLogger("logs/2026-04-17_10-00-00/pipeline.log")
    logger.start()          # all prints now go to console + file
    ...                     # run your code
    logger.stop()           # restore original stdout/stderr
    # or use as a context manager:
    with TeeLogger("logs/.../pipeline.log"):
        ...
"""

import sys
import io
import os
from datetime import datetime
from pathlib import Path


class _TeeStream:
    """
    A file-like object that writes to two streams simultaneously.
    One stream is the original console handle; the other is the log file.
    Each line written to the file gets a timestamp prefix.
    """

    def __init__(self, console_stream, log_file: io.TextIOWrapper, add_timestamps: bool = True):
        self._console = console_stream
        self._log = log_file
        self._add_timestamps = add_timestamps
        self._line_buf = ""  # accumulate partial lines for timestamping

    # -----------------------------------------------------------------
    # Core write / flush
    # -----------------------------------------------------------------

    def write(self, text: str) -> int:
        # Always mirror to console as-is (preserves colours, progress bars …)
        self._console.write(text)

        if self._add_timestamps:
            self._write_timestamped(text)
        else:
            self._log.write(text)

        return len(text)

    def _write_timestamped(self, text: str):
        """Prefix every complete line with a timestamp."""
        self._line_buf += text
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            ts = datetime.now().strftime("%H:%M:%S")
            self._log.write(f"[{ts}] {line}\n")
        # Flush any partial line without a timestamp so nothing is lost
        if self._line_buf:
            self._log.write(self._line_buf)
            self._line_buf = ""

    def flush(self):
        self._console.flush()
        self._log.flush()

    # -----------------------------------------------------------------
    # Proxy all other attribute accesses to the console stream so that
    # libraries that inspect sys.stdout (e.g. tqdm) work correctly.
    # -----------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._console, name)

    # fileno() must raise so that subprocess / os-level code falls back
    # to the console fd instead of our wrapper.
    def fileno(self):
        return self._console.fileno()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TeeLogger:
    """
    Intercepts sys.stdout and sys.stderr and mirrors them to a log file.

    Parameters
    ----------
    log_path : str | Path
        Destination log file.  Parent directories are created automatically.
    add_timestamps : bool
        If True (default), prefix every line written to the file with [HH:MM:SS].
    """

    def __init__(self, log_path, add_timestamps: bool = True):
        self.log_path = Path(log_path)
        self.add_timestamps = add_timestamps
        self._log_file = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._active = False

    # -----------------------------------------------------------------
    # start / stop
    # -----------------------------------------------------------------

    def start(self):
        if self._active:
            return  # already running — do not double-hook

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open in append mode so that multiple processes writing to the
        # same file (e.g. setup + pipeline) don't clobber each other.
        self._log_file = open(self.log_path, "a", encoding="utf-8", errors="replace")

        # Write a session header
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        self._log_file.write(f"\n{sep}\n")
        self._log_file.write(f"SESSION START  {ts}\n")
        self._log_file.write(f"{sep}\n")
        self._log_file.flush()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        sys.stdout = _TeeStream(self._orig_stdout, self._log_file, self.add_timestamps)
        sys.stderr = _TeeStream(self._orig_stderr, self._log_file, self.add_timestamps)

        self._active = True

    def stop(self):
        if not self._active:
            return

        # Flush any buffered content
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore originals
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        # Write a session footer
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        self._log_file.write(f"\n{sep}\n")
        self._log_file.write(f"SESSION END    {ts}\n")
        self._log_file.write(f"{sep}\n\n")
        self._log_file.flush()
        self._log_file.close()
        self._log_file = None

        self._active = False

    # -----------------------------------------------------------------
    # Context-manager support
    # -----------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Convenience helper used by child scripts (setup.py, extract_data.py)
# ---------------------------------------------------------------------------

def start_from_env(default_filename: str = "run.log") -> "TeeLogger | None":
    """
    Start a TeeLogger by reading the log directory from the
    ``UBENCH_LOG_DIR`` environment variable (set by main_pipeline.py
    before it spawns child processes / imports sub-modules).

    Returns the active TeeLogger, or None if the env var is not set.
    """
    log_dir = os.environ.get("UBENCH_LOG_DIR")
    if not log_dir:
        return None

    log_path = Path(log_dir) / default_filename
    logger = TeeLogger(log_path)
    logger.start()
    return logger

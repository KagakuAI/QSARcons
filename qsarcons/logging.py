import sys
import os
import logging
import threading

class DisableLogger:
    """Context‑manager that suppresses *all* logging inside its scope (thread-safe)."""

    _lock = threading.Lock()
    _active_count = 0  # counts nested/parallel suppressions

    def __enter__(self):
        with self._lock:
            if DisableLogger._active_count == 0:
                logging.disable(logging.CRITICAL)
            DisableLogger._active_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            DisableLogger._active_count -= 1
            if DisableLogger._active_count == 0:
                logging.disable(logging.NOTSET)

class HiddenPrints:
    """Context‑manager that suppresses *print* output inside its scope (thread-safe)."""

    _lock = threading.Lock()
    _active_count = 0
    _orig_stdout = sys.stdout

    def __enter__(self):
        with HiddenPrints._lock:
            if HiddenPrints._active_count == 0:
                HiddenPrints._orig_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
            HiddenPrints._active_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        with HiddenPrints._lock:
            HiddenPrints._active_count -= 1
            if HiddenPrints._active_count == 0:
                sys.stdout.close()
                sys.stdout = HiddenPrints._orig_stdout

class OutputSuppressor:
    """
    Completely suppress ALL output:
    - Python prints
    - logging
    - C/C++ libraries writing to stdout/stderr (CatBoost, XGBoost, etc.)
    Thread-safe and nestable.
    """

    _lock = threading.Lock()
    _active = 0

    def __enter__(self):
        with OutputSuppressor._lock:
            if OutputSuppressor._active == 0:
                # Save original file descriptors
                self._orig_stdout_fd = os.dup(1)
                self._orig_stderr_fd = os.dup(2)

                # Open null file
                self._devnull = os.open(os.devnull, os.O_WRONLY)

                # Redirect Python-level stdio
                self._orig_stdout = sys.stdout
                self._orig_stderr = sys.stderr
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

                # Redirect C-level stdout/stderr
                os.dup2(self._devnull, 1)
                os.dup2(self._devnull, 2)

                # Disable logging
                logging.disable(logging.CRITICAL)

            OutputSuppressor._active += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        with OutputSuppressor._lock:
            OutputSuppressor._active -= 1
            if OutputSuppressor._active == 0:
                # Restore file descriptors
                os.dup2(self._orig_stdout_fd, 1)
                os.dup2(self._orig_stderr_fd, 2)

                # Close temp files
                os.close(self._devnull)
                os.close(self._orig_stdout_fd)
                os.close(self._orig_stderr_fd)

                # Restore Python-level stdio
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = self._orig_stdout
                sys.stderr = self._orig_stderr

                # Re-enable logging
                logging.disable(logging.NOTSET)

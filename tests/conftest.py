# conftest.py
import os
import sys


def pytest_xdist_auto_num_workers(config):
    """Cap xdist worker count so notebook tests don't oversubscribe the CPU."""
    # Get available CPU count on the system
    cpus = os.cpu_count() or 1

    # Cap worker count (e.g., maximum 4 workers for notebooks)
    MAX_NOTEBOOK_WORKERS = 4

    # Reserve 1 core for the system, ensuring at least 1 worker runs
    available_cpus = max(1, cpus - 1)

    core_selection = min(available_cpus, MAX_NOTEBOOK_WORKERS)
    print(f"Running pytest with {core_selection} core(s)")

    return core_selection


class WarningFilterStream:
    """Stream wrapper that drops one noisy IPKernelApp warning."""

    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, text):
        """Write text, dropping the specific IPKernelApp TCP encryption warning."""
        if "Kernel is running over TCP without encryption" in text:
            return
        self.original_stream.write(text)

    def flush(self):
        """Flush the wrapped stream."""
        self.original_stream.flush()


# Redirect sys.stderr before tests run
sys.stderr = WarningFilterStream(sys.stderr)

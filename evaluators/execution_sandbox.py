"""OS-level secure execution sandbox for untrusted model-generated Python code.

Uses Bubblewrap (`bwrap`) to provide genuine OS-level container isolation:
- Unprivileged user, PID, network, IPC, UTS, and cgroup namespaces (`--unshare-all`).
- Network namespace is isolated with no default route (`--unshare-net`).
- Host filesystem is mounted read-only (`--ro-bind /usr /usr`, etc.).
- Host `/home`, `/root`, `/var`, `/run`, and Docker sockets are NOT mounted.
- Dedicated, ephemeral writable directory for each task (`/workspace`).
- Host environment is wiped (`--clearenv`), passing only safe defaults.
- All Linux capabilities dropped (`--cap-drop ALL`).
- Die with parent (`--die-with-parent`).
- In-container resource limits applied: RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE.
- Subprocess wall-clock timeout with clean process termination and workspace cleanup.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ExecutionResult:
    passed: bool
    error_type: Optional[str]  # None, "syntax_error", "assertion_error", "timeout", "memory_limit", "runtime_error", "security_violation"
    error_message: str
    stdout: str
    stderr: str
    duration_sec: float
    isolation_engine: str = "bwrap"


def is_bwrap_available() -> bool:
    """Check if bubblewrap binary is available on the system."""
    return shutil.which("bwrap") is not None


def execute_sandboxed(
    candidate_code: str,
    test_list: Sequence[str],
    test_setup_code: str = "",
    timeout_sec: float = 3.0,
    max_memory_mb: int = 512,
    python_executable: str = "/usr/bin/python",
) -> ExecutionResult:
    """Execute candidate function with tests inside a bubblewrap OS container.

    Args:
        candidate_code: Python code produced by the model.
        test_list: List of assertion statements, e.g. ["assert f(1) == 2"].
        test_setup_code: Optional setup code (imports, helper constants).
        timeout_sec: Wall-clock timeout in seconds.
        max_memory_mb: Virtual memory ceiling in MB.
        python_executable: Path to python binary inside container.
    """
    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        return ExecutionResult(
            passed=False,
            error_type="security_violation",
            error_message="Hard security gate: bubblewrap (bwrap) not installed; refusing un-isolated execution.",
            stdout="",
            stderr="bwrap runtime missing",
            duration_sec=0.0,
            isolation_engine="none",
        )

    test_assertions = "\n".join(test_list)
    cpu_limit = max(1, int(timeout_sec))

    # Build harness script setting in-container rlimits before candidate execution
    script_content = f"""# Isolated evaluation harness
import math
import os
import resource
import sys
from collections import *
from itertools import *

try:
    max_mem_bytes = {max_memory_mb} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit} + 1))
    if hasattr(resource, 'RLIMIT_FSIZE'):
        resource.setrlimit(resource.RLIMIT_FSIZE, (5 * 1024 * 1024, 5 * 1024 * 1024))
except Exception:
    pass

{candidate_code}

{test_setup_code}

# Test assertions
{test_assertions}
"""

    with tempfile.TemporaryDirectory(prefix="mbpp_sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "solution.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Bubblewrap command arguments
        bwrap_cmd = [
            bwrap_path,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop", "ALL",
            # Read-only system mounts
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--ro-bind-try", "/etc/ld.so.conf", "/etc/ld.so.conf",
            "--ro-bind-try", "/etc/ld.so.conf.d", "/etc/ld.so.conf.d",
            # Virtual filesystems
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            # Writable dedicated workspace
            "--bind", tmpdir, "/workspace",
            "--chdir", "/workspace",
            # Completely wiped environment
            "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PYTHONNOUSERSITE", "1",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            # Command to run
            python_executable, "-I", "-B", "/workspace/solution.py"
        ]

        start_time = time.perf_counter()
        try:
            res = subprocess.run(
                bwrap_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            duration = time.perf_counter() - start_time

            if res.returncode == 0:
                return ExecutionResult(
                    passed=True,
                    error_type=None,
                    error_message="",
                    stdout=res.stdout,
                    stderr=res.stderr,
                    duration_sec=duration,
                    isolation_engine="bwrap",
                )

            stderr = res.stderr
            if "AssertionError" in stderr:
                err_type = "assertion_error"
            elif "SyntaxError" in stderr:
                err_type = "syntax_error"
            elif "MemoryError" in stderr or res.returncode in (-9, 137):
                err_type = "memory_limit"
            elif (
                "PermissionError" in stderr
                or "Read-only file system" in stderr
                or "Network is unreachable" in stderr
                or "Docker socket" in stderr
            ):
                err_type = "security_violation"
            else:
                err_type = "runtime_error"

            last_line = stderr.splitlines()[-1] if stderr.splitlines() else f"Process exited with code {res.returncode}"
            return ExecutionResult(
                passed=False,
                error_type=err_type,
                error_message=last_line,
                stdout=res.stdout,
                stderr=stderr,
                duration_sec=duration,
                isolation_engine="bwrap",
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                passed=False,
                error_type="timeout",
                error_message=f"Execution exceeded timeout of {timeout_sec}s",
                stdout=exc.stdout if exc.stdout else "",
                stderr=exc.stderr if exc.stderr else "TimeoutExpired",
                duration_sec=duration,
                isolation_engine="bwrap",
            )
        except Exception as exc:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                passed=False,
                error_type="runtime_error",
                error_message=str(exc),
                stdout="",
                stderr=str(exc),
                duration_sec=duration,
                isolation_engine="bwrap",
            )

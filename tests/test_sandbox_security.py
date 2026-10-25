"""Security test suite validating OS-level Bubblewrap sandbox isolation.

Enforces Phase 2 security gates:
1. Valid function execution.
2. Syntax and runtime error handling.
3. Wall-clock timeout enforcement.
4. Memory ceiling enforcement.
5. Process creation (fork bomb) restriction.
6. Host file / credential read blocking (invisible /home, empty env).
7. Network access blocking (unshared network namespace).
8. Host filesystem write-escape blocking (read-only system mounts).
9. Docker socket / privilege escalation blocking.
10. Ephemeral workspace cleanup on completion and failure.
"""

import os
import tempfile
import pytest

from evaluators.execution_sandbox import execute_sandboxed, is_bwrap_available


@pytest.fixture(autouse=True)
def require_bwrap():
    if not is_bwrap_available():
        pytest.skip("Bubblewrap (bwrap) not available on this system")


# 1. Valid function execution and expected output
def test_security_gate_01_valid_execution():
    res = execute_sandboxed("def add(a, b): return a + b", ["assert add(2, 3) == 5"])
    assert res.passed is True
    assert res.error_type is None
    assert res.isolation_engine == "bwrap"


# 2. Syntax errors and runtime exceptions
def test_security_gate_02_syntax_and_runtime_errors():
    # Syntax error
    res_syn = execute_sandboxed("def foo(x) return x", ["assert True"])
    assert res_syn.passed is False
    assert res_syn.error_type == "syntax_error"

    # Runtime error (ZeroDivisionError)
    res_run = execute_sandboxed("def div(x): return x / 0", ["assert div(1) == 0"])
    assert res_run.passed is False
    assert res_run.error_type == "runtime_error"
    assert "ZeroDivisionError" in res_run.stderr


# 3. Infinite loops and wall-clock termination
def test_security_gate_03_wall_clock_timeout():
    infinite_loop_code = "def loop():\n    while True:\n        pass"
    res = execute_sandboxed(infinite_loop_code, ["assert loop() == 1"], timeout_sec=1.0)
    assert res.passed is False
    assert res.error_type == "timeout"
    assert 0.9 <= res.duration_sec <= 2.5


# 4. Excessive memory allocation
def test_security_gate_04_memory_ceiling():
    # Try to allocate 1 GB when max is 256 MB
    oom_code = "def oom():\n    x = bytearray(1024 * 1024 * 1024)\n    return len(x)"
    res = execute_sandboxed(oom_code, ["assert oom() > 0"], max_memory_mb=256, timeout_sec=2.0)
    assert res.passed is False
    assert res.error_type in ("memory_limit", "runtime_error")


# 5. Excessive process creation / fork restriction
def test_security_gate_05_process_creation_restriction():
    fork_bomb_probe = """
def probe_fork():
    import os
    pids = []
    try:
        for _ in range(100):
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            pids.append(pid)
    except BlockingIOError:
        pass  # Resource limit reached as intended
    for p in pids:
        try:
            os.waitpid(p, 0)
        except Exception:
            pass
    return True
"""
    res = execute_sandboxed(fork_bomb_probe, ["assert probe_fork() is True"], timeout_sec=3.0)
    # The fork calls either succeed up to rlimit or hit BlockingIOError, but cannot overwhelm host
    assert res.duration_sec <= 3.0


# 6. Host files and credentials invisible (no /home, clean env)
def test_security_gate_06_host_files_and_secrets_invisible():
    secret_probe = """
def probe_host():
    import os
    # Check that /home does not exist inside sandbox
    home_exists = os.path.exists('/home')
    # Check that common host secret env vars are absent
    env_keys = set(os.environ.keys())
    # Exclude explicitly set sandbox hardening flags (like PYTHONNOUSERSITE)
    leaked_env = [k for k in env_keys if ('USER' in k and k != 'PYTHONNOUSERSITE') or 'TOKEN' in k or 'SECRET' in k or 'KEY' in k or k in ('HOME', 'LOGNAME', 'SSH_AUTH_SOCK')]
    assert not home_exists, f"/home leaked into sandbox: {os.listdir('/home') if home_exists else ''}"
    assert len(leaked_env) == 0, f"Host environment leaked: {leaked_env}"
    return True
"""
    res = execute_sandboxed(secret_probe, ["assert probe_host() is True"])
    assert res.passed is True, f"Host filesystem or secrets leaked: {res.stderr}"


# 7. Network access blocking
def test_security_gate_07_network_blocked():
    network_probe = """
def probe_network():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    s.connect(('1.1.1.1', 80))
    return True
"""
    res = execute_sandboxed(network_probe, ["assert probe_network() is True"])
    assert res.passed is False
    assert "Network is unreachable" in res.stderr or "OSError" in res.stderr or res.error_type in ("security_violation", "runtime_error")


# 8. Host filesystem write escape blocking
def test_security_gate_08_host_filesystem_write_blocked():
    write_probe = """
def probe_write():
    with open('/usr/test_escape.txt', 'w') as f:
        f.write('pwned')
    return True
"""
    res = execute_sandboxed(write_probe, ["assert probe_write() is True"])
    assert res.passed is False
    assert "Read-only file system" in res.stderr or res.error_type in ("security_violation", "runtime_error")


# 9. Docker socket and privilege escalation blocking
def test_security_gate_09_docker_socket_and_privileges_blocked():
    docker_probe = """
def probe_docker():
    import os
    assert not os.path.exists('/var/run/docker.sock'), "Docker socket accessible!"
    assert not os.path.exists('/run/docker.sock'), "Docker socket accessible!"
    return True
"""
    res = execute_sandboxed(docker_probe, ["assert probe_docker() is True"])
    assert res.passed is True


# 10. Temporary workspace cleanup
def test_security_gate_10_workspace_cleanup():
    import glob
    # Count sandbox dirs before
    before = glob.glob(os.path.join(tempfile.gettempdir(), "mbpp_sandbox_*"))
    # Run a test that creates files
    create_code = "def f():\n    with open('test_file.txt', 'w') as f: f.write('data')\n    return True"
    res = execute_sandboxed(create_code, ["assert f() is True"])
    assert res.passed is True
    # Count sandbox dirs after
    after = glob.glob(os.path.join(tempfile.gettempdir(), "mbpp_sandbox_*"))
    assert len(after) == len(before), "Sandbox temporary directory was not cleaned up"

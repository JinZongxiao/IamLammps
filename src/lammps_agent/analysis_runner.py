"""Explicit Python analysis execution, separate from generic shell access."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_TAIL_BYTES = 12_000
_MAX_OUTPUT_BYTES = 1_000_000


def run_analysis_script(
    script: Path,
    work_dir: Path,
    timeout: int = 120,
    *,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> dict[str, object]:
    """Execute an approved analysis script with bounded captured output."""

    if script.suffix != ".py":
        raise ValueError("Analysis entry point must be a .py file")
    if max_output_bytes < _TAIL_BYTES:
        raise ValueError(f"max_output_bytes must be at least {_TAIL_BYTES}")
    work_dir = work_dir.resolve(strict=True)
    script = script.resolve(strict=False)
    if not script.is_relative_to(work_dir):
        raise ValueError("Analysis script must be inside its work directory")
    command = [sys.executable, "-I", script.name]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={
            "PATH": os.defpath,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )
    assert process.stdout is not None and process.stderr is not None
    stop_for_output = threading.Event()
    stdout_capture = _BoundedCapture(max_output_bytes, stop_for_output)
    stderr_capture = _BoundedCapture(max_output_bytes, stop_for_output)
    readers = [
        threading.Thread(
            target=stdout_capture.drain,
            args=(process.stdout,),
            name="iamlammps-analysis-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=stderr_capture.drain,
            args=(process.stderr,),
            name="iamlammps-analysis-stderr",
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + max(1, min(timeout, 600))
    timed_out = False
    output_limit_exceeded = False
    while process.poll() is None:
        if stop_for_output.wait(timeout=0.02):
            output_limit_exceeded = True
            _terminate_process_group(process)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_group(process)
            break
    process.wait()
    for reader in readers:
        reader.join(timeout=3)

    stdout = stdout_capture.text()
    stderr = stderr_capture.text()
    result = {
        "command": command,
        "return_code": -1 if timed_out or output_limit_exceeded else process.returncode,
        "timed_out": timed_out,
        "output_limit_exceeded": output_limit_exceeded,
        "wall_time": round(time.monotonic() - started, 2),
        "stdout": stdout,
        "stderr": stderr,
    }
    _write_result(work_dir, result)
    return result


class _BoundedCapture:
    def __init__(self, limit: int, exceeded: threading.Event) -> None:
        self.limit = limit
        self.exceeded = exceeded
        self.total = 0
        self.tail = bytearray()

    def drain(self, stream) -> None:
        while True:
            chunk = stream.read(65_536)
            if not chunk:
                return
            self.total += len(chunk)
            self.tail.extend(chunk)
            if len(self.tail) > _TAIL_BYTES:
                del self.tail[:-_TAIL_BYTES]
            if self.total > self.limit:
                self.exceeded.set()

    def text(self) -> str:
        return bytes(self.tail).decode("utf-8", errors="replace")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _write_result(work_dir: Path, result: dict[str, object]) -> None:
    """Persist through a temporary regular file without following the target."""

    path = work_dir / "analysis_result.json"
    temporary = work_dir / f".analysis_result.{os.getpid()}.{time.time_ns()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        raw = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

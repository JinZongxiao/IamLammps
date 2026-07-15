"""Process-boundary tests for the local LAMMPS runner."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lammps_agent.lammps_runner import run_lammps


def test_runner_builds_mpi_command_and_uses_isolated_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "in.lammps"
    input_path.write_text("run 0\n", encoding="utf-8")
    process = MagicMock()
    process.wait.return_value = 0
    monkeypatch.setenv("OPENAI_API_KEY", "fake-secret")

    with patch("lammps_agent.lammps_runner.subprocess.Popen", return_value=process) as popen:
        result = run_lammps(
            command="lmp",
            input_file=input_path,
            work_dir=tmp_path,
            timeout=30,
            mpi_np=4,
        )

    expected = ["mpirun", "-np", "4", "lmp", "-in", str(input_path.resolve())]
    assert result.command == expected
    assert result.return_code == 0
    process.wait.assert_called_once_with(timeout=30)
    args, kwargs = popen.call_args
    assert args == (expected,)
    assert kwargs["cwd"] == tmp_path
    assert kwargs["start_new_session"] is True
    assert "OPENAI_API_KEY" not in kwargs["env"]


def test_runner_persists_logs_result_and_new_output_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = run_dir / "in.lammps"
    input_path.write_text("run 0\n", encoding="utf-8")
    (run_dir / "already-there.dat").write_text("old", encoding="utf-8")

    fake_lammps = tmp_path / "fake-lammps"
    fake_lammps.write_text(
        "#!/bin/sh\n"
        "printf 'simulated stdout\\n'\n"
        "printf 'simulated stderr\\n' >&2\n"
        "printf 'trajectory' > trajectory.dump\n"
        "exit 7\n",
        encoding="utf-8",
    )
    fake_lammps.chmod(0o755)

    result = run_lammps(
        command=str(fake_lammps),
        input_file=input_path,
        work_dir=run_dir,
    )

    assert result.return_code == 7
    assert result.timed_out is False
    assert result.output_files == ["trajectory.dump"]
    assert (run_dir / "stdout.log").read_text(encoding="utf-8") == "simulated stdout\n"
    assert (run_dir / "stderr.log").read_text(encoding="utf-8") == "simulated stderr\n"

    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted == {
        "command": [str(fake_lammps), "-in", str(input_path.resolve())],
        "return_code": 7,
        "stdout_path": str(run_dir / "stdout.log"),
        "stderr_path": str(run_dir / "stderr.log"),
        "wall_time": result.wall_time,
        "timed_out": False,
        "output_files": ["trajectory.dump"],
    }


def test_timeout_terminates_the_entire_child_process_group(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = run_dir / "in.lammps"
    input_path.write_text("run 0\n", encoding="utf-8")

    fake_lammps = tmp_path / "slow-lammps"
    fake_lammps.write_text(
        "#!/bin/sh\n"
        "sleep 30 &\n"
        "child=$!\n"
        "printf '%s' \"$child\" > child.pid\n"
        "wait \"$child\"\n",
        encoding="utf-8",
    )
    fake_lammps.chmod(0o755)

    real_popen = subprocess.Popen

    def start_then_timeout(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 2
        while not (run_dir / "child.pid").exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise AssertionError("fixture process did not start its child")
            time.sleep(0.01)
        return process

    started = time.monotonic()
    with patch("lammps_agent.lammps_runner.subprocess.Popen", side_effect=start_then_timeout):
        result = run_lammps(
            command=str(fake_lammps),
            input_file=input_path,
            work_dir=run_dir,
            timeout=0.05,
        )

    assert result.return_code == -1
    assert result.timed_out is True
    assert time.monotonic() - started < 5

    child_pid = int((run_dir / "child.pid").read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, signal.SIGCONT)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"child process {child_pid} survived runner timeout")

    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted["return_code"] == -1
    assert persisted["timed_out"] is True


def test_runner_refuses_fixed_log_symlinks(tmp_path: Path) -> None:
    input_path = tmp_path / "in.lammps"
    input_path.write_text("run 0\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.log"
    outside.write_text("keep\n", encoding="utf-8")
    (tmp_path / "stdout.log").symlink_to(outside)

    with pytest.raises(OSError):
        run_lammps(command="lmp", input_file=input_path, work_dir=tmp_path)

    assert outside.read_text(encoding="utf-8") == "keep\n"

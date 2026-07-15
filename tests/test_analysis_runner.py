"""Process-boundary tests for explicit Python analysis execution."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lammps_agent.analysis_runner import run_analysis_script


def test_analysis_rejects_non_python_entry_point_before_starting_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "analysis.sh"

    with patch("lammps_agent.analysis_runner.subprocess.Popen") as popen:
        with pytest.raises(ValueError, match="must be a .py file"):
            run_analysis_script(script, tmp_path)

    popen.assert_not_called()


def test_analysis_runs_isolated_in_work_directory_and_persists_result(
    tmp_path: Path,
) -> None:
    script = tmp_path / "analyze.py"
    script.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "print(Path.cwd().name)\n"
        "print(os.environ.get('SECRET_TOKEN', 'isolated'))\n"
        "print('warning', file=__import__('sys').stderr)\n"
        "Path('derived.dat').write_text('42', encoding='utf-8')\n",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"SECRET_TOKEN": "must-not-leak"}):
        result = run_analysis_script(script, tmp_path, timeout=10)

    assert result["command"] == [os.sys.executable, "-I", "analyze.py"]
    assert result["return_code"] == 0
    assert result["timed_out"] is False
    assert result["output_limit_exceeded"] is False
    assert result["stdout"] == f"{tmp_path.name}\nisolated\n"
    assert result["stderr"] == "warning\n"
    assert result["wall_time"] >= 0
    assert (tmp_path / "derived.dat").read_text(encoding="utf-8") == "42"
    persisted = json.loads(
        (tmp_path / "analysis_result.json").read_text(encoding="utf-8")
    )
    assert persisted == result


def test_analysis_returns_only_the_tail_of_large_output(tmp_path: Path) -> None:
    script = tmp_path / "analyze.py"
    script.write_text(
        "import sys\n"
        "print('A' * 13000, end='')\n"
        "print('B' * 13000, end='', file=sys.stderr)\n",
        encoding="utf-8",
    )

    result = run_analysis_script(script, tmp_path, timeout=10)

    assert result["return_code"] == 0
    assert result["stdout"] == "A" * 12_000
    assert result["stderr"] == "B" * 12_000


def test_analysis_timeout_terminates_the_process_group(tmp_path: Path) -> None:
    script = tmp_path / "analyze.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    started = time.monotonic()
    result = run_analysis_script(script, tmp_path, timeout=1)

    assert result["return_code"] == -1
    assert result["timed_out"] is True
    assert result["output_limit_exceeded"] is False
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert time.monotonic() - started < 5


def test_analysis_timeout_escalates_to_sigkill_when_sigterm_is_ignored() -> None:
    process = MagicMock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [
        __import__("subprocess").TimeoutExpired(["python"], 3),
        0,
    ]

    from lammps_agent.analysis_runner import _terminate_process_group

    with patch("lammps_agent.analysis_runner.os.killpg") as killpg:
        _terminate_process_group(process)

    assert killpg.call_args_list[0].args == (12345, signal.SIGTERM)
    assert killpg.call_args_list[1].args == (12345, signal.SIGKILL)
    assert process.wait.call_count == 2


def test_analysis_stops_when_output_exceeds_limit(tmp_path: Path) -> None:
    script = tmp_path / "analyze.py"
    script.write_text("while True: print('x' * 1000)\n", encoding="utf-8")

    result = run_analysis_script(script, tmp_path, timeout=10, max_output_bytes=12_000)

    assert result["return_code"] == -1
    assert result["timed_out"] is False
    assert result["output_limit_exceeded"] is True
    assert len(result["stdout"].encode()) <= 12_000


def test_analysis_uses_fixed_path_and_does_not_expose_api_keys(tmp_path: Path) -> None:
    script = tmp_path / "analyze.py"
    script.write_text(
        "import os\n"
        "print(os.environ.get('PATH'))\n"
        "print(os.environ.get('OPENAI_API_KEY', 'hidden'))\n",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"PATH": "/secret/bin", "OPENAI_API_KEY": "secret"}):
        result = run_analysis_script(script, tmp_path)

    assert result["stdout"] == f"{os.defpath}\nhidden\n"


def test_analysis_result_replaces_symlink_without_writing_its_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("do not replace", encoding="utf-8")
    (tmp_path / "analysis_result.json").symlink_to(outside)
    script = tmp_path / "analyze.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    result = run_analysis_script(script, tmp_path)

    assert result["return_code"] == 0
    assert outside.read_text(encoding="utf-8") == "do not replace"
    assert not (tmp_path / "analysis_result.json").is_symlink()

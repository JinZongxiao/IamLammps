"""Real SPC/E integration test for the packaged 5000-step example."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from lammps_agent.lammps_runner import run_lammps
from lammps_agent.project_resources import initialize_project
from lammps_agent.script_policy import validate_lammps_script


@pytest.mark.integration
def test_real_spce_100_molecule_run(tmp_path: Path) -> None:
    configured = os.getenv("LAMMPS_COMMAND", "").strip()
    command = shutil.which(configured) if configured else None
    command = command or shutil.which("lmp") or shutil.which("lmp_mpi")
    if command is None:
        message = "The SPC/E integration test requires a LAMMPS executable"
        if os.getenv("IAMLAMMPS_REQUIRE_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)

    initialize_project(tmp_path)
    inputs = tmp_path / "runs" / "spce-100" / "inputs"
    input_path = inputs / "spce.in"
    validate_lammps_script(input_path.read_text(encoding="utf-8"))

    result = run_lammps(
        command=command,
        input_file=input_path,
        work_dir=inputs,
        timeout=60,
    )

    stdout = Path(result.stdout_path).read_text(encoding="utf-8")
    assert result.return_code == 0, Path(result.stderr_path).read_text(encoding="utf-8")
    assert result.timed_out is False
    assert "Created 300 atoms" in stdout
    assert "for 3000 steps with 300 atoms" in stdout
    assert "for 2000 steps with 300 atoms" in stdout
    assert (inputs / "rdf_oo.dat").stat().st_size > 0

"""Tests for backends package."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.lammps_agent.backends.base import JobBackend
from src.lammps_agent.backends.local import LocalBackend
from src.lammps_agent.backends.slurm import SlurmBackend
from src.lammps_agent.core_types import SimulationTask, SimulationResult
from src.lammps_agent.lammps_runner import LammpsRunResult


def _make_mock_lammps_result(return_code: int = 0) -> LammpsRunResult:
    return LammpsRunResult(
        command=["lmp_mpi", "-in", "in.lammps"],
        return_code=return_code,
        stdout_path="/tmp/stdout.log",
        stderr_path="/tmp/stderr.log",
        wall_time=1.23,
        timed_out=False,
        output_files=["restart.minimize"],
    )


class TestLocalBackend:
    def test_submit_success(self, tmp_path: Path):
        backend = LocalBackend(lammps_command="lmp_mpi")
        task = SimulationTask(code="run 0", work_dir=tmp_path)

        mock_result = _make_mock_lammps_result(return_code=0)
        with patch("src.lammps_agent.backends.local.run_lammps", return_value=mock_result):
            job_id = backend.submit(task)

        assert isinstance(job_id, str)
        assert len(job_id) == 12
        assert backend.poll(job_id) == "completed"

        result = backend.collect(job_id)
        assert result.success is True
        assert result.raw_result is mock_result
        assert result.timing["wall_time"] == 1.23

    def test_submit_failure(self, tmp_path: Path):
        backend = LocalBackend(lammps_command="lmp_mpi")
        task = SimulationTask(code="bad script", work_dir=tmp_path)

        mock_result = _make_mock_lammps_result(return_code=1)
        with patch("src.lammps_agent.backends.local.run_lammps", return_value=mock_result):
            job_id = backend.submit(task)

        assert backend.poll(job_id) == "failed"
        result = backend.collect(job_id)
        assert result.success is False
        assert "return_code=1" in result.errors

    def test_poll_unknown_job(self):
        backend = LocalBackend()
        assert backend.poll("nonexistent") == "pending"

    def test_run_sync(self, tmp_path: Path):
        backend = LocalBackend(lammps_command="lmp_mpi")
        task = SimulationTask(code="run 0", work_dir=tmp_path)

        mock_result = _make_mock_lammps_result(return_code=0)
        with patch("src.lammps_agent.backends.local.run_lammps", return_value=mock_result):
            result = backend.run_sync(task)

        assert isinstance(result, SimulationResult)
        assert result.success is True


class TestSlurmBackend:
    def test_submit_not_implemented(self):
        backend = SlurmBackend(partition="gpu")
        task = SimulationTask(code="run 100")
        with pytest.raises(NotImplementedError, match="SlurmBackend.submit"):
            backend.submit(task)

    def test_poll_not_implemented(self):
        backend = SlurmBackend()
        with pytest.raises(NotImplementedError, match="SlurmBackend.poll"):
            backend.poll("12345")

    def test_collect_not_implemented(self):
        backend = SlurmBackend()
        with pytest.raises(NotImplementedError, match="SlurmBackend.collect"):
            backend.collect("12345")

    def test_init_params(self):
        backend = SlurmBackend(partition="compute", account="myaccount", qos="high")
        assert backend.partition == "compute"
        assert backend.account == "myaccount"
        assert backend.qos == "high"

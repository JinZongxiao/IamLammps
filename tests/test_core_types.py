"""Tests for core_types module."""

from pathlib import Path

from src.lammps_agent.core_types import SimulationTask, SimulationResult


def test_simulation_task_defaults():
    task = SimulationTask(code="run 100")
    assert task.code == "run 100"
    assert task.backend == "local"
    assert task.resources == {}
    assert task.metadata == {}
    assert task.work_dir is None


def test_simulation_task_full():
    task = SimulationTask(
        code="minimize 1e-10 1e-10 1000 10000",
        backend="slurm",
        resources={"mpi_np": 4, "timeout": 300},
        metadata={"workflow": "tensile"},
        work_dir=Path("/tmp/test"),
    )
    assert task.backend == "slurm"
    assert task.resources["mpi_np"] == 4
    assert task.work_dir == Path("/tmp/test")


def test_simulation_result_defaults():
    result = SimulationResult(success=True)
    assert result.success is True
    assert result.outputs == {}
    assert result.timing == {}
    assert result.errors == []
    assert result.raw_result is None


def test_simulation_result_with_raw():
    sentinel = object()
    result = SimulationResult(
        success=False,
        outputs={"stdout_path": "/tmp/stdout.log"},
        timing={"wall_time": 3.14},
        errors=["return_code=1"],
        raw_result=sentinel,
    )
    assert result.success is False
    assert result.raw_result is sentinel
    assert result.errors == ["return_code=1"]

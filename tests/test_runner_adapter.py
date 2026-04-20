"""Tests for runner_adapter module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from src.lammps_agent.runner_adapter import get_backend, run_via_backend
from src.lammps_agent.backends.local import LocalBackend
from src.lammps_agent.config import AgentConfig


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        llm_provider="openai_compatible",
        openai_base_url="http://localhost",
        openai_api_key="",
        openai_model="test",
        local_base_url="http://localhost",
        local_model="test",
        lammps_command="lmp_mpi",
        enabled_skills=["thermo_tail"],
        memory_file="runs/memory.jsonl",
        memory_max_entries=100,
        memory_context_chars=2000,
        max_retries=3,
        backend="local",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_get_backend_local():
    config = _make_config(backend="local")
    backend = get_backend(config)
    assert isinstance(backend, LocalBackend)
    assert backend.lammps_command == "lmp_mpi"


def test_get_backend_slurm():
    config = _make_config(backend="slurm")
    backend = get_backend(config)
    from src.lammps_agent.backends.slurm import SlurmBackend
    assert isinstance(backend, SlurmBackend)


def test_get_backend_unknown():
    config = _make_config(backend="unknown")
    import pytest
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend(config)


def test_run_via_backend(tmp_path: Path):
    config = _make_config(backend="local")

    from src.lammps_agent.core_types import SimulationResult
    mock_result = SimulationResult(success=True, outputs={"test": True})

    with patch("src.lammps_agent.runner_adapter.get_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.run_sync.return_value = mock_result
        mock_get.return_value = mock_backend

        result = run_via_backend(config, code="run 0", work_dir=tmp_path)

    assert result.success is True
    mock_backend.run_sync.assert_called_once()

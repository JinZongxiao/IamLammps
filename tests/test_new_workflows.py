"""Tests for new workflows: interface_mechanics and he_trapping."""

from unittest.mock import MagicMock

import pytest

from src.lammps_agent.workflows import (
    build_workflow,
    WorkflowConfig,
    TensileWorkflow,
    NanoindentWorkflow,
    CrackWorkflow,
    DBTTSweepWorkflow,
    InterfaceMechanicsWorkflow,
    HeTrappingWorkflow,
)
from src.lammps_agent.pipeline import SimulationPipeline


def _make_mock_agent():
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.lammps_timeout = None
    return agent


class TestBuildWorkflowFactory:
    """Verify that the factory returns the correct types — including old ones."""

    def test_tensile(self):
        wf = build_workflow("tensile", WorkflowConfig(), _make_mock_agent())
        assert isinstance(wf, TensileWorkflow)

    def test_nanoindent(self):
        wf = build_workflow("nanoindent", WorkflowConfig(), _make_mock_agent())
        assert isinstance(wf, NanoindentWorkflow)

    def test_crack(self):
        wf = build_workflow("crack", WorkflowConfig(), _make_mock_agent())
        assert isinstance(wf, CrackWorkflow)

    def test_dbtt(self):
        wf = build_workflow("dbtt", WorkflowConfig(), _make_mock_agent())
        assert isinstance(wf, DBTTSweepWorkflow)

    def test_interface(self):
        wf = build_workflow("interface", WorkflowConfig(), _make_mock_agent())
        assert isinstance(wf, InterfaceMechanicsWorkflow)

    def test_he_trapping(self):
        wf = build_workflow("he_trapping", WorkflowConfig(), _make_mock_agent())
        assert isinstance(wf, HeTrappingWorkflow)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown workflow"):
            build_workflow("nonexistent", WorkflowConfig(), _make_mock_agent())


class TestInterfaceMechanicsWorkflow:
    def test_build_pipeline_stages(self):
        agent = _make_mock_agent()
        wf = InterfaceMechanicsWorkflow(
            config=WorkflowConfig(temperature=500.0),
            agent=agent,
            supercell=(8, 8, 16),
            oxide_type="Al2O3",
        )
        pipeline = wf.build_pipeline()
        assert isinstance(pipeline, SimulationPipeline)
        stage_names = [s.name for s in pipeline.stages]
        assert stage_names == ["build_interface", "minimize", "equilibrate", "tensile_normal"]

    def test_stage_dependencies(self):
        wf = InterfaceMechanicsWorkflow(WorkflowConfig(), _make_mock_agent())
        pipeline = wf.build_pipeline()
        stages = {s.name: s for s in pipeline.stages}
        assert stages["build_interface"].depends_on is None
        assert stages["minimize"].depends_on == "build_interface"
        assert stages["equilibrate"].depends_on == "minimize"
        assert stages["tensile_normal"].depends_on == "equilibrate"

    def test_default_params(self):
        wf = InterfaceMechanicsWorkflow(WorkflowConfig(), _make_mock_agent())
        assert wf.oxide_type == "Y2O3"
        assert wf.interface_plane == "(110)"
        assert wf.supercell == (10, 10, 20)


class TestHeTrappingWorkflow:
    def test_build_pipeline_stages(self):
        wf = HeTrappingWorkflow(
            config=WorkflowConfig(temperature=300.0),
            agent=_make_mock_agent(),
            n_vacancies=5,
            n_he_atoms=10,
        )
        pipeline = wf.build_pipeline()
        assert isinstance(pipeline, SimulationPipeline)
        stage_names = [s.name for s in pipeline.stages]
        assert stage_names == ["create_defects", "minimize", "insert_he", "anneal", "analyze"]

    def test_stage_dependencies(self):
        wf = HeTrappingWorkflow(WorkflowConfig(), _make_mock_agent())
        pipeline = wf.build_pipeline()
        stages = {s.name: s for s in pipeline.stages}
        assert stages["create_defects"].depends_on is None
        assert stages["minimize"].depends_on == "create_defects"
        assert stages["insert_he"].depends_on == "minimize"
        assert stages["anneal"].depends_on == "insert_he"
        assert stages["analyze"].depends_on == "anneal"

    def test_default_params(self):
        wf = HeTrappingWorkflow(WorkflowConfig(), _make_mock_agent())
        assert wf.n_vacancies == 10
        assert wf.n_he_atoms == 20
        assert wf.anneal_temperature == 1000.0
        assert wf.supercell == (10, 10, 10)

"""Smoke tests for domain extension packages: structures, dft, mlip."""

import pytest


class TestStructures:
    def test_import(self):
        from src.lammps_agent.structures import InterfaceBuilder
        assert InterfaceBuilder is not None

    def test_instantiate(self):
        from src.lammps_agent.structures import InterfaceBuilder
        builder = InterfaceBuilder(oxide_name="Y2O3")
        assert builder.oxide.name == "Y2O3"

    def test_unknown_oxide_raises(self):
        from src.lammps_agent.structures import InterfaceBuilder
        with pytest.raises(ValueError, match="Unknown oxide"):
            InterfaceBuilder(oxide_name="TiO2")

    def test_list_oxides(self):
        from src.lammps_agent.structures import InterfaceBuilder
        oxides = InterfaceBuilder.list_oxides()
        assert "Y2O3" in oxides
        assert "Al2O3" in oxides

    def test_build_bilayer_not_implemented(self):
        from src.lammps_agent.structures import InterfaceBuilder
        builder = InterfaceBuilder("Y2O3")
        with pytest.raises(NotImplementedError):
            builder.build_bilayer_slab()

    def test_compute_area(self):
        from src.lammps_agent.structures import InterfaceBuilder
        builder = InterfaceBuilder("Y2O3")
        assert builder.compute_interface_area(10.0, 20.0) == 200.0


class TestDFT:
    def test_import(self):
        from src.lammps_agent.dft import VASPWorkflow
        assert VASPWorkflow is not None

    def test_instantiate(self):
        from src.lammps_agent.dft import VASPWorkflow
        wf = VASPWorkflow(vasp_command="vasp_std")
        assert wf.vasp_command == "vasp_std"

    def test_methods_not_implemented(self):
        from src.lammps_agent.dft import VASPWorkflow
        wf = VASPWorkflow()
        with pytest.raises(NotImplementedError):
            wf.generate_input(structure=None)
        with pytest.raises(NotImplementedError):
            wf.submit(input_dir=None)
        from pathlib import Path
        with pytest.raises(NotImplementedError):
            wf.parse_output(Path("/tmp"))

    def test_dft_result(self):
        from src.lammps_agent.dft.vasp_workflow import DFTResult
        r = DFTResult(energy=-100.0, converged=True)
        assert r.energy == -100.0
        assert r.converged is True


class TestMLIP:
    def test_import(self):
        from src.lammps_agent.mlip import MLIPTrainer
        assert MLIPTrainer is not None

    def test_instantiate(self):
        from src.lammps_agent.mlip import MLIPTrainer
        trainer = MLIPTrainer(framework="deepmd")
        assert trainer.framework == "deepmd"

    def test_unknown_framework(self):
        from src.lammps_agent.mlip import MLIPTrainer
        with pytest.raises(ValueError, match="Unknown framework"):
            MLIPTrainer(framework="unknown")

    def test_supported_frameworks(self):
        from src.lammps_agent.mlip import MLIPTrainer
        for fw in ("deepmd", "mace", "ace"):
            trainer = MLIPTrainer(framework=fw)
            assert trainer.framework == fw

    def test_methods_not_implemented(self):
        from src.lammps_agent.mlip import MLIPTrainer
        trainer = MLIPTrainer()
        with pytest.raises(NotImplementedError):
            trainer.prepare_dataset([])
        with pytest.raises(NotImplementedError):
            trainer.train()

    def test_training_config(self):
        from src.lammps_agent.mlip.trainer import TrainingConfig
        cfg = TrainingConfig(framework="mace", epochs=200)
        assert cfg.framework == "mace"
        assert cfg.epochs == 200

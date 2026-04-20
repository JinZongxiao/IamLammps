"""Tests for new skills: interface_analysis and defect_analysis."""

from pathlib import Path

from src.lammps_agent.skills import _SKILL_REGISTRY, run_postprocess_skills
from src.lammps_agent.skills.interface_analysis import InterfaceAnalysisSkill
from src.lammps_agent.skills.defect_analysis import DefectAnalysisSkill


class TestInterfaceAnalysisSkill:
    def test_registered(self):
        assert "interface_analysis" in _SKILL_REGISTRY

    def test_no_stdout(self, tmp_path: Path):
        skill = InterfaceAnalysisSkill()
        report = skill.analyze(tmp_path)
        assert report.name == "interface_analysis"
        assert "No stdout.log" in report.summary

    def test_with_thermo_data(self, tmp_path: Path):
        stdout = tmp_path / "stdout.log"
        stdout.write_text(
            "Step PotEng Lz Pzz\n"
            "0 -100.0 50.0 0.0\n"
            "100 -99.5 50.1 -5000.0\n"
            "200 -98.0 50.3 -10000.0\n"
            "300 -97.0 50.5 -8000.0\n",
            encoding="utf-8",
        )
        skill = InterfaceAnalysisSkill()
        report = skill.analyze(tmp_path)
        assert report.name == "interface_analysis"
        assert "delta_PE" in report.summary
        assert "peak_stress" in report.summary
        assert report.details["data_points"] == 4

    def test_insufficient_data(self, tmp_path: Path):
        stdout = tmp_path / "stdout.log"
        stdout.write_text("Step PotEng\n0 -100.0\n", encoding="utf-8")
        skill = InterfaceAnalysisSkill()
        report = skill.analyze(tmp_path)
        assert "Insufficient" in report.summary


class TestDefectAnalysisSkill:
    def test_registered(self):
        assert "defect_analysis" in _SKILL_REGISTRY

    def test_no_dump_files(self, tmp_path: Path):
        skill = DefectAnalysisSkill()
        report = skill.analyze(tmp_path)
        assert report.name == "defect_analysis"
        assert "No dump files" in report.summary

    def test_with_coordination_dump(self, tmp_path: Path):
        dump = tmp_path / "dump.crack_final.lammpstrj"
        dump.write_text(
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n5\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type x y z c_coord\n"
            "1 1 0 0 0 14\n"
            "2 1 1 0 0 14\n"
            "3 1 2 0 0 10\n"  # vacancy-like (coord < 12)
            "4 1 3 0 0 14\n"
            "5 1 4 0 0 18\n",  # interstitial-like (coord > 16)
            encoding="utf-8",
        )
        skill = DefectAnalysisSkill()
        report = skill.analyze(tmp_path)
        assert report.name == "defect_analysis"
        assert report.details["n_atoms"] == 5
        assert report.details["n_vacancy"] == 1
        assert report.details["n_interstitial"] == 1

    def test_dump_without_coordination(self, tmp_path: Path):
        dump = tmp_path / "dump.test.lammpstrj"
        dump.write_text(
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n3\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type x y z\n"
            "1 1 0 0 0\n"
            "2 1 1 0 0\n"
            "3 1 2 0 0\n",
            encoding="utf-8",
        )
        skill = DefectAnalysisSkill()
        report = skill.analyze(tmp_path)
        assert "No coordination column" in report.summary
        assert report.details["n_atoms"] == 3


class TestSkillsIntegration:
    def test_run_new_skills_via_registry(self, tmp_path: Path):
        reports = run_postprocess_skills(tmp_path, ["interface_analysis", "defect_analysis"])
        assert len(reports) == 2
        assert reports[0].name == "interface_analysis"
        assert reports[1].name == "defect_analysis"

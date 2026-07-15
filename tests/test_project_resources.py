from __future__ import annotations

import json
from pathlib import Path

from lammps_agent.project_resources import initialize_project, run_doctor
from lammps_agent.skill_registry import SkillRegistry


def test_initialized_metadata_and_skills_are_valid(tmp_path: Path) -> None:
    initialize_project(tmp_path)

    skills = SkillRegistry(tmp_path).discover()
    metadata = json.loads(
        (
            tmp_path
            / "runs"
            / "spce-100"
            / "reference"
            / "plot_metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert {skill.name for skill in skills} == {
        "lammps-debugging",
        "potential-selection",
        "thermo-analysis",
    }
    assert all(skill.recommended_tools for skill in skills)
    assert metadata["plot_types"] == ["thermo", "rdf_oo"]
    assert "not evidence of converged" in metadata["scientific_status"]


def test_doctor_reads_configuration_without_disclosing_values(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "doctor-test-secret"
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=openai_agents\n"
        f"OPENAI_API_KEY={secret}\n"
        "OPENAI_MODEL=test-model\n"
        "LAMMPS_COMMAND=missing-lammps-for-test\n",
        encoding="utf-8",
    )
    for name in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "LAMMPS_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)

    checks = run_doctor(tmp_path)
    rendered = "\n".join(f"{check.name}: {check.detail}" for check in checks)

    assert next(check for check in checks if check.name == "model configuration").ok
    assert not next(check for check in checks if check.name == "lammps executable").ok
    assert secret not in rendered
    assert "test-model" not in rendered


def test_doctor_checks_required_lammps_packages(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "fake-lmp"
    executable.write_text(
        "#!/bin/sh\n"
        "printf 'Large-scale Atomic/Molecular Massively Parallel Simulator\\n'\n"
        "printf 'Installed packages: KSPACE MOLECULE RIGID\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("LAMMPS_COMMAND", str(executable))

    checks = run_doctor(tmp_path)

    assert next(check for check in checks if check.name == "lammps executable").ok
    assert next(check for check in checks if check.name == "lammps packages").ok


def test_doctor_reports_missing_lammps_package(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "fake-lmp"
    executable.write_text(
        "#!/bin/sh\nprintf 'Installed packages: KSPACE MOLECULE\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("LAMMPS_COMMAND", str(executable))

    package_check = next(
        check for check in run_doctor(tmp_path) if check.name == "lammps packages"
    )

    assert not package_check.ok
    assert package_check.detail == "missing required packages: RIGID"

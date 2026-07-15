from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lammps_agent.main import main
from lammps_agent.project_resources import DoctorCheck, initialize_project


def test_no_arguments_enters_interactive_shell_on_terminal() -> None:
    shell = MagicMock()
    shell.run.return_value = 0
    config = MagicMock()

    with patch("lammps_agent.main.sys.stdin.isatty", return_value=True), patch(
        "lammps_agent.main.sys.stdout.isatty", return_value=True
    ), patch("lammps_agent.main._load_config", return_value=config), patch(
        "lammps_agent.interactive.InteractiveShell", return_value=shell
    ) as shell_type:
        assert main([]) == 0

    shell_type.assert_called_once_with(config)
    shell.run.assert_called_once_with()


def test_no_arguments_returns_usage_error_without_terminal(capsys) -> None:
    with patch("lammps_agent.main.sys.stdin.isatty", return_value=False), patch(
        "lammps_agent.main.sys.stdout.isatty", return_value=False
    ), patch("lammps_agent.main._load_config") as load_config:
        assert main([]) == 2

    load_config.assert_not_called()
    assert "stdin is not a terminal" in capsys.readouterr().err


def test_explicit_interactive_flag_works_without_terminal() -> None:
    shell = MagicMock()
    shell.run.return_value = 0
    config = MagicMock()

    with patch("lammps_agent.main.sys.stdin.isatty", return_value=False), patch(
        "lammps_agent.main.sys.stdout.isatty", return_value=False
    ), patch("lammps_agent.main._load_config", return_value=config), patch(
        "lammps_agent.interactive.InteractiveShell", return_value=shell
    ) as shell_type:
        assert main(["--interactive"]) == 0

    shell_type.assert_called_once_with(config)
    shell.run.assert_called_once_with()


@pytest.mark.parametrize("arguments", [["--task", "legacy"], ["--workflow", "tensile"]])
def test_removed_batch_modes_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


def test_init_materializes_packaged_resources(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"

    assert main(["init", str(project)]) == 0

    assert (project / ".env.example").is_file()
    assert (project / ".iamlammps" / "mcp.example.json").is_file()
    skill = project / ".iamlammps" / "skills" / "thermo-analysis" / "SKILL.md"
    assert "recommended-tools:" in skill.read_text(encoding="utf-8")
    sample = project / "runs" / "spce-100" / "inputs" / "spce.in"
    script = sample.read_text(encoding="utf-8")
    assert "create_atoms 0 random 100" in script
    assert "run 3000" in script
    assert "run 2000" in script
    assert "compute rdf_oo all rdf" in script
    assert "created .env.example" in capsys.readouterr().out


def test_init_preserves_existing_files(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    env = project / ".env.example"
    env.write_text("KEEP_ME=true\n", encoding="utf-8")

    assert main(["init", str(project)]) == 0

    assert env.read_text(encoding="utf-8") == "KEEP_ME=true\n"
    assert "kept    .env.example" in capsys.readouterr().out


def test_init_is_repeatable_without_overwriting(tmp_path: Path) -> None:
    first = initialize_project(tmp_path)
    sample = tmp_path / "runs" / "spce-100" / "inputs" / "spce.in"
    sample.write_text("user version\n", encoding="utf-8")

    second = initialize_project(tmp_path)

    assert first.created
    assert not second.created
    assert Path("runs/spce-100/inputs/spce.in") in second.skipped
    assert sample.read_text(encoding="utf-8") == "user version\n"


def test_init_rejects_symlinked_resource_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / ".iamlammps").symlink_to(outside, target_is_directory=True)

    assert main(["init", str(project)]) == 1
    assert list(outside.iterdir()) == []


def test_doctor_prints_named_checks_without_secret(tmp_path: Path, capsys) -> None:
    secret = "test-secret-that-must-not-appear"
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=openai_agents\n"
        f"OPENAI_API_KEY={secret}\n"
        "OPENAI_MODEL=test-model\n",
        encoding="utf-8",
    )
    checks = (
        DoctorCheck("python", True, "3.13.0"),
        DoctorCheck("model configuration", True, "configured"),
        DoctorCheck("lammps executable", False, "not found"),
    )

    with patch("lammps_agent.main.run_doctor", return_value=checks):
        assert main(["doctor", str(tmp_path)]) == 1

    output = capsys.readouterr().out
    assert "[PASS] python" in output
    assert "[FAIL] lammps executable" in output
    assert "2/3 checks passed" in output
    assert secret not in output


def test_doctor_returns_success_when_all_checks_pass(tmp_path: Path) -> None:
    checks = (DoctorCheck("python", True, "3.13.0"),)
    with patch("lammps_agent.main.run_doctor", return_value=checks):
        assert main(["doctor", str(tmp_path)]) == 0

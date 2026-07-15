"""Direct SDK FunctionTool tests for the simulation agent's safety boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from agents.tool_context import ToolContext

from lammps_agent.config import AgentConfig
from lammps_agent.plotting import PlotDataError
from lammps_agent.potential_catalog import (
    LammpsPotentialCatalog,
    PotentialCandidate,
)
from lammps_agent.run_artifacts import RunManifest, RunStatus
from lammps_agent.simulation_tools import (
    SimulationContext,
    _scientific_preflight,
    build_simulation_tools,
)
from lammps_agent.skill_registry import SkillRegistry
from lammps_agent.web import SearchResult, WebResearch
from lammps_agent.workspace import Workspace


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        llm_provider="deepseek",
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="",
        openai_model="unused",
        lammps_command="lmp-test",
        enabled_analyzers=["thermo_tail"],
        lammps_timeout=30,
        mpi_np=2,
        potentials_dir=str(tmp_path / "potentials"),
        deepseek_api_key="not-used-by-tools",
    )


def _context(tmp_path: Path) -> SimulationContext:
    return SimulationContext(
        config=_config(tmp_path),
        workspace=Workspace(tmp_path),
        project_root=tmp_path,
        skill_registry=SkillRegistry(tmp_path),
        web=WebResearch(),
        potential_catalog=LammpsPotentialCatalog(),
    )


def _tools() -> dict[str, object]:
    return {tool.name: tool for tool in build_simulation_tools()}


def _invoke(tool: object, context: SimulationContext, **arguments: object) -> str:
    sdk_context = ToolContext(
        context=context,
        tool_name=tool.name,  # type: ignore[attr-defined]
        tool_call_id="test-call",
        tool_arguments=json.dumps(arguments),
    )
    return asyncio.run(
        tool.on_invoke_tool(  # type: ignore[attr-defined]
            sdk_context,
            json.dumps(arguments),
        )
    )


def test_mutating_and_execution_tools_require_sdk_approval() -> None:
    tools = _tools()

    for name in (
        "write_file",
        "replace_file",
        "run_lammps_input",
        "run_python_analysis",
        "stage_local_potential",
        "install_potential",
    ):
        assert tools[name].needs_approval is True  # type: ignore[attr-defined]
    assert tools["read_file"].needs_approval is False  # type: ignore[attr-defined]
    assert tools["analyze_run"].needs_approval is False  # type: ignore[attr-defined]
    assert tools["web_search"].needs_approval is True  # type: ignore[attr-defined]
    assert tools["fetch_web_result"].needs_approval is True  # type: ignore[attr-defined]
    assert tools["search_potentials"].needs_approval is True  # type: ignore[attr-defined]


def test_write_lammps_validates_content_and_returns_real_hash(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    script = "units metal\natom_style atomic\nrun 0\n"

    result = json.loads(
        _invoke(
            tools["write_file"],
            context,
            path="runs/demo/in.lammps",
            content=script,
            expected_sha256=None,
        )
    )

    expected = hashlib.sha256(script.encode()).hexdigest()
    assert result == {
        "path": "runs/demo/in.lammps",
        "size": len(script),
        "sha256": expected,
    }
    assert (tmp_path / result["path"]).read_text(encoding="utf-8") == script

    rejected = _invoke(
        tools["write_file"],
        context,
        path="runs/bad/in.lammps",
        content="shell curl https://example.com\n",
        expected_sha256=None,
    )
    assert "command 'shell' is not allowed" in rejected
    assert not (tmp_path / "runs/bad/in.lammps").exists()


def test_existing_file_returns_structured_conflict(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    existing = tmp_path / "runs" / "demo" / "inputs" / "run.in"
    existing.parent.mkdir(parents=True)
    existing.write_text("run 0\n", encoding="utf-8")

    result = json.loads(
        _invoke(
            tools["write_file"],
            context,
            path="runs/demo/inputs/run.in",
            content="run 1\n",
            expected_sha256=None,
        )
    )

    assert result["code"] == "FILE_EXISTS"
    assert result["sha256"] == hashlib.sha256(b"run 0\n").hexdigest()
    assert existing.read_text(encoding="utf-8") == "run 0\n"


def test_lists_reads_and_cas_replaces_workspace_files(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    path = tmp_path / "runs" / "demo" / "inputs" / "run.in"
    path.parent.mkdir(parents=True)
    source = "units real\nrun 0\n"
    path.write_text(source, encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()

    listed = json.loads(
        _invoke(tools["list_files"], context, directory="runs/demo", max_files=10)
    )
    read = json.loads(
        _invoke(
            tools["read_file"],
            context,
            path="runs/demo/inputs/run.in",
            max_bytes=1_000,
        )
    )
    replaced = json.loads(
        _invoke(
            tools["replace_file"],
            context,
            path="runs/demo/inputs/run.in",
            old_text="run 0",
            new_text="run 10",
            expected_sha256=digest,
        )
    )

    assert [item["path"] for item in listed] == ["runs/demo/inputs/run.in"]
    assert read["content"] == source
    assert replaced["sha256"] == hashlib.sha256(b"units real\nrun 10\n").hexdigest()
    assert path.read_text(encoding="utf-8") == "units real\nrun 10\n"


def test_replace_lammps_rejects_stale_hash_and_ambiguous_text(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    path = tmp_path / "runs" / "demo" / "inputs" / "run.in"
    path.parent.mkdir(parents=True)
    source = "run 0\nrun 0\n"
    path.write_text(source, encoding="utf-8")

    stale = _invoke(
        tools["replace_file"],
        context,
        path="runs/demo/inputs/run.in",
        old_text="run 0",
        new_text="run 1",
        expected_sha256="0" * 64,
    )
    ambiguous = _invoke(
        tools["replace_file"],
        context,
        path="runs/demo/inputs/run.in",
        old_text="run 0",
        new_text="run 1",
        expected_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )

    assert "changed since it was read" in stale
    assert "must occur exactly once" in ambiguous


@pytest.mark.parametrize(
    "path",
    [
        "runs/demo/latest.json",
        "runs/demo/executions/run-id/manifest.json",
        "Runs/demo/Executions/run-id/stdout.log",
    ],
)
def test_agent_cannot_create_or_update_managed_run_artifacts(
    tmp_path: Path, path: str
) -> None:
    tools = _tools()
    context = _context(tmp_path)

    result = _invoke(
        tools["write_file"],
        context,
        path=path,
        content="forged evidence\n",
        expected_sha256=None,
    )

    assert "managed run artifacts are read-only" in result
    assert not (tmp_path / path).exists()


def test_agent_cannot_replace_existing_manifest(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    manifest = tmp_path / "runs" / "demo" / "executions" / "run-id" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    source = '{"status":"succeeded"}\n'
    manifest.write_text(source, encoding="utf-8")

    result = _invoke(
        tools["replace_file"],
        context,
        path="runs/demo/executions/run-id/manifest.json",
        old_text="succeeded",
        new_text="failed",
        expected_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )

    assert "managed run artifacts are read-only" in result
    assert manifest.read_text(encoding="utf-8") == source


def test_scientific_preflight_warns_on_unit_and_cutoff_conflicts() -> None:
    warnings = _scientific_preflight(
        "units real\n"
        "lattice sc 0.1\n"
        "region box block 0 10 0 10 0 10\n"
        "pair_style lj/cut 6.0\n"
        "create_atoms 1 random 0 12345 box\n"
    )

    assert any("lattice scale" in warning for warning in warnings)
    assert any("half the smallest" in warning for warning in warnings)
    assert any("non-positive particle count" in warning for warning in warnings)

    water_warnings = _scientific_preflight(
        "units real\natom_style full\ncreate_atoms 1 single 0 0 0\n"
        "pair_style lj/cut/coul/long 6.5\nfix water all shake 0.0001 20 0 b 1 a 1\n"
    )
    assert any("without an explicit region" in warning for warning in water_warnings)
    assert any("no kspace_style" in warning for warning in water_warnings)


def test_run_lammps_rejects_changed_reviewed_file_without_starting_process(
    tmp_path: Path,
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    path = tmp_path / "runs" / "run" / "inputs" / "in.lammps"
    path.parent.mkdir(parents=True)
    original = "units metal\natom_style atomic\nrun 0\n"
    path.write_text(original, encoding="utf-8")
    reviewed_hash = hashlib.sha256(original.encode()).hexdigest()
    path.write_text("units metal\natom_style atomic\nrun 1\n", encoding="utf-8")

    with patch("lammps_agent.simulation_tools.RunArtifactStore.execute") as run:
        result = _invoke(
            tools["run_lammps_input"],
            context,
            input_path="runs/run/inputs/in.lammps",
            expected_sha256=reviewed_hash,
            timeout_seconds=None,
            mpi_processes=None,
        )

    assert "changed after it was reviewed" in result
    run.assert_not_called()


def test_run_lammps_rejects_reviewed_input_outside_runs_without_starting_process(
    tmp_path: Path,
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    path = tmp_path / "inputs" / "in.lammps"
    path.parent.mkdir()
    script = "units metal\natom_style atomic\nrun 0\n"
    path.write_text(script, encoding="utf-8")
    digest = hashlib.sha256(script.encode()).hexdigest()

    with patch("lammps_agent.simulation_tools.RunArtifactStore.execute") as run:
        result = _invoke(
            tools["run_lammps_input"],
            context,
            input_path="inputs/in.lammps",
            expected_sha256=digest,
            timeout_seconds=None,
            mpi_processes=None,
        )

    assert "must be below runs/<job>/inputs/" in result
    run.assert_not_called()


def test_run_lammps_uses_reviewed_path_and_bounded_overrides(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    path = tmp_path / "runs" / "run" / "inputs" / "in.lammps"
    path.parent.mkdir(parents=True)
    script = "units metal\natom_style atomic\nrun 0\n"
    path.write_text(script, encoding="utf-8")
    digest = hashlib.sha256(script.encode()).hexdigest()
    artifact = tmp_path / "runs" / "run" / "executions" / "test-run"
    artifact.mkdir(parents=True)
    stdout = artifact / "stdout.log"
    stderr = artifact / "stderr.log"
    stdout.write_text("LAMMPS finished\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    fake_result = RunManifest(
        schema_version=1,
        run_id="test-run",
        job="run",
        status=RunStatus.SUCCEEDED,
        created_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        source_input="runs/run/inputs/in.lammps",
        input_sha256=digest,
        artifact_dir="runs/run/executions/test-run",
        workspace_dir="runs/run/executions/test-run/workspace",
        stdout_path="runs/run/executions/test-run/stdout.log",
        stderr_path="runs/run/executions/test-run/stderr.log",
        lammps_log_path=None,
        manifest_path="runs/run/executions/test-run/manifest.json",
        return_code=0,
        wall_time=1.0,
        command=("lmp-test",),
        output_files=("runs/run/executions/test-run/workspace/dump.xyz",),
    )

    with patch(
        "lammps_agent.simulation_tools.RunArtifactStore.execute", return_value=fake_result
    ) as run:
        payload = json.loads(
            _invoke(
                tools["run_lammps_input"],
                context,
                input_path="runs/run/inputs/in.lammps",
                expected_sha256=digest,
                timeout_seconds=12,
                mpi_processes=3,
            )
        )

    assert run.call_args.args == (Path("runs/run/inputs/in.lammps"),)
    assert run.call_args.kwargs["command"] == "lmp-test"
    assert run.call_args.kwargs["expected_sha256"] == digest
    assert run.call_args.kwargs["timeout"] == 12
    assert run.call_args.kwargs["mpi_np"] == 3
    assert "OPENAI_API_KEY" not in run.call_args.kwargs["environment"]
    assert payload["return_code"] == 0
    assert payload["stdout_tail"] == "LAMMPS finished\n"
    assert payload["run_id"] == "test-run"
    assert payload["status"] == "succeeded"
    assert payload["output_files"] == ["runs/run/executions/test-run/workspace/dump.xyz"]


@pytest.mark.parametrize(
    ("timeout", "mpi", "message"),
    [(0, 1, "timeout_seconds"), (1, 257, "mpi_processes")],
)
def test_run_lammps_rejects_unbounded_process_parameters(
    tmp_path: Path, timeout: int, mpi: int, message: str
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    path = tmp_path / "runs" / "demo" / "inputs" / "run.in"
    path.parent.mkdir(parents=True)
    source = "units real\nrun 0\n"
    path.write_text(source, encoding="utf-8")

    with patch("lammps_agent.simulation_tools.RunArtifactStore.execute") as execute:
        result = _invoke(
            tools["run_lammps_input"],
            context,
            input_path="runs/demo/inputs/run.in",
            expected_sha256=hashlib.sha256(source.encode()).hexdigest(),
            timeout_seconds=timeout,
            mpi_processes=mpi,
        )

    assert message in result
    execute.assert_not_called()


def test_parse_and_analyze_run_use_local_deterministic_data(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    run_id = "20260101T000000.000000Z-aaaaaaaaaaaa"
    run_dir = tmp_path / "runs" / "demo" / "executions" / run_id
    run_dir.mkdir(parents=True)
    thermo = "Step Temp PotEng\n0 300 -1.0\n10 310 -1.1\n"
    (run_dir / "stdout.log").write_text(thermo, encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    manifest = RunManifest(
        schema_version=1,
        run_id=run_id,
        job="demo",
        status=RunStatus.SUCCEEDED,
        created_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        source_input="runs/demo/inputs/in.lammps",
        input_sha256="0" * 64,
        artifact_dir=f"runs/demo/executions/{run_id}",
        workspace_dir=f"runs/demo/executions/{run_id}/workspace",
        stdout_path=f"runs/demo/executions/{run_id}/stdout.log",
        stderr_path=f"runs/demo/executions/{run_id}/stderr.log",
        lammps_log_path=None,
        manifest_path=f"runs/demo/executions/{run_id}/manifest.json",
        return_code=0,
        wall_time=1.0,
        command=("lmp",),
        output_files=(),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )

    parsed = json.loads(
        _invoke(
            tools["parse_thermo"],
            context,
            run_id=run_id,
            tail_rows=1,
        )
    )
    reports = json.loads(
        _invoke(
            tools["analyze_run"],
            context,
            run_id=run_id,
            analyzer_names=["thermo_tail"],
        )
    )
    listed = json.loads(_invoke(tools["list_runs"], context, job="demo"))
    inspected = json.loads(_invoke(tools["inspect_run"], context, run_id=run_id))
    unknown = json.loads(
        _invoke(
            tools["analyze_run"],
            context,
            run_id=run_id,
            analyzer_names=["not-registered"],
        )
    )
    with patch(
        "lammps_agent.simulation_tools.create_run_plots",
        side_effect=PlotDataError("missing plot data"),
    ):
        plot_error = json.loads(
            _invoke(tools["plot_run"], context, run_id=run_id, plot_types=["thermo"])
        )

    assert parsed == {"Step": [10.0], "Temp": [310.0], "PotEng": [-1.1]}
    assert reports[0]["name"] == "thermo_tail"
    assert reports[0]["details"]["thermo_rows"] == 2
    assert [item["run_id"] for item in listed] == [run_id]
    assert inspected["run_id"] == run_id
    assert unknown == {
        "code": "UNKNOWN_ANALYZER",
        "message": "Unknown analyzer: not-registered",
        "analyzer": "not-registered",
    }
    assert plot_error["code"] == "PLOT_DATA_ERROR"


def test_python_analysis_can_only_execute_below_runs(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    script = tmp_path / "analysis.py"
    source = "print('not allowed')\n"
    script.write_text(source, encoding="utf-8")

    with patch("lammps_agent.simulation_tools.run_analysis_script") as run:
        result = _invoke(
            tools["run_python_analysis"],
            context,
            script_path="analysis.py",
            expected_sha256=hashlib.sha256(source.encode()).hexdigest(),
            timeout_seconds=10,
        )

    assert "only be executed below runs/" in result
    run.assert_not_called()


def test_python_analysis_requires_hash_and_returns_bounded_runner_result(
    tmp_path: Path,
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    script = tmp_path / "runs" / "demo" / "analyze.py"
    script.parent.mkdir(parents=True)
    source = "print('ok')\n"
    script.write_text(source, encoding="utf-8")

    with patch("lammps_agent.simulation_tools.run_analysis_script") as run:
        stale = _invoke(
            tools["run_python_analysis"],
            context,
            script_path="runs/demo/analyze.py",
            expected_sha256="0" * 64,
            timeout_seconds=10,
        )
        run.return_value = {"return_code": 0, "stdout": "ok\n"}
        completed = json.loads(
            _invoke(
                tools["run_python_analysis"],
                context,
                script_path="runs/demo/analyze.py",
                expected_sha256=hashlib.sha256(source.encode()).hexdigest(),
                timeout_seconds=10,
            )
        )

    assert "changed after it was reviewed" in stale
    assert completed == {"return_code": 0, "stdout": "ok\n"}
    run.assert_called_once_with(script, script.parent, 10)


def test_lists_local_potential_metadata_without_network(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    potential_dir = tmp_path / "potentials"
    potential_dir.mkdir()
    potential = potential_dir / "W_zhou.eam.alloy"
    potential.write_text("local potential data", encoding="utf-8")

    result = json.loads(_invoke(tools["list_local_potentials"], context))

    assert len(result) == 1
    assert result[0]["name"] == "W_Zhou2004_EAM"
    assert result[0]["elements"] == ["W"]
    assert result[0]["pair_style"] == "eam/alloy"
    assert result[0]["file"] == str(potential)


def test_stages_registered_potential_and_returns_exact_pair_commands(
    tmp_path: Path,
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    potential_dir = tmp_path / "potentials"
    potential_dir.mkdir()
    source = potential_dir / "W_zhou.eam.alloy"
    source.write_text("potential data", encoding="utf-8")

    result = json.loads(
        _invoke(
            tools["stage_local_potential"],
            context,
            potential_name="W_Zhou2004_EAM",
            run_directory="runs/w-demo/inputs",
        )
    )

    target = tmp_path / result["path"]
    assert target.read_text(encoding="utf-8") == "potential data"
    assert result["filename"] == "W_zhou.eam.alloy"
    assert result["pair_style"] == "eam/alloy"
    assert result["pair_commands"] == (
        "pair_style eam/alloy\npair_coeff * * W_zhou.eam.alloy W"
    )
    assert result["sha256"] == hashlib.sha256(b"potential data").hexdigest()

    repeated = _invoke(
        tools["stage_local_potential"],
        context,
        potential_name="W_Zhou2004_EAM",
        run_directory="runs/w-demo/inputs",
    )
    assert "Potential already staged" in repeated


def test_local_potential_can_only_be_staged_below_runs(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    potential_dir = tmp_path / "potentials"
    potential_dir.mkdir()
    (potential_dir / "W_zhou.eam.alloy").write_text("data", encoding="utf-8")

    result = _invoke(
        tools["stage_local_potential"],
        context,
        potential_name="W_Zhou2004_EAM",
        run_directory="potentials/copied",
    )

    assert "must be staged in runs/<job>/inputs" in result
    assert not (tmp_path / "potentials/copied/W_zhou.eam.alloy").exists()


def test_install_potential_rejects_metadata_not_matching_approved_candidate(
    tmp_path: Path,
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    candidate = PotentialCandidate(
        candidate_id="lammps-1",
        filename="W.eam.alloy",
        source="LAMMPS official GitHub release",
        revision="stable_22Jul2025_update3",
        download_url=(
            "https://raw.githubusercontent.com/lammps/lammps/"
            "stable_22Jul2025_update3/potentials/W.eam.alloy"
        ),
        git_blob_sha="abc123",
        size=42,
    )
    context.potential_catalog._candidates[candidate.candidate_id] = candidate

    for expected_filename, expected_revision in (
        ("different.eam.alloy", candidate.revision),
        (candidate.filename, "different-revision"),
    ):
        with patch.object(context.potential_catalog, "install") as install:
            result = _invoke(
                tools["install_potential"],
                context,
                candidate_id=candidate.candidate_id,
                expected_filename=expected_filename,
                expected_revision=expected_revision,
                expected_download_url=candidate.download_url,
                expected_git_blob_sha=candidate.git_blob_sha,
                expected_size=candidate.size,
                destination_directory="potentials",
            )

        assert "metadata does not match the approved values" in result
        install.assert_not_called()


def test_fetch_web_result_rejects_url_not_matching_approved_result(
    tmp_path: Path,
) -> None:
    tools = _tools()
    context = _context(tmp_path)
    result_id = "web-1"
    context.web._results[result_id] = SearchResult(
        result_id=result_id,
        title="LAMMPS documentation",
        url="https://docs.lammps.org/pair_eam.html",
        snippet="EAM pair style documentation",
    )

    with (
        patch("lammps_agent.web.socket.getaddrinfo") as resolve,
        patch("lammps_agent.web.httpx.stream") as stream,
    ):
        result = _invoke(
            tools["fetch_web_result"],
            context,
            result_id=result_id,
            expected_url="https://attacker.example/substituted-page",
            max_chars=12_000,
        )

    assert "URL does not match the approved URL" in result
    resolve.assert_not_called()
    stream.assert_not_called()


def test_list_and_load_registered_instruction_only_skill(tmp_path: Path) -> None:
    tools = _tools()
    context = _context(tmp_path)
    skill_file = tmp_path / ".iamlammps" / "skills" / "analyze" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        """---
name: analyze
description: Analyze completed simulations
recommended-tools: [read_file, parse_thermo]
---
Check convergence before interpreting a result.
""",
        encoding="utf-8",
    )

    listed = json.loads(_invoke(tools["list_skills"], context))
    loaded = json.loads(_invoke(tools["load_skill"], context, name="analyze"))

    assert [skill["name"] for skill in listed] == ["analyze"]
    assert loaded["recommended_tools"] == ["read_file", "parse_thermo"]
    assert loaded["instructions"] == "Check convergence before interpreting a result."
    assert loaded["path"] == str(skill_file)

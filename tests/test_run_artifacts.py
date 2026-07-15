"""Traceability and isolation tests for immutable execution artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lammps_agent.lammps_runner import LammpsRunResult
from lammps_agent.run_artifacts import RunArtifactStore, RunManifest, RunStatus


def _input(tmp_path: Path, job: str = "water") -> tuple[Path, str]:
    inputs = tmp_path / "runs" / job / "inputs"
    inputs.mkdir(parents=True)
    source = inputs / "in.lammps"
    source.write_text("units real\ninclude molecule.inc\nrun 0\n", encoding="utf-8")
    (inputs / "molecule.inc").write_text("# molecule data\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return source, digest


def _runner(
    *,
    command: str,
    input_file: Path,
    work_dir: Path,
    output_dir: Path,
    timeout: int | None,
    mpi_np: int | None,
    persist_result: bool,
    environment: dict[str, str] | None,
) -> LammpsRunResult:
    del timeout, mpi_np, persist_result, environment
    (output_dir / "stdout.log").write_text(
        "Step Temp\n0 300\n1 301\nLoop time\n", encoding="utf-8"
    )
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    (work_dir / "log.lammps").write_text("LAMMPS log\n", encoding="utf-8")
    (work_dir / "trajectory.dump").write_text("trajectory\n", encoding="utf-8")
    return LammpsRunResult(
        command=[command, "-in", str(input_file)],
        return_code=0,
        stdout_path=str(output_dir / "stdout.log"),
        stderr_path=str(output_dir / "stderr.log"),
        wall_time=0.25,
        output_files=["log.lammps", "trajectory.dump"],
    )


def test_execute_snapshots_inputs_and_preserves_consecutive_runs(tmp_path: Path) -> None:
    source, digest = _input(tmp_path)
    store = RunArtifactStore(tmp_path)

    first = store.execute(
        "runs/water/inputs/in.lammps",
        command="lmp",
        expected_sha256=digest,
        runner=_runner,
    )
    source.write_text("units real\nrun 10\n", encoding="utf-8")
    second_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    second = store.execute(
        "runs/water/inputs/in.lammps",
        command="lmp",
        expected_sha256=second_digest,
        runner=_runner,
    )

    assert first.run_id != second.run_id
    assert first.status is RunStatus.SUCCEEDED
    assert Path(first.stdout_path).is_absolute() is False
    assert all(Path(path).is_absolute() is False for path in first.output_files)
    assert all(not Path(part).is_absolute() for part in first.command)
    first_input = tmp_path / first.workspace_dir / "in.lammps"
    assert first_input.read_text(encoding="utf-8") == (
        "units real\ninclude molecule.inc\nrun 0\n"
    )
    assert (tmp_path / first.workspace_dir / "molecule.inc").is_file()
    assert (first_input.stat().st_mode & 0o222) == 0
    assert (tmp_path / first.stdout_path).is_file()
    assert (tmp_path / second.stdout_path).is_file()
    assert first.lammps_log_path == f"{first.workspace_dir}/log.lammps"
    assert store.inspect_run(first.run_id) == first
    assert [run.run_id for run in store.list_runs("water")] == [
        second.run_id,
        first.run_id,
    ]

    latest = json.loads((tmp_path / "runs/water/latest.json").read_text(encoding="utf-8"))
    assert latest == {"run_id": second.run_id, "manifest": second.manifest_path}


@pytest.mark.parametrize(
    ("return_code", "timed_out", "expected"),
    [(3, False, RunStatus.FAILED), (-1, True, RunStatus.TIMED_OUT)],
)
def test_failed_and_timed_out_runs_keep_manifests(
    tmp_path: Path,
    return_code: int,
    timed_out: bool,
    expected: RunStatus,
) -> None:
    _, digest = _input(tmp_path)

    def runner(**kwargs: object) -> LammpsRunResult:
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "stdout.log").write_text("partial output\n", encoding="utf-8")
        (output_dir / "stderr.log").write_text("failure\n", encoding="utf-8")
        return LammpsRunResult(
            command=["lmp"],
            return_code=return_code,
            stdout_path=str(output_dir / "stdout.log"),
            stderr_path=str(output_dir / "stderr.log"),
            timed_out=timed_out,
        )

    manifest = RunArtifactStore(tmp_path).execute(
        "runs/water/inputs/in.lammps",
        command="lmp",
        expected_sha256=digest,
        runner=runner,
    )

    assert manifest.status is expected
    assert (tmp_path / manifest.manifest_path).is_file()
    assert (tmp_path / manifest.stderr_path).read_text(encoding="utf-8") == "failure\n"


def test_execute_rejects_wrong_layout_hash_and_symlink(tmp_path: Path) -> None:
    source, digest = _input(tmp_path)
    store = RunArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="runs/<job>/inputs"):
        store.execute("runs/water/in.lammps", command="lmp", runner=_runner)
    with pytest.raises(ValueError, match="changed"):
        store.execute(
            "runs/water/inputs/in.lammps",
            command="lmp",
            expected_sha256="0" * 64,
            runner=_runner,
        )

    source.unlink()
    outside = tmp_path / "outside.in"
    outside.write_text("run 0\n", encoding="utf-8")
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        store.execute(
            "runs/water/inputs/in.lammps",
            command="lmp",
            expected_sha256=digest,
            runner=_runner,
        )


def test_launch_exception_is_a_traceable_failed_run(tmp_path: Path) -> None:
    _, digest = _input(tmp_path)

    def runner(**kwargs: object) -> LammpsRunResult:
        raise FileNotFoundError("lmp-missing")

    store = RunArtifactStore(tmp_path)
    manifest = store.execute(
        "runs/water/inputs/in.lammps",
        command="lmp-missing",
        expected_sha256=digest,
        runner=runner,
    )

    assert manifest.status is RunStatus.FAILED
    assert manifest.return_code == -1
    assert "lmp-missing" in (tmp_path / manifest.stderr_path).read_text(encoding="utf-8")


def test_inspect_run_rejects_invalid_and_ambiguous_ids(tmp_path: Path) -> None:
    _, digest = _input(tmp_path, "one")
    store = RunArtifactStore(tmp_path)
    manifest = store.execute(
        "runs/one/inputs/in.lammps",
        command="lmp",
        expected_sha256=digest,
        runner=_runner,
    )
    duplicate = tmp_path / "runs/two/executions" / manifest.run_id
    duplicate.mkdir(parents=True)
    payload = manifest.to_dict()
    payload["job"] = "two"
    payload["manifest_path"] = f"runs/two/executions/{manifest.run_id}/manifest.json"
    (duplicate / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid run_id"):
        store.inspect_run("../escape")
    with pytest.raises(ValueError, match="ambiguous run_id"):
        store.inspect_run(manifest.run_id)


def test_inspect_run_rejects_manifest_cross_run_paths(tmp_path: Path) -> None:
    _, digest = _input(tmp_path)
    store = RunArtifactStore(tmp_path)
    manifest = store.execute(
        "runs/water/inputs/in.lammps",
        command="lmp",
        expected_sha256=digest,
        runner=_runner,
    )
    manifest_path = tmp_path / manifest.manifest_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stdout_path"] = "runs/water/inputs/in.lammps"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stdout_path mismatch"):
        store.inspect_run(manifest.run_id)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", 1, "must be a string"),
        ("schema_version", True, "must be an integer"),
        ("lammps_log_path", 1, "string or null"),
        ("return_code", False, "integer or null"),
        ("wall_time", "fast", "must be numeric"),
        ("command", ["lmp", 1], "list of strings"),
        ("output_files", "dump.xyz", "list of strings"),
    ],
)
def test_manifest_rejects_untrusted_json_types(
    field: str, value: object, message: str
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": "20260101T000000.000000Z-aaaaaaaaaaaa",
        "job": "water",
        "status": "succeeded",
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "source_input": "runs/water/inputs/in.lammps",
        "input_sha256": "0" * 64,
        "artifact_dir": "runs/water/executions/run",
        "workspace_dir": "runs/water/executions/run/workspace",
        "stdout_path": "runs/water/executions/run/stdout.log",
        "stderr_path": "runs/water/executions/run/stderr.log",
        "lammps_log_path": None,
        "manifest_path": "runs/water/executions/run/manifest.json",
        "return_code": 0,
        "wall_time": 1.0,
        "command": ["lmp"],
        "output_files": [],
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        RunManifest.from_dict(payload)

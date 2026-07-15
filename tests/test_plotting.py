"""Deterministic thermo and O-O RDF plotting tests."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from lammps_agent.lammps_runner import LammpsRunResult
from lammps_agent.plotting import (
    PlotDataError,
    PlottingDependencyError,
    UnknownPlotTypeError,
    parse_rdf_oo,
    plot_run,
)
from lammps_agent.run_artifacts import RunArtifactStore

RDF = """\
# Time-averaged data for fix rdfout
# TimeStep Number-of-rows
# Row c_rdf[1] c_rdf[2] c_rdf[3]
1000 4
1 0.50 0.00 0.00
2 1.00 0.10 0.02
3 1.50 1.30 0.40
4 2.00 0.95 0.80
2000 4
1 0.50 0.00 0.00
2 1.00 0.15 0.03
3 1.50 1.25 0.45
4 2.00 1.00 0.85
"""


def _run_with_data(tmp_path: Path, *, rdf: str = RDF, thermo: str | None = None) -> str:
    inputs = tmp_path / "runs/water/inputs"
    inputs.mkdir(parents=True)
    source = inputs / "in.lammps"
    source.write_text("run 0\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    thermo_output = thermo or (
        "Step Temp TotEng\n0 298 -10.0\n1000 301 -9.8\n2000 300 -9.9\nLoop time\n"
    )

    def runner(**kwargs: object) -> LammpsRunResult:
        output_dir = Path(kwargs["output_dir"])
        work_dir = Path(kwargs["work_dir"])
        (output_dir / "stdout.log").write_text(thermo_output, encoding="utf-8")
        (output_dir / "stderr.log").write_text("", encoding="utf-8")
        (work_dir / "rdf_oo.dat").write_text(rdf, encoding="utf-8")
        return LammpsRunResult(
            command=["lmp"],
            return_code=0,
            stdout_path=str(output_dir / "stdout.log"),
            stderr_path=str(output_dir / "stderr.log"),
            output_files=["rdf_oo.dat"],
        )

    return RunArtifactStore(tmp_path).execute(
        "runs/water/inputs/in.lammps",
        command="lmp",
        expected_sha256=digest,
        runner=runner,
    ).run_id


def test_parse_rdf_uses_last_complete_block() -> None:
    radius, rdf = parse_rdf_oo(RDF)
    assert radius == [0.5, 1.0, 1.5, 2.0]
    assert rdf == [0.0, 0.15, 1.25, 1.0]


@pytest.mark.parametrize(
    "source",
    ["", "0 1\n1 nan 2\n", "0 2\n1 1.0 0.5\n", "0 2\n1 1.0 0.5\n2 0.5 0.6\n"],
)
def test_parse_rdf_rejects_empty_malformed_or_nonmonotonic_data(source: str) -> None:
    with pytest.raises(PlotDataError):
        parse_rdf_oo(source)


def test_plot_run_rejects_unknown_type_before_loading_matplotlib(tmp_path: Path) -> None:
    with pytest.raises(UnknownPlotTypeError):
        plot_run(tmp_path, "not-a-run", ["structure"])


def test_plot_run_reports_missing_dependency_without_installing(tmp_path: Path) -> None:
    run_id = _run_with_data(tmp_path)
    real_import = importlib.import_module

    def import_module(name: str):
        if name == "matplotlib":
            raise ImportError("missing for test")
        return real_import(name)

    with patch("lammps_agent.plotting.importlib.import_module", side_effect=import_module):
        with pytest.raises(PlottingDependencyError, match="matplotlib is required"):
            plot_run(tmp_path, run_id, ["thermo"])


def test_plot_run_creates_valid_png_metadata_when_matplotlib_is_installed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    run_id = _run_with_data(tmp_path)

    outputs = plot_run(tmp_path, run_id, ["thermo", "rdf_oo"])

    assert [item.plot_type for item in outputs] == ["thermo", "rdf_oo"]
    for item in outputs:
        assert item.source_run_id == run_id
        assert item.width == 1000
        assert item.height == 600
        assert item.bytes > 100
        assert item.path.startswith(f"runs/water/executions/{run_id}/artifacts/plots/")
        assert (tmp_path / item.path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

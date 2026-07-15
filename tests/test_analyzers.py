"""Public analyzer registry contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lammps_agent.analyzers import (
    AnalyzerReport,
    UnknownAnalyzerError,
    build_analyzers,
    run_analyzers,
)


def _run_dir(tmp_path: Path, text: str) -> Path:
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "stdout.log").write_text(text, encoding="utf-8")
    return directory


def test_analyzer_package_is_the_canonical_public_api(successful_run_dir) -> None:
    report = run_analyzers(successful_run_dir, ["thermo_tail"])[0]
    assert isinstance(report, AnalyzerReport)
    assert report.name == "thermo_tail"
    assert build_analyzers(["convergence"])[0].name == "convergence"


def test_unknown_analyzer_is_explicit_and_structured(successful_run_dir) -> None:
    with pytest.raises(UnknownAnalyzerError) as raised:
        run_analyzers(successful_run_dir, ["not-registered"])

    assert raised.value.as_dict() == {
        "code": "UNKNOWN_ANALYZER",
        "message": "Unknown analyzer: not-registered",
        "analyzer": "not-registered",
    }


def test_stress_strain_numeric_result_and_units(tmp_path: Path) -> None:
    rows = "\n".join(
        f"{i} {100.0 * (1.0 + 0.002 * i):.6f} {-4000.0 * i:.6f}"
        for i in range(8)
    )
    run_dir = _run_dir(tmp_path, f"units metal\nStep Lz Pzz\n{rows}\nLoop time\n")

    report = run_analyzers(run_dir, ["stress_strain"])[0]

    assert report.details["youngs_modulus_GPa"] == pytest.approx(200.0, abs=0.01)
    assert report.details["ultimate_tensile_strength_GPa"] == pytest.approx(2.8)

    (run_dir / "stdout.log").write_text(
        f"units real\nStep Lz Pzz\n{rows}\nLoop time\n", encoding="utf-8"
    )
    unsupported = run_analyzers(run_dir, ["stress_strain"])[0]
    assert unsupported.details["status"] == "unsupported_units"


def test_mechanical_units_are_read_from_immutable_input_snapshot(tmp_path: Path) -> None:
    rows = "\n".join(f"{i} {100 + i} {-1000 * i}" for i in range(6))
    run_dir = _run_dir(tmp_path, f"Step Lz Pzz\n{rows}\nLoop time\n")
    workspace = run_dir / "workspace"
    workspace.mkdir()
    (workspace / "in.lammps").write_text("units metal\nrun 0\n", encoding="utf-8")

    report = run_analyzers(run_dir, ["stress_strain"])[0]

    assert report.details.get("status") != "unsupported_units"


def test_mechanical_analyzers_reject_nan_and_insufficient_data(tmp_path: Path) -> None:
    run_dir = _run_dir(
        tmp_path,
        "units metal\nStep Lz Pzz\n0 10 0\n1 10.1 nan\n2 10.2 2\n"
        "3 10.3 3\n4 10.4 4\nLoop time\n",
    )
    assert run_analyzers(run_dir, ["stress_strain"])[0].details["status"] == "invalid_data"

    (run_dir / "stdout.log").write_text(
        "units metal\nStep Pyy Ly\n0 0 10\n1 1 10.1\nLoop time\n",
        encoding="utf-8",
    )
    crack = run_analyzers(run_dir, ["crack_analysis"])[0]
    assert "Insufficient" in crack.summary


def test_indent_requires_explicit_displacement_and_rejects_nan(tmp_path: Path) -> None:
    force_only = "\n".join(f"{i} {float(i)}" for i in range(8))
    run_dir = _run_dir(
        tmp_path,
        f"units metal\nStep f_indent[3]\n{force_only}\nLoop time\n",
    )
    report = run_analyzers(run_dir, ["indent_analysis"])[0]
    assert report.details["status"] == "insufficient_data"

    rows = "\n".join(
        f"{i} {('nan' if i == 3 else float(i))} {10.0 - 0.1 * i}" for i in range(8)
    )
    (run_dir / "stdout.log").write_text(
        f"units metal\nStep f_indent[3] v_indenter_z\n{rows}\nLoop time\n",
        encoding="utf-8",
    )
    invalid = run_analyzers(run_dir, ["indent_analysis"])[0]
    assert invalid.details["status"] == "invalid_data"


def test_indent_and_crack_numeric_results(tmp_path: Path) -> None:
    rows = "\n".join(f"{i} {float(i)} {10.0 - 0.1 * i}" for i in range(8))
    run_dir = _run_dir(
        tmp_path,
        f"units metal\nStep f_indent[3] v_indenter_z\n{rows}\nLoop time\n",
    )
    indent = run_analyzers(run_dir, ["indent_analysis"])[0]
    assert indent.details["P_max_nN"] == pytest.approx(11.214, abs=1e-4)
    assert indent.details["h_max_Angstrom"] == pytest.approx(0.7)

    crack_rows = "\n".join(f"{i} {-10000.0 * i} {100.0 + i}" for i in range(6))
    (run_dir / "stdout.log").write_text(
        f"units metal\nStep Pyy Ly\n{crack_rows}\nLoop time\n",
        encoding="utf-8",
    )
    crack = run_analyzers(run_dir, ["crack_analysis"])[0]
    assert crack.details["max_stress_GPa"] == pytest.approx(5.0)
    assert crack.details["max_stress_step"] == 5
    assert crack.details["fracture_strain"] == pytest.approx(0.05)

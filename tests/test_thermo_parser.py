"""Tests for the thermo output parser."""

from __future__ import annotations

from pathlib import Path

from src.lammps_agent.thermo_parser import parse_thermo_table

from .conftest import SAMPLE_THERMO_OUTPUT, SAMPLE_THERMO_ZERO_TEMP, SAMPLE_THERMO_FAILED


def test_parse_normal_thermo():
    result = parse_thermo_table(SAMPLE_THERMO_OUTPUT)
    assert "Step" in result
    assert "Temp" in result
    assert "TotEng" in result
    assert len(result["Step"]) == 11
    assert result["Step"][0] == 0.0
    assert result["Step"][-1] == 1000.0
    assert result["Temp"][0] == 300.0


def test_parse_zero_temp_thermo():
    result = parse_thermo_table(SAMPLE_THERMO_ZERO_TEMP)
    assert "Step" in result
    assert len(result["Step"]) == 6
    # Temperature should be essentially zero.
    assert all(t < 1e-10 for t in result["Temp"])


def test_parse_failed_run_no_thermo():
    result = parse_thermo_table(SAMPLE_THERMO_FAILED)
    # No thermo table in a failed run.
    assert len(result.get("Step", [])) == 0


def test_parse_from_file(successful_run_dir: Path):
    result = parse_thermo_table(successful_run_dir / "stdout.log")
    assert "Step" in result
    assert len(result["Step"]) == 11

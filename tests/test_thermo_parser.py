"""Tests for the thermo output parser."""

from __future__ import annotations

import math
from pathlib import Path

from lammps_agent.thermo_parser import (
    has_sustained_linear_drift,
    parse_thermo_table,
)

from .conftest import SAMPLE_THERMO_FAILED, SAMPLE_THERMO_OUTPUT, SAMPLE_THERMO_ZERO_TEMP


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


def test_parse_extended_lammps_float_formats():
    output = """\
 Step Temp PotEng TotEng
 0 .5 1. nan
 1 +inf -Infinity -2.5e+1
Loop time of 0.1 on 1 procs
"""

    result = parse_thermo_table(output)

    assert result["Step"] == [0.0, 1.0]
    assert result["Temp"][0] == 0.5
    assert math.isinf(result["Temp"][1]) and result["Temp"][1] > 0
    assert result["PotEng"][0] == 1.0
    assert math.isinf(result["PotEng"][1]) and result["PotEng"][1] < 0
    assert math.isnan(result["TotEng"][0])
    assert result["TotEng"][1] == -25.0


def test_sustained_drift_detector_distinguishes_ramp_from_noise():
    assert has_sustained_linear_drift([100.0 - 0.2 * index for index in range(20)])
    assert not has_sustained_linear_drift(
        [100.0, 100.1, 99.9, 100.05, 99.95] * 4
    )
    assert not has_sustained_linear_drift([100.0] * 20)

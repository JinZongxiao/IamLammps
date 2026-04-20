"""Shared fixtures for IamLAMMPS tests."""

from __future__ import annotations

import pytest
from pathlib import Path

# Sample LAMMPS thermo output for a successful LJ NVT run.
SAMPLE_THERMO_OUTPUT = """\
LAMMPS (22 Jul 2025 - Update 3)
Lattice spacing in x,y,z = 1.6795962 1.6795962 1.6795962
Created orthogonal box = (0 0 0) to (16.795962 16.795962 16.795962)
  1 by 1 by 1 MPI processor grid
Created 4000 atoms
Setting up Verlet run ...
Per MPI rank memory allocation (min/avg/max) = 3.722 | 3.722 | 3.722 Mbytes
   Step          Temp          PotEng         KinEng         TotEng         Press          Volume
         0   300            -6.7733681      0.449          -6.3243681     -5.0            4738.2137
       100   298.5          -6.7600000      0.447          -6.3130000     -4.8            4738.2137
       200   301.2          -6.7500000      0.451          -6.2990000     -4.9            4738.2137
       300   299.8          -6.7450000      0.449          -6.2960000     -4.85           4738.2137
       400   300.1          -6.7440000      0.450          -6.2940000     -4.82           4738.2137
       500   300.0          -6.7435000      0.450          -6.2935000     -4.81           4738.2137
       600   299.9          -6.7433000      0.450          -6.2933000     -4.80           4738.2137
       700   300.1          -6.7432000      0.450          -6.2932000     -4.80           4738.2137
       800   300.0          -6.7431000      0.450          -6.2931000     -4.80           4738.2137
       900   300.0          -6.7431000      0.450          -6.2931000     -4.80           4738.2137
      1000   300.0          -6.7431000      0.450          -6.2931000     -4.80           4738.2137
Loop time of 7.15653 on 1 procs for 10000 steps with 4000 atoms
Total wall time: 0:00:07
"""

SAMPLE_THERMO_ZERO_TEMP = """\
LAMMPS (22 Jul 2025 - Update 3)
Setting up Verlet run ...
   Step          Temp          PotEng         KinEng         TotEng         Press          Volume
         0   0             -6.7733681      0             -6.7733681     -6.2353173      4738.2137
       100   1.79e-28      -6.7733681      2.69e-28      -6.7733681     -6.2353173      4738.2137
       200   6.82e-28      -6.7733681      1.02e-27      -6.7733681     -6.2353173      4738.2137
       300   1.58e-27      -6.7733681      2.36e-27      -6.7733681     -6.2353173      4738.2137
       400   2.67e-27      -6.7733681      4.01e-27      -6.7733681     -6.2353173      4738.2137
       500   4.20e-27      -6.7733681      6.31e-27      -6.7733681     -6.2353173      4738.2137
Loop time of 1.0 on 1 procs
Total wall time: 0:00:01
"""

SAMPLE_THERMO_FAILED = """\
LAMMPS (22 Jul 2025 - Update 3)
ERROR: Cannot create atoms with undefined lattice (src/create_atoms.cpp:411)
"""


@pytest.fixture
def successful_run_dir(tmp_path: Path) -> Path:
    """Create a mock run directory for a successful simulation."""
    run_dir = tmp_path / "run_success"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(SAMPLE_THERMO_OUTPUT, encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "in.lammps").write_text("# mock script", encoding="utf-8")
    return run_dir


@pytest.fixture
def zero_temp_run_dir(tmp_path: Path) -> Path:
    """Create a mock run directory with T=0 throughout (missing velocity)."""
    run_dir = tmp_path / "run_zero_temp"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(SAMPLE_THERMO_ZERO_TEMP, encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    return run_dir


@pytest.fixture
def failed_run_dir(tmp_path: Path) -> Path:
    """Create a mock run directory for a failed simulation."""
    run_dir = tmp_path / "run_failed"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(SAMPLE_THERMO_FAILED, encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    return run_dir

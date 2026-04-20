"""End-to-end mechanical property tests for pure W.

Bypasses LLM by providing known-correct LAMMPS scripts for each pipeline stage.
Tests the full agent flow: potential copy, restart file passing, LAMMPS execution,
skill analysis, and reward computation.

Usage:
    python tests/test_e2e_mechanics.py [tensile|nanoindent|crack|all]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lammps_agent.config import load_config
from src.lammps_agent.agent import LammpsAgent

# ---------------------------------------------------------------------------
# Tensile workflow scripts (3 stages): minimize → equilibrate → deform
# Using small 3x3x6 supercell for speed
# ---------------------------------------------------------------------------

TENSILE_MINIMIZE = """\
units metal
atom_style atomic
boundary p p p

lattice bcc 3.165
region box block 0 3 0 3 0 6
create_box 1 box
create_atoms 1 box
mass 1 183.84

pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W

thermo 100
thermo_style custom step pe etotal press vol lx ly lz
minimize 1.0e-10 1.0e-10 10000 100000
write_restart restart.minimize
"""

TENSILE_EQUILIBRATE = """\
units metal
atom_style atomic
boundary p p p

read_restart restart.minimize
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

velocity all create 300.0 12345 dist gaussian
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
timestep 0.001

thermo 100
thermo_style custom step temp press pe ke etotal vol lx ly lz
run 2000
write_restart restart.equilibrate
"""

TENSILE_DEFORM = """\
units metal
atom_style atomic
boundary p p p

read_restart restart.equilibrate
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

# Uniaxial tensile along z
fix 1 all npt temp 300.0 300.0 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0
fix 2 all deform 1 z erate 0.0001 remap x

timestep 0.001
thermo 50
thermo_style custom step temp pxx pyy pzz pxy pxz pyz lx ly lz pe ke etotal
run 3000
"""

# ---------------------------------------------------------------------------
# Nanoindentation workflow scripts (3 stages): minimize → equilibrate → indent
# Using small 6x6x10 supercell
# ---------------------------------------------------------------------------

INDENT_MINIMIZE = """\
units metal
atom_style atomic
boundary p p s

lattice bcc 3.165
region box block 0 6 0 6 0 10
create_box 1 box
create_atoms 1 box
mass 1 183.84

pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W

# Define layers: bottom 15% fixed, next 15% thermostat, rest mobile
variable z_lo equal lz*0.0
variable z_fixed equal lz*0.15
variable z_thermo equal lz*0.30

region fixed_region block INF INF INF INF INF ${z_fixed}
region thermo_region block INF INF INF INF ${z_fixed} ${z_thermo}
group fixed_atoms region fixed_region
group thermo_atoms region thermo_region
group mobile_atoms subtract all fixed_atoms

thermo 100
thermo_style custom step pe etotal press vol
minimize 1.0e-8 1.0e-8 10000 100000
write_restart restart.minimize
"""

INDENT_EQUILIBRATE = """\
units metal
atom_style atomic
boundary p p s

read_restart restart.minimize
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

# Recreate groups after restart
variable z_fixed equal lz*0.15
variable z_thermo equal lz*0.30
region fixed_region block INF INF INF INF INF ${z_fixed}
region thermo_region block INF INF INF INF ${z_fixed} ${z_thermo}
group fixed_atoms region fixed_region
group thermo_atoms region thermo_region
group mobile_atoms subtract all fixed_atoms

fix freeze fixed_atoms setforce 0.0 0.0 0.0
velocity mobile_atoms create 300.0 54321 dist gaussian
fix 1 mobile_atoms nvt temp 300.0 300.0 0.1
timestep 0.001

thermo 100
thermo_style custom step temp pe ke etotal press
run 2000
write_restart restart.equilibrate
"""

INDENT_INDENT = """\
units metal
atom_style atomic
boundary p p s

read_restart restart.equilibrate
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

# Recreate groups
variable z_fixed equal lz*0.15
variable z_thermo equal lz*0.30
region fixed_region block INF INF INF INF INF ${z_fixed}
region thermo_region block INF INF INF INF ${z_fixed} ${z_thermo}
group fixed_atoms region fixed_region
group thermo_atoms region thermo_region
group mobile_atoms subtract all fixed_atoms

fix freeze fixed_atoms setforce 0.0 0.0 0.0

# Spherical indenter from above
variable R equal 15.0
variable K equal 10.0
variable x_center equal lx/2
variable y_center equal ly/2
variable z_top equal lz+20.0
variable indent_vel equal 0.001
variable indenter_z equal v_z_top-v_indent_vel*step*dt

fix nve_mobile mobile_atoms nve
fix thermostat thermo_atoms langevin 300.0 300.0 0.1 99999
fix indent_fix mobile_atoms indent ${K} sphere v_x_center v_y_center v_indenter_z ${R}

timestep 0.001
thermo 100
thermo_style custom step temp f_indent_fix[1] f_indent_fix[2] f_indent_fix[3] v_indenter_z pe
run 2000
"""

# ---------------------------------------------------------------------------
# Crack workflow scripts (4 stages): build_crack → minimize → load → analyze
# Using small 10x6x2 supercell
# ---------------------------------------------------------------------------

CRACK_BUILD = """\
units metal
atom_style atomic
boundary p p p

lattice bcc 3.165
region box block 0 10 0 6 0 2
create_box 1 box
create_atoms 1 box
mass 1 183.84

pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W

# Create a pre-crack notch
variable y_mid equal ly/2
variable half_w equal 1.0
variable crack_len equal lx*0.3
region crack block 0 ${crack_len} $(v_y_mid-v_half_w) $(v_y_mid+v_half_w) INF INF
delete_atoms region crack

# Define boundary groups
variable y_upper equal ly*0.9
variable y_lower equal ly*0.1
region upper_region block INF INF ${y_upper} INF INF INF
region lower_region block INF INF INF ${y_lower} INF INF
group upper_atoms region upper_region
group lower_atoms region lower_region
group mobile_atoms subtract all upper_atoms lower_atoms

thermo 100
thermo_style custom step pe etotal press
minimize 1.0e-6 1.0e-6 5000 50000
write_restart restart.cracked
"""

CRACK_MINIMIZE = """\
units metal
atom_style atomic
boundary p p p

read_restart restart.cracked
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

thermo 100
thermo_style custom step pe etotal press
minimize 1.0e-8 1.0e-8 20000 200000
write_restart restart.minimized
"""

CRACK_LOAD = """\
units metal
atom_style atomic
boundary p p p

read_restart restart.minimized
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

# Recreate groups
variable y_upper equal ly*0.9
variable y_lower equal ly*0.1
region upper_region block INF INF ${y_upper} INF INF INF
region lower_region block INF INF INF ${y_lower} INF INF
group upper_atoms region upper_region
group lower_atoms region lower_region
group mobile_atoms subtract all upper_atoms lower_atoms

velocity mobile_atoms create 300.0 67890 dist gaussian

# Mode I loading: pull top/bottom apart
fix pull_up upper_atoms aveforce 0.0 0.001 0.0
fix pull_down lower_atoms aveforce 0.0 -0.001 0.0
fix nve_mobile mobile_atoms nve
timestep 0.001

thermo 100
thermo_style custom step temp pyy ly pe ke etotal
run 3000
write_restart restart.loaded
"""

CRACK_ANALYZE = """\
units metal
atom_style atomic
boundary p p p

read_restart restart.loaded
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

compute coord all coord/atom cutoff 3.5

dump 1 all custom 1 dump.crack_final.lammpstrj id type x y z c_coord
run 0
"""

# ---------------------------------------------------------------------------
# Script registry: maps (workflow, stage_index) to the correct script
# ---------------------------------------------------------------------------

TENSILE_SCRIPTS = [TENSILE_MINIMIZE, TENSILE_EQUILIBRATE, TENSILE_DEFORM]
INDENT_SCRIPTS = [INDENT_MINIMIZE, INDENT_EQUILIBRATE, INDENT_INDENT]
CRACK_SCRIPTS = [CRACK_BUILD, CRACK_MINIMIZE, CRACK_LOAD, CRACK_ANALYZE]


def make_stage_mocker(scripts: list[str]):
    """Return a side_effect function that returns scripts[call_count]."""
    call_count = [0]

    def mock_generate(task: str, error_context: str = "") -> str:
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(scripts):
            return scripts[idx]
        # Fallback for retries: return the last script again
        return scripts[-1]

    return mock_generate


def run_workflow_test(name: str, scripts: list[str], skill_names: list[str]) -> bool:
    """Run a multi-stage workflow with known-correct scripts."""
    print(f"\n{'='*60}")
    print(f"  WORKFLOW TEST: {name}")
    print(f"{'='*60}")

    config = load_config()
    config.enabled_skills = skill_names
    config.max_retries = 0  # No retries — scripts should work first time

    agent = LammpsAgent(config)
    mocker = make_stage_mocker(scripts)

    t0 = time.time()
    with patch.object(agent, "generate_input_script", side_effect=mocker):
        from src.lammps_agent.workflows import build_workflow, WorkflowConfig

        wf_config = WorkflowConfig(
            temperature=300.0,
            potential_name="W_Zhou2004_EAM",
        )

        # Use smaller supercells for speed
        if name == "tensile":
            workflow = build_workflow(name, wf_config, agent, supercell=(3, 3, 6), strain_rate=1e8)
        elif name == "nanoindent":
            workflow = build_workflow(name, wf_config, agent, supercell=(6, 6, 10))
        elif name == "crack":
            workflow = build_workflow(name, wf_config, agent, supercell=(10, 6, 2))
        else:
            workflow = build_workflow(name, wf_config, agent)

        # For pipeline-based workflows, replace the agent's generate_input_script
        # The pipeline creates run_dirs and calls agent.run() for each stage.
        # Our mock is already patched on the agent instance.
        result = workflow.run()

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Success: {result.success}")
    if result.failed_stage:
        print(f"Failed stage: {result.failed_stage}")

    for stage_name, output in result.stages:
        rc = output.lammps_result.return_code if output.lammps_result else None
        wt = output.lammps_result.wall_time if output.lammps_result else 0
        retries = output.retry_count

        pot_file = output.run_dir / "W_zhou.eam.alloy"
        pot_ok = "YES" if pot_file.exists() else "NO"

        print(f"  Stage '{stage_name}': rc={rc}, time={wt}s, retries={retries}, potential={pot_ok}")

        if rc != 0 and output.lammps_result:
            stderr_path = Path(output.lammps_result.stderr_path)
            stdout_path = Path(output.lammps_result.stdout_path)
            if stderr_path.exists():
                err = stderr_path.read_text().strip()
                if err:
                    print(f"    STDERR: {err[:200]}")
            if stdout_path.exists():
                lines = stdout_path.read_text().strip().splitlines()
                for line in lines[-5:]:
                    print(f"    STDOUT: {line}")

        # Show skill reports
        for report in output.skill_reports:
            print(f"    [{report.name}] {report.summary}")

    # Check restart file passing between stages
    if len(result.stages) > 1:
        for i in range(1, len(result.stages)):
            stage_name, output = result.stages[i]
            run_dir = output.run_dir
            # Check if restart files from previous stage exist in this run_dir
            restart_files = list(run_dir.glob("restart.*"))
            if restart_files:
                names = [f.name for f in restart_files]
                print(f"  Stage '{stage_name}' inherited: {names}")

    if result.success:
        print(f"\n  *** {name.upper()} WORKFLOW: PASSED ***")
    else:
        print(f"\n  *** {name.upper()} WORKFLOW: FAILED ***")

    return result.success


# ---------------------------------------------------------------------------
# DBTT sweep test: 2 temperatures × 3 stages = 6 LAMMPS runs
# ---------------------------------------------------------------------------

def _tensile_minimize_script() -> str:
    """Same minimize for all temperatures (no T dependence)."""
    return TENSILE_MINIMIZE


def _tensile_equilibrate_script(T: float) -> str:
    return f"""\
units metal
atom_style atomic
boundary p p p

read_restart restart.minimize
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

velocity all create {T} 12345 dist gaussian
fix 1 all npt temp {T} {T} 0.1 iso 0.0 0.0 1.0
timestep 0.001

thermo 100
thermo_style custom step temp press pe ke etotal vol lx ly lz
run 2000
write_restart restart.equilibrate
"""


def _tensile_deform_script(T: float) -> str:
    return f"""\
units metal
atom_style atomic
boundary p p p

read_restart restart.equilibrate
pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W
mass 1 183.84

fix 1 all npt temp {T} {T} 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0
fix 2 all deform 1 z erate 0.0001 remap x

timestep 0.001
thermo 50
thermo_style custom step temp pxx pyy pzz pxy pxz pyz lx ly lz pe ke etotal
run 3000
"""


def make_dbtt_mocker(temperatures: list[float]):
    """Return a mock that serves the right script for each (T, stage) pair.

    DBTT calls TensileWorkflow.run() for each T in order.
    Each TensileWorkflow has 3 stages: minimize, equilibrate, deform.
    So calls arrive as: T1-min, T1-eq, T1-def, T2-min, T2-eq, T2-def, ...
    """
    scripts: list[str] = []
    for T in temperatures:
        scripts.append(_tensile_minimize_script())
        scripts.append(_tensile_equilibrate_script(T))
        scripts.append(_tensile_deform_script(T))

    call_count = [0]

    def mock_generate(task: str, error_context: str = "") -> str:
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(scripts):
            return scripts[idx]
        return scripts[-1]

    return mock_generate


def run_dbtt_test() -> bool:
    """Run DBTT sweep with 2 temperatures (300K, 600K) as smoke test."""
    print(f"\n{'='*60}")
    print(f"  WORKFLOW TEST: DBTT sweep (smoke test, 2 temperatures)")
    print(f"{'='*60}")

    temperatures = [300.0, 600.0]

    config = load_config()
    config.enabled_skills = ["thermo_tail", "convergence", "thermo_validator", "stress_strain"]
    config.max_retries = 0

    agent = LammpsAgent(config)
    mocker = make_dbtt_mocker(temperatures)

    t0 = time.time()
    with patch.object(agent, "generate_input_script", side_effect=mocker):
        from src.lammps_agent.workflows.dbtt_sweep import DBTTSweepWorkflow
        from src.lammps_agent.workflows.base import WorkflowConfig

        wf_config = WorkflowConfig(
            temperature=300.0,  # base, overridden per-T
            potential_name="W_Zhou2004_EAM",
        )
        workflow = DBTTSweepWorkflow(
            config=wf_config,
            agent=agent,
            temperatures=temperatures,
            supercell=(3, 3, 6),
            strain_rate=1e8,
        )
        result = workflow.run()

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Success: {result.success}")

    for stage_name, output in result.stages:
        rc = output.lammps_result.return_code if output.lammps_result else None
        print(f"  Stage '{stage_name}': rc={rc}")
        # Show stress_strain skill for deform stages
        for report in output.skill_reports:
            if report.name == "stress_strain" and "Insufficient" not in report.summary:
                print(f"    [{report.name}] {report.summary}")

    if result.success:
        print(f"\n  *** DBTT SWEEP: PASSED ***")
    else:
        print(f"\n  *** DBTT SWEEP: FAILED ***")

    return result.success


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}

    if target in ("tensile", "all"):
        results["tensile"] = run_workflow_test(
            "tensile", TENSILE_SCRIPTS,
            ["thermo_tail", "convergence", "thermo_validator", "stress_strain"],
        )

    if target in ("nanoindent", "all"):
        results["nanoindent"] = run_workflow_test(
            "nanoindent", INDENT_SCRIPTS,
            ["thermo_tail", "convergence", "thermo_validator", "indent_analysis"],
        )

    if target in ("crack", "all"):
        results["crack"] = run_workflow_test(
            "crack", CRACK_SCRIPTS,
            ["thermo_tail", "convergence", "thermo_validator", "crack_analysis"],
        )

    if target in ("dbtt", "all"):
        results["dbtt"] = run_dbtt_test()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:15s} {status}")
        if not passed:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end test: W BCC minimization without LLM.

Bypasses the LLM call by directly providing a known-correct LAMMPS script,
then exercises the full agent execution pipeline:
  - potential file copied to run_dir
  - LAMMPS execution
  - post-processing skills
  - reward computation
  - memory recording
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lammps_agent.config import load_config
from src.lammps_agent.agent import LammpsAgent


W_MINIMIZE_SCRIPT = """\
units metal
atom_style atomic
boundary p p p

lattice bcc 3.165
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 183.84

pair_style eam/alloy
pair_coeff * * W_zhou.eam.alloy W

thermo 1
thermo_style custom step pe etotal press vol
minimize 1.0e-4 1.0e-6 100 1000
"""


def main() -> int:
    config = load_config()
    # Force default skills for testing.
    config.enabled_skills = ["thermo_tail", "convergence", "thermo_validator"]

    agent = LammpsAgent(config)

    # Patch generate_input_script to return our known-correct script
    # instead of calling the LLM.
    with patch.object(agent, "generate_input_script", return_value=W_MINIMIZE_SCRIPT):
        output = agent.run(
            task="BCC W 5x5x5 minimization with EAM (test)",
            execute=True,
        )

    print(f"Run directory: {output.run_dir}")
    print(f"Input file:    {output.input_path}")

    # Check potential file was copied.
    pot_file = output.run_dir / "W_zhou.eam.alloy"
    if pot_file.exists():
        print(f"Potential file: PRESENT ({pot_file.stat().st_size} bytes)")
    else:
        print("Potential file: MISSING  <-- BUG!")
        return 1

    if output.lammps_result is None:
        print("LAMMPS was not executed!")
        return 1

    rc = output.lammps_result.return_code
    print(f"Return code:   {rc}")
    print(f"Wall time:     {output.lammps_result.wall_time}s")
    print(f"Retry count:   {output.retry_count}")

    if rc != 0:
        stderr_path = Path(output.lammps_result.stderr_path)
        if stderr_path.exists():
            print(f"STDERR:\n{stderr_path.read_text()}")
        stdout_path = Path(output.lammps_result.stdout_path)
        if stdout_path.exists():
            lines = stdout_path.read_text().strip().splitlines()
            print(f"STDOUT (last 10 lines):")
            for line in lines[-10:]:
                print(f"  {line}")
        return 1

    print(f"\nSkill reports ({len(output.skill_reports)}):")
    for report in output.skill_reports:
        print(f"  [{report.name}] {report.summary}")

    print("\nSUCCESS: W BCC minimization end-to-end test passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

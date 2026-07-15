---
name: lammps-debugging
description: Diagnose and repair a failing LAMMPS input through an inspect-edit-run loop
recommended-tools: [list_runs, inspect_run, read_file, write_file, run_lammps_input, parse_thermo]
---
Read the immutable run record and identify the first actionable LAMMPS error before
changing an input. Make the smallest coherent edit while preserving scientific intent.
Treat these tools as recommendations only; normal tool policy and approval still apply.

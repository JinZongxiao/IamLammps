"""LAMMPS error parser — diagnose common errors and suggest fixes for the retry loop."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LammpsError:
    """Structured representation of a LAMMPS runtime error."""

    error_type: str  # machine-readable tag, e.g. "missing_lattice"
    raw_message: str  # the original ERROR line from LAMMPS output
    source_file: str  # e.g. "create_atoms.cpp:411"
    suggested_fix: str  # human-readable hint injected into the LLM retry prompt


# Each entry: (regex applied to the full stdout/log text, error_type, suggested_fix template).
# The regex should capture the raw ERROR line in group 0.
# Use {match} in suggested_fix to interpolate the raw message when helpful.
_ERROR_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # --- Atom creation / lattice ---
    (
        re.compile(r"ERROR:\s*Cannot create atoms with undefined lattice.*", re.IGNORECASE),
        "missing_lattice",
        "Add a 'lattice' command (e.g. 'lattice bcc 3.165' for W or 'lattice fcc 0.8442' for LJ) "
        "BEFORE 'create_atoms'. The lattice command defines where atoms are placed.",
    ),
    (
        re.compile(r"ERROR:\s*Create_atoms command before simulation box is defined.*", re.IGNORECASE),
        "create_atoms_before_box",
        "Move 'create_atoms' AFTER 'create_box'. You must define the simulation box first.",
    ),
    # --- Pair / force field ---
    (
        re.compile(r"ERROR:\s*Pair_coeff command before simulation box is defined.*", re.IGNORECASE),
        "pair_before_box",
        "Move 'pair_style' and 'pair_coeff' AFTER 'create_box'. "
        "The correct order is: region → create_box → create_atoms → mass → pair_style → pair_coeff.",
    ),
    (
        re.compile(r"ERROR:\s*All pair coeffs are not set.*", re.IGNORECASE),
        "missing_pair_coeff",
        "Not all atom type pairs have pair_coeff defined. "
        "Ensure pair_coeff is set for every type pair (e.g. 'pair_coeff 1 1 ...' for a single-type system).",
    ),
    (
        re.compile(r"ERROR:\s*Cannot open (.*) potential file\s+(\S+).*", re.IGNORECASE),
        "missing_potential_file",
        "The potential file specified in pair_coeff does not exist at the given path. "
        "Check that the file path is correct and the file is present in the working directory.",
    ),
    (
        re.compile(r"ERROR:\s*Unknown pair style\s+(\S+).*", re.IGNORECASE),
        "unknown_pair_style",
        "The pair_style '{match_group_1}' is not recognized. "
        "Check spelling or ensure LAMMPS was compiled with the required package.",
    ),
    (
        re.compile(r"ERROR:\s*Incorrect args for pair coefficients.*", re.IGNORECASE),
        "bad_pair_coeff_args",
        "The arguments to pair_coeff are wrong for the chosen pair_style. "
        "Check the LAMMPS documentation for the correct number and format of arguments.",
    ),
    # --- Mass ---
    (
        re.compile(r"ERROR:\s*All masses are not set.*", re.IGNORECASE),
        "missing_mass",
        "Mass is not set for all atom types. Add 'mass <type> <value>' for every atom type "
        "BEFORE 'run' or 'minimize'.",
    ),
    # --- Fix / integration ---
    (
        re.compile(r"ERROR:\s*Illegal fix\s+(\S+)\s+command.*", re.IGNORECASE),
        "illegal_fix",
        "The fix command has incorrect syntax. Check the LAMMPS documentation for the "
        "correct arguments for this fix style.",
    ),
    (
        re.compile(r"ERROR:\s*Unknown fix style\s+(\S+).*", re.IGNORECASE),
        "unknown_fix_style",
        "The fix style is not recognized. Check spelling or ensure LAMMPS was compiled "
        "with the required package.",
    ),
    (
        re.compile(r"ERROR:\s*Temperature compute degrees of freedom < 0.*", re.IGNORECASE),
        "zero_dof",
        "No degrees of freedom for temperature computation — likely no atoms in the system "
        "or all atoms are frozen. Check create_atoms and group definitions.",
    ),
    # --- Region / box ---
    (
        re.compile(r"ERROR:\s*Region .* has been previously defined.*", re.IGNORECASE),
        "duplicate_region",
        "A region with this name already exists. Use a different name or remove the duplicate definition.",
    ),
    (
        re.compile(r"ERROR:\s*Simulation box is not yet defined.*", re.IGNORECASE),
        "no_simulation_box",
        "The simulation box has not been created. Add 'region' + 'create_box' commands "
        "before any command that requires a box.",
    ),
    # --- Input parsing ---
    (
        re.compile(r"ERROR:\s*Unknown command:\s*(.+)", re.IGNORECASE),
        "unknown_command",
        "LAMMPS does not recognize this command. Check for typos. "
        "The problematic command is: '{match_group_1}'.",
    ),
    (
        re.compile(r"ERROR:\s*Illegal\s+(\S+)\s+command.*", re.IGNORECASE),
        "illegal_command",
        "The '{match_group_1}' command has wrong syntax or arguments. "
        "Check the LAMMPS documentation for correct usage.",
    ),
    (
        re.compile(r"ERROR:\s*Expected integer parameter.*", re.IGNORECASE),
        "bad_integer_param",
        "A command expected an integer argument but received something else. "
        "Check the arguments of the last command before this error.",
    ),
    # --- Minimization ---
    (
        re.compile(r"ERROR:\s*Cannot use minimize with no atoms.*", re.IGNORECASE),
        "minimize_no_atoms",
        "There are no atoms in the system. Ensure create_atoms successfully placed atoms "
        "before calling minimize.",
    ),
    # --- Dump / output ---
    (
        re.compile(r"ERROR:\s*Cannot open dump file.*", re.IGNORECASE),
        "cannot_open_dump",
        "Cannot open the dump file for writing. Check the file path and directory permissions.",
    ),
    # --- Catch-all for any ERROR line ---
    (
        re.compile(r"(ERROR:\s*.+)"),
        "unknown_error",
        "LAMMPS reported an error: {match}. Review the script for syntax or logical issues.",
    ),
]


def parse_lammps_error(stdout: str, stderr: str) -> LammpsError | None:
    """Parse LAMMPS stdout (which contains log output) and stderr for the first error.

    Returns a structured ``LammpsError`` with a diagnosis and suggested fix,
    or ``None`` if no error was detected.
    """
    # LAMMPS writes most output (including ERROR lines) to stdout.
    # Merge both streams so we never miss an error.
    combined = f"{stdout}\n{stderr}"

    for pattern, error_type, fix_template in _ERROR_PATTERNS:
        match = pattern.search(combined)
        if match is None:
            continue

        raw_message = match.group(0).strip()

        # Extract LAMMPS source file reference if present (e.g. "(src/create_atoms.cpp:411)")
        source_match = re.search(r"\(src/(\S+)\)", raw_message)
        source_file = source_match.group(1) if source_match else ""

        # Build suggested fix with optional interpolations.
        suggested_fix = fix_template.replace("{match}", raw_message)
        for i in range(1, 4):
            placeholder = f"{{match_group_{i}}}"
            if placeholder in suggested_fix:
                try:
                    suggested_fix = suggested_fix.replace(placeholder, match.group(i))
                except (IndexError, TypeError):
                    suggested_fix = suggested_fix.replace(placeholder, "???")

        return LammpsError(
            error_type=error_type,
            raw_message=raw_message,
            source_file=source_file,
            suggested_fix=suggested_fix,
        )

    return None

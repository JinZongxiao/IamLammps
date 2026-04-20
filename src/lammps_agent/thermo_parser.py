"""Shared LAMMPS thermo output parser.

Extracts the thermo table from LAMMPS stdout/log into a dict of named columns.
Used by reward.py, convergence skill, and thermo_validator skill.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse_thermo_table(source: str | Path) -> dict[str, list[float]]:
    """Parse LAMMPS thermo output into ``{column_name: [values]}``.

    *source* can be a file path or a raw text string.  The parser looks for
    a header line that starts with ``Step`` and is followed by numeric data
    rows.  Multiple thermo blocks (e.g. from restart) are merged.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8", errors="ignore")
    else:
        text = source

    columns: dict[str, list[float]] = {}
    header: list[str] = []
    in_table = False

    # Regex for a line that looks like a thermo data row:
    # starts with optional whitespace then digits, followed by more numeric fields.
    data_re = re.compile(
        r"^\s*\d+(?:\s+[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)+\s*$"
    )

    for line in text.splitlines():
        stripped = line.strip()

        # Detect header: a line whose first token is "Step" (case-sensitive as
        # LAMMPS always prints it exactly this way).
        tokens = stripped.split()
        if tokens and tokens[0] == "Step":
            header = tokens
            in_table = True
            # Initialise columns (reset on new header to handle multi-block).
            for col in header:
                columns.setdefault(col, [])
            continue

        if in_table:
            if data_re.match(line):
                values = stripped.split()
                if len(values) == len(header):
                    for col, val in zip(header, values):
                        try:
                            columns[col].append(float(val))
                        except ValueError:
                            pass
            else:
                # Non-data line → end of current thermo block.
                if stripped:
                    in_table = False

    return columns

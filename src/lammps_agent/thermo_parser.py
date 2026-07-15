"""Shared LAMMPS thermo output parser.

Extracts the thermo table from LAMMPS stdout/log into a dict of named columns.
Used by the agent's thermo tools and deterministic analysis skills.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

_NUMBER_PATTERN = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)"
_DATA_ROW_RE = re.compile(rf"^\s*{_NUMBER_PATTERN}(?:\s+{_NUMBER_PATTERN})+\s*$", re.IGNORECASE)


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
            if _DATA_ROW_RE.match(line):
                values = stripped.split()
                if len(values) == len(header):
                    # Convert the complete row before mutating any column so a
                    # malformed value can never leave columns out of alignment.
                    parsed = [float(value) for value in values]
                    for col, value in zip(header, parsed):
                        columns[col].append(value)
            else:
                # Non-data line → end of current thermo block.
                if stripped:
                    in_table = False

    return columns


def has_sustained_linear_drift(
    series: Sequence[float],
    *,
    tail_fraction: float = 0.5,
    min_points: int = 5,
    r_squared_threshold: float = 0.9,
) -> bool:
    """Return whether the tail of *series* has a persistent linear trend.

    Tail variance alone misclassifies a steady ramp as convergence because a
    short ramp has less variance than the full ramp.  This check requires both
    a strong linear fit and a total fitted change larger than fit noise and
    floating-point resolution.
    """
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    if min_points < 3:
        raise ValueError("min_points must be at least 3")
    if not 0 <= r_squared_threshold <= 1:
        raise ValueError("r_squared_threshold must be in [0, 1]")

    values = [float(value) for value in series]
    if len(values) < min_points or not all(math.isfinite(value) for value in values):
        return False

    tail_count = min(
        len(values), max(min_points, math.ceil(len(values) * tail_fraction))
    )
    tail = values[-tail_count:]
    x_mean = (tail_count - 1) / 2.0
    y_mean = math.fsum(tail) / tail_count
    x_variance = math.fsum((index - x_mean) ** 2 for index in range(tail_count))
    slope = math.fsum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(tail)
    ) / x_variance

    fitted = [y_mean + slope * (index - x_mean) for index in range(tail_count)]
    total_squares = math.fsum((value - y_mean) ** 2 for value in tail)
    if total_squares == 0:
        return False
    residual_squares = math.fsum(
        (value - estimate) ** 2 for value, estimate in zip(tail, fitted)
    )
    r_squared = max(0.0, 1.0 - residual_squares / total_squares)

    drift_span = abs(slope) * (tail_count - 1)
    residual_rms = math.sqrt(residual_squares / tail_count)
    magnitude = max(1.0, *(abs(value) for value in tail))
    resolution = magnitude * 1e-12
    return (
        r_squared >= r_squared_threshold
        and drift_span > max(3.0 * residual_rms, resolution)
    )

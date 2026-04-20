"""Thermo tail extraction skill (moved from skills.py)."""

from __future__ import annotations

import re
from pathlib import Path

from .base import BasePostprocessSkill, SkillReport


class ThermoTailSkill(BasePostprocessSkill):
    name = "thermo_tail"

    def analyze(self, run_dir: Path) -> SkillReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return SkillReport(
                name=self.name,
                summary="stdout.log is missing, cannot analyze thermo output",
                details={"thermo_rows": 0},
            )

        text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        thermo_rows: list[str] = []
        row_pattern = re.compile(
            r"^\s*\d+(?:\s+[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)+\s*$"
        )
        for line in lines:
            if row_pattern.match(line):
                thermo_rows.append(line.strip())

        if not thermo_rows:
            return SkillReport(
                name=self.name,
                summary="No numeric thermo table rows detected",
                details={"thermo_rows": 0},
            )

        tail = thermo_rows[-5:]
        return SkillReport(
            name=self.name,
            summary=f"Detected {len(thermo_rows)} thermo rows, showing last {len(tail)}",
            details={"thermo_rows": len(thermo_rows), "tail": tail},
        )

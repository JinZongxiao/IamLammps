"""Base classes for post-processing skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillReport:
    name: str
    summary: str
    details: dict


class BasePostprocessSkill:
    name = "base"

    def analyze(self, run_dir: Path) -> SkillReport:
        raise NotImplementedError

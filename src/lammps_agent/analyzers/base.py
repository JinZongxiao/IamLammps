"""Base types for deterministic post-processing analyzers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnalyzerReport:
    name: str
    summary: str
    details: dict[str, object]


class BaseAnalyzer:
    name = "base"

    def analyze(self, run_dir: Path) -> AnalyzerReport:
        raise NotImplementedError

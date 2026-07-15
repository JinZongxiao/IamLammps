"""Deterministic post-processing analyzers available to the agent."""

from __future__ import annotations

from pathlib import Path

from .base import AnalyzerReport, BaseAnalyzer
from .convergence import ConvergenceAnalyzer
from .crack_analysis import CrackAnalysisAnalyzer
from .indent_analysis import IndentAnalysisAnalyzer
from .stress_strain import StressStrainAnalyzer
from .thermo_tail import ThermoTailAnalyzer
from .thermo_validator import ThermoValidatorAnalyzer

__all__ = [
    "AnalyzerError",
    "UnknownAnalyzerError",
    "BaseAnalyzer",
    "AnalyzerReport",
    "ThermoTailAnalyzer",
    "ConvergenceAnalyzer",
    "ThermoValidatorAnalyzer",
    "StressStrainAnalyzer",
    "IndentAnalysisAnalyzer",
    "CrackAnalysisAnalyzer",
    "build_analyzers",
    "run_analyzers",
]

_ANALYZER_REGISTRY: dict[str, type[BaseAnalyzer]] = {
    ThermoTailAnalyzer.name: ThermoTailAnalyzer,
    ConvergenceAnalyzer.name: ConvergenceAnalyzer,
    ThermoValidatorAnalyzer.name: ThermoValidatorAnalyzer,
    StressStrainAnalyzer.name: StressStrainAnalyzer,
    IndentAnalysisAnalyzer.name: IndentAnalysisAnalyzer,
    CrackAnalysisAnalyzer.name: CrackAnalysisAnalyzer,
}


class AnalyzerError(ValueError):
    """A deterministic analyzer request could not be fulfilled."""

    code = "ANALYZER_ERROR"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class UnknownAnalyzerError(AnalyzerError):
    code = "UNKNOWN_ANALYZER"

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Unknown analyzer: {name}")

    def as_dict(self) -> dict[str, str]:
        return {**super().as_dict(), "analyzer": self.name}


def register_analyzer(cls: type[BaseAnalyzer]) -> None:
    """Register an additional deterministic analyzer at runtime."""
    _ANALYZER_REGISTRY[cls.name] = cls


def build_analyzers(enabled: list[str]) -> list[BaseAnalyzer]:
    analyzers: list[BaseAnalyzer] = []
    for name in enabled:
        cls = _ANALYZER_REGISTRY.get(name)
        if cls is None:
            raise UnknownAnalyzerError(name)
        analyzers.append(cls())
    return analyzers


def run_analyzers(run_dir: Path, enabled: list[str]) -> list[AnalyzerReport]:
    reports: list[AnalyzerReport] = []
    for analyzer in build_analyzers(enabled):
        try:
            reports.append(analyzer.analyze(run_dir))
        except Exception as exc:
            reports.append(
                AnalyzerReport(
                    name=analyzer.name,
                    summary=f"Analyzer failed: {exc}",
                    details={"error": str(exc)},
                )
            )
    return reports


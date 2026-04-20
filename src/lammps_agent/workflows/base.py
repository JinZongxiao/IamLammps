"""Base classes for mechanical test workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..pipeline import PipelineResult, SimulationPipeline


@dataclass
class WorkflowConfig:
    temperature: float = 300.0
    potential_name: str = "W_Zhou2004_EAM"
    mpi_np: int | None = None
    timeout_per_stage: int | None = None


class BaseWorkflow(ABC):
    """Base class for mechanical test workflows."""

    def __init__(self, config: WorkflowConfig, agent: "LammpsAgent") -> None:  # noqa: F821
        self.config = config
        self.agent = agent

    @abstractmethod
    def build_pipeline(self) -> SimulationPipeline:
        """Construct the multi-stage pipeline for this workflow."""
        ...

    def run(self) -> PipelineResult:
        """Build and execute the pipeline."""
        pipeline = self.build_pipeline()
        return pipeline.run()

"""DBTT temperature sweep workflow — runs tensile tests at multiple temperatures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from ..pipeline import PipelineResult
from .base import BaseWorkflow, WorkflowConfig
from .tensile import TensileWorkflow


@dataclass
class DBTTResult:
    temperature_results: list[tuple[float, PipelineResult]]
    all_success: bool


class DBTTSweepWorkflow(BaseWorkflow):
    """Run tensile tests across a temperature range to determine DBTT."""

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        temperatures: list[float] | None = None,
        supercell: tuple[int, int, int] = (10, 10, 20),
        strain_rate: float = 1e8,
    ) -> None:
        super().__init__(config, agent)
        self.temperatures = temperatures or list(range(200, 1300, 100))
        self.supercell = supercell
        self.strain_rate = strain_rate

    def build_pipeline(self):
        """Not used directly — run() orchestrates multiple TensileWorkflows."""
        raise NotImplementedError("DBTTSweepWorkflow uses run() directly")

    def run(self) -> PipelineResult:
        """Run tensile test at each temperature and collect results.

        Returns a PipelineResult where stages are labelled by temperature.
        """
        all_stages = []
        all_success = True

        for T in self.temperatures:
            print(f"[DBTT] Running tensile test at T={T} K ...", file=sys.stderr)
            wf_config = WorkflowConfig(
                temperature=T,
                potential_name=self.config.potential_name,
                mpi_np=self.config.mpi_np,
                timeout_per_stage=self.config.timeout_per_stage,
            )
            tensile = TensileWorkflow(
                config=wf_config,
                agent=self.agent,
                supercell=self.supercell,
                strain_rate=self.strain_rate,
            )
            result = tensile.run()

            # Collect stages with temperature prefix.
            for stage_name, output in result.stages:
                all_stages.append((f"T{T}_{stage_name}", output))

            if not result.success:
                all_success = False
                print(f"[DBTT] T={T} K FAILED at stage {result.failed_stage}", file=sys.stderr)

        return PipelineResult(
            stages=all_stages,
            success=all_success,
            failed_stage=None if all_success else "see individual stages",
        )

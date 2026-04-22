"""DBTT temperature sweep workflow — runs tensile tests at multiple temperatures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
        # Production parameters (passed through to TensileWorkflow)
        equilibration_steps: int | None = None,
        timestep: float | None = None,
        thermo_every: int | None = None,
        dump_every: int | None = None,
    ) -> None:
        super().__init__(config, agent)
        self.temperatures = temperatures or list(range(200, 1300, 100))
        self.supercell = supercell
        self.strain_rate = strain_rate
        self.equilibration_steps = equilibration_steps
        self.timestep = timestep
        self.thermo_every = thermo_every
        self.dump_every = dump_every

    # Report config — set by main.py after construction
    _report_config: "DBTTReportConfig | None" = None  # type: ignore[name-defined]

    def build_pipeline(self):
        """Not used directly — run() orchestrates multiple TensileWorkflows."""
        raise NotImplementedError("DBTTSweepWorkflow uses run() directly")

    def run(self) -> PipelineResult:
        """Run tensile test at each temperature and collect results."""
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
                equilibration_steps=self.equilibration_steps,
                timestep=self.timestep,
                thermo_every=self.thermo_every,
                dump_every=self.dump_every,
            )
            # Each tensile run generates its own per-run report; suppress it here
            # so the DBTT aggregated report is the primary output.
            tensile._report_config = None  # keep per-T reports off (tensile auto-generates them)
            result = tensile.run()

            for stage_name, output in result.stages:
                all_stages.append((f"T{T}_{stage_name}", output))

            if not result.success:
                all_success = False
                print(f"[DBTT] T={T} K FAILED at stage {result.failed_stage}", file=sys.stderr)

        pipeline_result = PipelineResult(
            stages=all_stages,
            success=all_success,
            failed_stage=None if all_success else "see individual stages",
        )

        # Generate aggregated DBTT report
        self._try_generate_report(pipeline_result)

        return pipeline_result

    def _try_generate_report(self, pipeline_result: PipelineResult) -> None:
        try:
            from ..report import DBTTReporter, DBTTReportConfig
            cfg = self._report_config or DBTTReportConfig(
                temperatures=self.temperatures,
                supercell=self.supercell,
                potential_name=self.config.potential_name,
                strain_rate=self.strain_rate,
                lammps_command=self.agent.config.lammps_command,
                report_dir=None,  # auto-determined from stage results
            )
            DBTTReporter(cfg).generate(pipeline_result)
        except Exception as exc:
            print(f"[REPORT WARNING] DBTT report failed: {exc}", file=sys.stderr)

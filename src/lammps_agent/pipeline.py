"""Multi-stage simulation pipeline.

Orchestrates sequences of LAMMPS simulations where each stage can depend on
the output files (restart, data) of a previous stage.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .agent import AgentOutput, LammpsAgent
from .potentials.manager import PotentialManager


@dataclass
class PipelineStage:
    name: str                              # "minimize", "equilibrate", "deform"
    task_description: str                  # Natural-language task for the LLM
    depends_on: str | None = None          # Name of previous stage
    input_files: list[str] = field(default_factory=list)  # Files inherited
    timeout: int | None = None
    potential_name: str | None = None      # If set, potential is prepared
    extra_prompt_context: str = ""         # Additional context for the LLM


@dataclass
class PipelineResult:
    stages: list[tuple[str, AgentOutput]]  # (stage_name, output)
    success: bool
    failed_stage: str | None = None


class SimulationPipeline:
    """Execute a sequence of LAMMPS simulation stages."""

    def __init__(
        self,
        stages: list[PipelineStage],
        agent: LammpsAgent,
        potential_manager: PotentialManager | None = None,
    ) -> None:
        self.stages = stages
        self.agent = agent
        self.potential_manager = potential_manager or PotentialManager()

    def run(self) -> PipelineResult:
        """Execute all stages sequentially.  Stop on first failure."""
        results: list[tuple[str, AgentOutput]] = []
        outputs_by_name: dict[str, AgentOutput] = {}

        for stage in self.stages:
            _log_stage(stage.name, "starting")

            # Build enriched task.
            prev = outputs_by_name.get(stage.depends_on) if stage.depends_on else None
            task = self._prepare_stage_task(stage, prev)

            # If the stage specifies a potential, prepare it.
            if stage.potential_name:
                pot = self.potential_manager.resolve_potential(stage.potential_name)
                pair_cmds = self.potential_manager.generate_pair_commands(pot)
                task += (
                    f"\n\nPotential setup (use these exact commands):\n"
                    f"  {pair_cmds}\n"
                    f"  mass 1 {pot.atomic_masses.get(pot.elements[0], 183.84)}"
                )

            # Override agent timeout if the stage specifies one.
            original_timeout = self.agent.config.lammps_timeout
            if stage.timeout is not None:
                self.agent.config.lammps_timeout = stage.timeout

            # Create run_dir early so we can copy inherited files before
            # the agent generates and executes the script.
            run_dir = self.agent._create_run_dir()

            # Copy inherited files from previous stage into this run_dir.
            if prev and stage.input_files:
                self._copy_inherited_files(
                    prev.run_dir, run_dir, stage.input_files
                )

            output = self.agent.run(task=task, execute=True, run_dir=run_dir)

            self.agent.config.lammps_timeout = original_timeout

            results.append((stage.name, output))
            outputs_by_name[stage.name] = output

            if output.lammps_result and output.lammps_result.return_code == 0:
                _log_stage(stage.name, "succeeded")
            else:
                _log_stage(stage.name, "FAILED")
                return PipelineResult(
                    stages=results,
                    success=False,
                    failed_stage=stage.name,
                )

        return PipelineResult(stages=results, success=True, failed_stage=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_stage_task(
        self, stage: PipelineStage, prev_output: AgentOutput | None
    ) -> str:
        """Build the full task description, injecting inherited-file context."""
        parts = [stage.task_description]

        if stage.extra_prompt_context:
            parts.append(stage.extra_prompt_context)

        if prev_output and stage.input_files:
            file_list = ", ".join(stage.input_files)
            parts.append(
                f"The following file(s) from the previous stage are available "
                f"in the working directory: {file_list}. "
                f"Use 'read_restart <file>' or 'read_data <file>' as appropriate "
                f"to continue from the previous stage's output."
            )

        return "\n\n".join(parts)

    def _copy_inherited_files(
        self, src_dir: Path, dst_dir: Path, patterns: list[str]
    ) -> list[Path]:
        """Copy files matching *patterns* from *src_dir* to *dst_dir*."""
        copied: list[Path] = []
        for pattern in patterns:
            for src in src_dir.glob(pattern):
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(dst)
        return copied


def _log_stage(name: str, status: str) -> None:
    print(f"[PIPELINE] Stage '{name}': {status}", file=sys.stderr)

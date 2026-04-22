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
from .lammps_runner import LammpsRunResult, run_lammps


@dataclass
class PipelineStage:
    name: str                              # "minimize", "equilibrate", "deform"
    task_description: str                  # Natural-language task for the LLM
    depends_on: str | None = None          # Name of previous stage
    input_files: list[str] = field(default_factory=list)  # Files inherited
    timeout: int | None = None
    potential_name: str | None = None      # If set, potential is prepared
    extra_prompt_context: str = ""         # Additional context for the LLM
    script_template: str | None = None     # If set: use this verbatim, skip LLM entirely


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

            prev = outputs_by_name.get(stage.depends_on) if stage.depends_on else None

            # Create run_dir early so we can copy inherited files before execution.
            run_dir = self.agent._create_run_dir()

            # Copy inherited files from previous stage into this run_dir.
            if prev and stage.input_files:
                self._copy_inherited_files(prev.run_dir, run_dir, stage.input_files)

            # --- Deterministic template path (no LLM, no retry) ---
            if stage.script_template is not None:
                output = self._run_template_stage(stage, run_dir)
            else:
                # --- LLM generation path (existing behaviour) ---
                task = self._prepare_stage_task(stage, prev)

                if stage.potential_name:
                    pot = self.potential_manager.resolve_potential(stage.potential_name)
                    pair_cmds = self.potential_manager.generate_pair_commands(pot)
                    task += (
                        f"\n\nPotential setup (use these exact commands):\n"
                        f"  {pair_cmds}\n"
                        f"  mass 1 {pot.atomic_masses.get(pot.elements[0], 183.84)}"
                    )

                original_timeout = self.agent.config.lammps_timeout
                if stage.timeout is not None:
                    self.agent.config.lammps_timeout = stage.timeout

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

    def _run_template_stage(self, stage: PipelineStage, run_dir: Path) -> "AgentOutput":
        """Execute a stage using a pre-built deterministic script (no LLM, no retry).

        On failure: logs the first ERROR line from stdout with its approximate
        script line number, then returns a failed AgentOutput (fail-fast).
        """
        from .agent import AgentOutput  # avoid circular at module level

        # Resolve potential file path inside run_dir (template uses filename only)
        script = stage.script_template  # already has absolute pot path injected by caller

        input_path = run_dir / "in.lammps"
        input_path.write_text(script, encoding="utf-8")

        original_timeout = self.agent.config.lammps_timeout
        if stage.timeout is not None:
            self.agent.config.lammps_timeout = stage.timeout

        lmp_result = run_lammps(
            self.agent.config.lammps_command,
            input_path,
            run_dir,
            timeout=self.agent.config.lammps_timeout,
            mpi_np=self.agent.config.mpi_np,
        )
        self.agent.config.lammps_timeout = original_timeout

        if lmp_result.return_code != 0:
            # Emit a targeted diagnostic: find the ERROR line + last LAMMPS command
            stdout = Path(lmp_result.stdout_path).read_text(encoding="utf-8", errors="ignore")
            _log_template_failure(stage.name, script, stdout)

        return AgentOutput(
            run_dir=run_dir,
            input_path=input_path,
            lammps_result=lmp_result,
            skill_reports=[],
            compressed_memory_context="",
            retry_count=0,
        )

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


def _log_template_failure(stage_name: str, script: str, stdout: str) -> None:
    """Print a targeted diagnostic when a deterministic-template stage fails."""
    lines = script.splitlines()

    # Find the last "Last command:" line in LAMMPS output to identify failing line
    last_cmd = None
    error_msg = None
    for ln in stdout.splitlines():
        if ln.startswith("Last command:"):
            last_cmd = ln[len("Last command:"):].strip()
        if ln.startswith("ERROR"):
            error_msg = ln

    print(f"[TEMPLATE FAIL] Stage '{stage_name}' script error:", file=sys.stderr)
    if error_msg:
        print(f"  {error_msg}", file=sys.stderr)
    if last_cmd:
        # Find the line number in the script
        for i, sl in enumerate(lines, 1):
            if sl.strip() == last_cmd.strip():
                print(f"  script line {i}: {sl}", file=sys.stderr)
                break
        else:
            print(f"  failing command: {last_cmd}", file=sys.stderr)
    print(
        f"  [TEMPLATE FAIL] This stage uses a deterministic template — no retry attempted.",
        file=sys.stderr,
    )

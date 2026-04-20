from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import AgentConfig
from .error_parser import LammpsError, parse_lammps_error
from .lammps_runner import LammpsRunResult, run_lammps
from .llm_clients import ChatMessage, build_chat_client
from .memory import SimulationMemoryStore
from .potentials.manager import PotentialManager
from .reward import compute_reward
from .skills import SkillReport, run_postprocess_skills


SYSTEM_PROMPT = """\
You are an expert LAMMPS input script generator.
Return a valid LAMMPS input script only. Do not explain.

Strict command ordering rules — violating these causes fatal errors:
1. units, atom_style, boundary (must come first)
2. lattice (REQUIRED before create_atoms ... box/region)
3. region
4. create_box
5. create_atoms (lattice MUST already be defined)
6. mass (for every atom type)
7. pair_style, pair_coeff (MUST come after create_box)
8. neighbor, neigh_modify (optional)
8.5. velocity (REQUIRED for dynamics at finite temperature —
     e.g. 'velocity all create 300.0 12345 dist gaussian'.
     Without this, all atoms start at zero velocity and temperature will be 0.)
9. fix (time integration: nve/nvt/npt)
10. timestep, thermo, thermo_style, dump (output settings)
11. run or minimize

Key rules:
- NEVER place pair_coeff before create_box — this is a fatal error.
- NEVER use "create_atoms N box" without a preceding "lattice" command — this is a fatal error.
- For LJ fluid: use "lattice fcc 0.8442" (or appropriate density).
- For crystals: choose the correct lattice type and constant for the material.
- Always set mass for all atom types before run/minimize.
- If dynamics is requested, include a time integration fix (e.g. fix nve or nvt).
- For ANY dynamics at finite temperature, ALWAYS include a velocity command.
- Prefer robust defaults and include thermo output.

Tungsten (W) simulation rules:
- units metal (mandatory for EAM potentials)
- atom_style atomic
- lattice bcc 3.165 (BCC tungsten, a = 3.165 Angstrom)
- pair_style eam/alloy
- pair_coeff * * W_zhou.eam.alloy W
- mass 1 183.84 (atomic mass of W)
- For dynamics: velocity all create <T> <random_seed> — REQUIRED or T=0
- Typical timestep: 0.001 (ps, i.e. 1 fs in metal units)
- For restarted simulations: use read_restart instead of lattice/create_atoms.
  After read_restart, you still need pair_style/pair_coeff/mass but NOT lattice/region/create_box/create_atoms.
"""


@dataclass
class AgentOutput:
    run_dir: Path
    input_path: Path
    lammps_result: LammpsRunResult | None
    skill_reports: list[SkillReport]
    compressed_memory_context: str
    retry_count: int = 0
    error_history: list[LammpsError] | None = None


class LammpsAgent:
    def __init__(self, config: AgentConfig, runs_root: Path | None = None) -> None:
        self.config = config
        self.client = build_chat_client(config)
        self.runs_root = runs_root or Path("runs")
        self.memory_store = SimulationMemoryStore(
            memory_file=Path(config.memory_file),
            max_entries=config.memory_max_entries,
        )
        self.potential_manager = PotentialManager()

    def generate_input_script(
        self, task: str, error_context: str = ""
    ) -> str:
        compressed_context = self.memory_store.compressed_context(
            max_chars=self.config.memory_context_chars
        )
        user_prompt = self._build_user_prompt(task, compressed_context, error_context)
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
        text = self.client.chat(messages)
        return self._extract_script(text)

    def run(self, task: str, execute: bool = True, run_dir: Path | None = None) -> AgentOutput:
        run_dir = run_dir or self._create_run_dir()

        # Copy known potential files to run_dir so LAMMPS can find them.
        self._prepare_potentials(run_dir)

        compressed_context = self.memory_store.compressed_context(
            max_chars=self.config.memory_context_chars
        )
        max_retries = self.config.max_retries

        error_context = ""
        error_history: list[LammpsError] = []
        result: LammpsRunResult | None = None
        skill_reports: list[SkillReport] = []
        input_path = run_dir / "in.lammps"
        retry_count = 0

        for attempt in range(max_retries + 1):
            input_text = self.generate_input_script(task, error_context)
            # Save each attempt with a suffix so we can inspect the history.
            if attempt == 0:
                input_path.write_text(input_text, encoding="utf-8")
            else:
                retry_path = run_dir / f"in.lammps.retry{attempt}"
                retry_path.write_text(input_text, encoding="utf-8")
                # Overwrite the canonical input file for execution.
                input_path.write_text(input_text, encoding="utf-8")

            if not execute:
                break

            result = run_lammps(
                self.config.lammps_command, input_path, run_dir,
                timeout=self.config.lammps_timeout,
                mpi_np=self.config.mpi_np,
            )

            if result.return_code == 0:
                # Success — run post-processing skills and break.
                skill_reports = run_postprocess_skills(
                    run_dir, self.config.enabled_skills
                )
                break

            # --- Failure: diagnose and prepare retry context ---
            retry_count = attempt + 1
            stdout_text = Path(result.stdout_path).read_text(
                encoding="utf-8", errors="ignore"
            )
            stderr_text = Path(result.stderr_path).read_text(
                encoding="utf-8", errors="ignore"
            )
            error = parse_lammps_error(stdout_text, stderr_text)

            if error is not None:
                error_history.append(error)
                _log_retry(attempt + 1, max_retries, error)
                error_context = self._build_error_context(
                    attempt + 1, input_text, error
                )
            else:
                _log_retry_unknown(attempt + 1, max_retries, stdout_text, stderr_text)
                error_context = (
                    f"Attempt {attempt + 1} failed (return code {result.return_code}) "
                    "but the error could not be parsed. Review the script carefully."
                )

            if attempt == max_retries:
                # Exhausted retries — still run skills on the last attempt for diagnostics.
                skill_reports = run_postprocess_skills(
                    run_dir, self.config.enabled_skills
                )

        summary = self._build_memory_summary(result, skill_reports, retry_count)

        # Compute reward signal for this run.
        reward = compute_reward(
            return_code=result.return_code if result else None,
            run_dir=run_dir,
            retry_count=retry_count,
            task=task,
        )

        # Determine the last error type (if any).
        last_error_type = error_history[-1].error_type if error_history else None

        # Compute a short hash of the final script for deduplication.
        script_text = input_path.read_text(encoding="utf-8", errors="ignore")
        script_hash = hashlib.md5(script_text.encode()).hexdigest()[:12]

        self.memory_store.append(
            self.memory_store.make_entry(
                task=task,
                run_dir=run_dir,
                return_code=result.return_code if result else None,
                summary=summary,
                reward_score=reward.total,
                error_type=last_error_type,
                retry_count=retry_count,
                script_hash=script_hash,
            )
        )

        return AgentOutput(
            run_dir=run_dir,
            input_path=input_path,
            lammps_result=result,
            skill_reports=skill_reports,
            compressed_memory_context=compressed_context,
            retry_count=retry_count,
            error_history=error_history or None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_root / timestamp
        # Add numeric suffix if the directory already exists (e.g. multiple
        # stages within the same second in a pipeline).
        if run_dir.exists():
            for i in range(1, 1000):
                candidate = self.runs_root / f"{timestamp}_{i}"
                if not candidate.exists():
                    run_dir = candidate
                    break
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _prepare_potentials(self, run_dir: Path) -> None:
        """Copy all registered potential files into *run_dir*.

        This ensures that LAMMPS can find potential files by relative name
        (e.g. ``W_zhou.eam.alloy``) regardless of the working directory.
        """
        for entry in self.potential_manager.registry.list_all():
            if entry.file_path.exists():
                self.potential_manager.prepare_for_run(entry, run_dir)

    @staticmethod
    def _extract_script(text: str) -> str:
        # If model wraps script in markdown code fences, keep only script body.
        fenced = re.search(r"```(?:lammps|LAMMPS)?\n(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip() + "\n"
        return text.strip() + "\n"

    @staticmethod
    def _build_user_prompt(
        task: str, compressed_context: str, error_context: str = ""
    ) -> str:
        parts: list[str] = []
        parts.append(f"Current task:\n{task}")
        if error_context:
            parts.append(error_context)
        if compressed_context:
            parts.append(
                "Use the memory below to avoid repeating previous mistakes "
                "and improve script quality.\n" + compressed_context
            )
        return "\n\n".join(parts)

    @staticmethod
    def _build_error_context(
        attempt: int, failed_script: str, error: LammpsError
    ) -> str:
        return (
            f"--- RETRY (attempt {attempt} failed) ---\n"
            f"Error type: {error.error_type}\n"
            f"LAMMPS error message: {error.raw_message}\n"
            f"Fix instruction: {error.suggested_fix}\n"
            f"\n"
            f"The script that caused this error:\n"
            f"```\n{failed_script}```\n"
            f"\n"
            f"Generate a corrected script that fixes the error above. "
            f"Return ONLY the complete corrected LAMMPS script."
        )

    @staticmethod
    def _build_memory_summary(
        result: LammpsRunResult | None,
        skill_reports: list[SkillReport],
        retry_count: int = 0,
    ) -> str:
        parts: list[str] = []
        if result is None:
            parts.append("script_generated_only")
        else:
            parts.append(f"return_code={result.return_code}")
        if retry_count > 0:
            parts.append(f"retries={retry_count}")
        for report in skill_reports:
            parts.append(f"{report.name}:{report.summary}")
        return " | ".join(parts)


# ------------------------------------------------------------------
# Logging helpers (stderr so they don't pollute JSON output on stdout)
# ------------------------------------------------------------------

def _log_retry(attempt: int, max_retries: int, error: LammpsError) -> None:
    print(
        f"[RETRY {attempt}/{max_retries}] {error.error_type}: {error.raw_message}",
        file=sys.stderr,
    )
    print(f"  Fix: {error.suggested_fix}", file=sys.stderr)


def _log_retry_unknown(
    attempt: int, max_retries: int, stdout: str, stderr: str
) -> None:
    # Show last 3 lines of output for debugging.
    tail = "\n".join((stdout + stderr).strip().splitlines()[-3:])
    print(
        f"[RETRY {attempt}/{max_retries}] Unknown error. Tail:\n{tail}",
        file=sys.stderr,
    )

"""Narrow, auditable tools exposed to the simulation coding agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents import FunctionTool, function_tool
from agents.run_context import RunContextWrapper

from .analysis_runner import run_analysis_script
from .analyzers import UnknownAnalyzerError, run_analyzers
from .config import AgentConfig
from .plotting import PlotError
from .plotting import plot_run as create_run_plots
from .potential_catalog import LammpsPotentialCatalog
from .potentials.manager import PotentialManager
from .potentials.registry import PotentialRegistry
from .run_artifacts import RunArtifactStore
from .script_policy import validate_lammps_script
from .thermo_parser import parse_thermo_table
from .web import WebResearch
from .workspace import Workspace


@dataclass
class SimulationContext:
    config: AgentConfig
    workspace: Workspace
    project_root: Path
    skill_registry: Any
    web: WebResearch
    potential_catalog: LammpsPotentialCatalog


def build_simulation_tools() -> list[FunctionTool]:
    """Create the common function-tool set for OpenAI and DeepSeek models."""

    @function_tool
    async def list_files(
        ctx: RunContextWrapper[SimulationContext],
        directory: str,
        max_files: int,
    ) -> str:
        """List bounded text files and SHA-256 hashes below a workspace directory."""
        files = ctx.context.workspace.list_text_files(directory, max_files=max_files)
        return _json([asdict(item) for item in files])

    @function_tool
    async def read_file(
        ctx: RunContextWrapper[SimulationContext],
        path: str,
        max_bytes: int,
    ) -> str:
        """Read a UTF-8 workspace file and return content, size, and SHA-256."""
        return _json(asdict(ctx.context.workspace.read_text(path, max_bytes=max_bytes)))

    @function_tool(needs_approval=True)
    async def write_file(
        ctx: RunContextWrapper[SimulationContext],
        path: str,
        content: str,
        expected_sha256: str | None,
    ) -> str:
        """Create a text file, or overwrite only the exact SHA-256 version previously read."""
        _reject_managed_artifact_write(ctx.context.workspace, path)
        if _is_lammps_input(path):
            validate_lammps_script(content)
        try:
            result = ctx.context.workspace.write_text(
                path, content, expected_sha256=expected_sha256
            )
        except FileExistsError:
            existing = ctx.context.workspace.read_text(path)
            return _json(
                {
                    "code": "FILE_EXISTS",
                    "path": existing.path,
                    "sha256": existing.sha256,
                    "size": existing.size,
                    "message": (
                        "read the current hash and update this path; "
                        "do not create an equivalent path"
                    ),
                }
            )
        return _json(asdict(result))

    @function_tool(needs_approval=True)
    async def replace_file(
        ctx: RunContextWrapper[SimulationContext],
        path: str,
        old_text: str,
        new_text: str,
        expected_sha256: str,
    ) -> str:
        """Replace one exact text occurrence in the exact file version previously read."""
        workspace = ctx.context.workspace
        _reject_managed_artifact_write(workspace, path)
        if _is_lammps_input(path):
            current = workspace.read_text(path)
            if current.sha256 != expected_sha256:
                raise ValueError("LAMMPS input changed since it was read")
            if current.content.count(old_text) != 1:
                raise ValueError("old_text must occur exactly once")
            validate_lammps_script(current.content.replace(old_text, new_text, 1))
        result = workspace.replace_text(
            path,
            old_text,
            new_text,
            expected_sha256=expected_sha256,
        )
        return _json(asdict(result))

    @function_tool(needs_approval=True)
    async def run_lammps_input(
        ctx: RunContextWrapper[SimulationContext],
        input_path: str,
        expected_sha256: str,
        timeout_seconds: int | None,
        mpi_processes: int | None,
    ) -> str:
        """Run a reviewed runs/<job>/inputs file in an immutable execution snapshot."""
        workspace = ctx.context.workspace
        script_file = workspace.read_text(input_path)
        if script_file.sha256 != expected_sha256:
            raise ValueError("LAMMPS input changed after it was reviewed")
        validate_lammps_script(script_file.content)
        resolved = workspace.resolve(input_path)
        relative = resolved.relative_to(workspace.root)
        if len(relative.parts) < 4 or relative.parts[0] != "runs" or relative.parts[2] != "inputs":
            raise ValueError("LAMMPS inputs must be below runs/<job>/inputs/")
        timeout = (
            ctx.context.config.lammps_timeout
            if timeout_seconds is None
            else timeout_seconds
        )
        mpi = ctx.context.config.mpi_np if mpi_processes is None else mpi_processes
        if timeout is not None and not 1 <= timeout <= 86_400:
            raise ValueError("timeout_seconds must be between 1 and 86400")
        if mpi is not None and not 1 <= mpi <= 256:
            raise ValueError("mpi_processes must be between 1 and 256")
        warnings = _scientific_preflight(script_file.content)
        store = RunArtifactStore(workspace.root)
        manifest = store.execute(
            relative,
            command=ctx.context.config.lammps_command,
            expected_sha256=expected_sha256,
            timeout=timeout,
            mpi_np=mpi,
            environment=_minimal_process_environment(),
        )
        payload = {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "return_code": manifest.return_code,
            "wall_time": manifest.wall_time,
            "artifact_dir": manifest.artifact_dir,
            "input_sha256": manifest.input_sha256,
            "output_files": list(manifest.output_files),
            "stdout_tail": _tail(store.resolve_manifest_path(manifest.stdout_path)),
            "stderr_tail": _tail(store.resolve_manifest_path(manifest.stderr_path)),
            "preflight_warnings": warnings,
        }
        return _json(payload)

    @function_tool
    async def list_runs(
        ctx: RunContextWrapper[SimulationContext],
        job: str,
    ) -> str:
        """List immutable LAMMPS execution manifests for one job."""
        manifests = RunArtifactStore(ctx.context.workspace.root).list_runs(job)
        return _json([manifest.to_dict() for manifest in manifests])

    @function_tool
    async def inspect_run(
        ctx: RunContextWrapper[SimulationContext],
        run_id: str,
    ) -> str:
        """Inspect one immutable execution manifest by run ID."""
        manifest = RunArtifactStore(ctx.context.workspace.root).inspect_run(run_id)
        return _json(manifest.to_dict())

    @function_tool
    async def parse_thermo(
        ctx: RunContextWrapper[SimulationContext],
        run_id: str,
        tail_rows: int,
    ) -> str:
        """Parse thermo columns from one immutable run selected by run ID."""
        store = RunArtifactStore(ctx.context.workspace.root)
        manifest = store.inspect_run(run_id)
        selected = manifest.lammps_log_path or manifest.stdout_path
        path = store.resolve_manifest_path(selected)
        table = parse_thermo_table(path)
        if not table:
            raise ValueError(f"run {run_id} contains no thermo data")
        rows = max(1, min(tail_rows, 200))
        return _json({name: values[-rows:] for name, values in table.items()})

    @function_tool
    async def analyze_run(
        ctx: RunContextWrapper[SimulationContext],
        run_id: str,
        analyzer_names: list[str],
    ) -> str:
        """Run deterministic analyzers on one immutable run selected by run ID."""
        store = RunArtifactStore(ctx.context.workspace.root)
        manifest = store.inspect_run(run_id)
        directory = store.resolve_manifest_path(manifest.artifact_dir)
        names = analyzer_names or ctx.context.config.enabled_analyzers
        try:
            reports = run_analyzers(directory, names)
        except UnknownAnalyzerError as exc:
            return _json(exc.as_dict())
        return _json([asdict(report) for report in reports])

    @function_tool
    async def plot_run(
        ctx: RunContextWrapper[SimulationContext],
        run_id: str,
        plot_types: list[str],
    ) -> str:
        """Create validated deterministic thermo or O-O RDF PNGs for one run."""
        try:
            plots = create_run_plots(ctx.context.workspace.root, run_id, plot_types)
        except PlotError as exc:
            return _json(exc.as_dict())
        return _json([plot.to_dict() for plot in plots])

    @function_tool(needs_approval=True)
    async def run_python_analysis(
        ctx: RunContextWrapper[SimulationContext],
        script_path: str,
        expected_sha256: str,
        timeout_seconds: int,
    ) -> str:
        """Run a reviewed Python analysis script without exposing a generic shell."""
        script = ctx.context.workspace.read_text(script_path)
        if script.sha256 != expected_sha256:
            raise ValueError("Analysis script changed after it was reviewed")
        path = ctx.context.workspace.resolve(script_path)
        relative = path.relative_to(ctx.context.workspace.root)
        if len(relative.parts) < 3 or relative.parts[0] != "runs":
            raise ValueError("Python analysis scripts may only be executed below runs/")
        return _json(run_analysis_script(path, path.parent, timeout_seconds))

    @function_tool
    async def list_local_potentials(ctx: RunContextWrapper[SimulationContext]) -> str:
        """List locally installed/discovered potentials and their declared metadata."""
        registry = PotentialRegistry(ctx.context.config.potentials_dir)
        return _json(
            [
                {
                    "name": entry.name,
                    "elements": entry.elements,
                    "pair_style": entry.pair_style,
                    "file": str(entry.file_path),
                    "metadata": entry.metadata,
                }
                for entry in registry.list_all()
            ]
        )

    @function_tool(needs_approval=True)
    async def stage_local_potential(
        ctx: RunContextWrapper[SimulationContext],
        potential_name: str,
        run_directory: str,
    ) -> str:
        """Copy one registered local potential into a run directory with exact commands."""
        workspace = ctx.context.workspace
        run_dir = workspace.resolve(run_directory)
        relative = run_dir.relative_to(workspace.root)
        if len(relative.parts) != 3 or relative.parts[0] != "runs" or relative.parts[2] != "inputs":
            raise ValueError("Local potentials must be staged in runs/<job>/inputs/")
        run_dir.mkdir(parents=True, exist_ok=True)
        registry = PotentialRegistry(ctx.context.config.potentials_dir)
        manager = PotentialManager(registry)
        potential = manager.resolve_potential(potential_name)
        target = run_dir / potential.file_path.name
        if target.exists():
            raise FileExistsError(f"Potential already staged: {target.name}")
        with tempfile.NamedTemporaryFile(dir=run_dir, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            shutil.copy2(potential.file_path, temp_path)
            temp_path.replace(target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        commands = manager.generate_pair_commands(potential, local_path=target)
        return _json(
            {
                "potential_name": potential.name,
                "path": target.relative_to(workspace.root).as_posix(),
                "filename": target.name,
                "sha256": digest,
                "elements": potential.elements,
                "pair_style": potential.pair_style,
                "pair_commands": commands,
                "metadata": potential.metadata,
            }
        )

    @function_tool(needs_approval=True)
    async def search_potentials(
        ctx: RunContextWrapper[SimulationContext],
        query: str,
        limit: int,
    ) -> str:
        """Search the latest pinned official LAMMPS release for potential data files."""
        results = ctx.context.potential_catalog.search(query, limit)
        return _json([asdict(result) for result in results])

    @function_tool(needs_approval=True)
    async def install_potential(
        ctx: RunContextWrapper[SimulationContext],
        candidate_id: str,
        expected_filename: str,
        expected_revision: str,
        expected_download_url: str,
        expected_git_blob_sha: str,
        expected_size: int,
        destination_directory: str,
    ) -> str:
        """Download an earlier official-catalog candidate into a workspace directory."""
        workspace = ctx.context.workspace
        destination = workspace.resolve(destination_directory)
        relative = destination.relative_to(workspace.root)
        if not relative.parts or relative.parts[0] not in {"potentials", "runs"}:
            raise ValueError("Potentials may only be installed below potentials/ or runs/")
        candidate = ctx.context.potential_catalog.get_candidate(candidate_id)
        if (
            candidate.filename != expected_filename
            or candidate.revision != expected_revision
            or candidate.download_url != expected_download_url
            or candidate.git_blob_sha != expected_git_blob_sha
            or candidate.size != expected_size
        ):
            raise ValueError("Potential candidate metadata does not match the approved values")
        return _json(ctx.context.potential_catalog.install(candidate_id, destination))

    @function_tool(needs_approval=True)
    async def web_search(
        ctx: RunContextWrapper[SimulationContext],
        query: str,
        limit: int,
    ) -> str:
        """Search the public web for scientific documentation and references."""
        results = ctx.context.web.search(query, limit)
        return _json(ctx.context.web.as_dicts(results))

    @function_tool(needs_approval=True)
    async def fetch_web_result(
        ctx: RunContextWrapper[SimulationContext],
        result_id: str,
        expected_url: str,
        max_chars: int,
    ) -> str:
        """Fetch bounded text from a result ID returned by web_search."""
        return _json(ctx.context.web.fetch(result_id, max_chars, expected_url))

    @function_tool
    async def list_skills(ctx: RunContextWrapper[SimulationContext]) -> str:
        """List project simulation skills that can be loaded on demand."""
        return _json(
            [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "recommended_tools": skill.recommended_tools,
                }
                for skill in ctx.context.skill_registry.discover()
            ]
        )

    @function_tool
    async def load_skill(
        ctx: RunContextWrapper[SimulationContext],
        name: str,
    ) -> str:
        """Load the trusted instructions for one registered project simulation skill."""
        return _json(asdict(ctx.context.skill_registry.get(name)))

    return [
        list_files,
        read_file,
        write_file,
        replace_file,
        run_lammps_input,
        list_runs,
        inspect_run,
        parse_thermo,
        analyze_run,
        plot_run,
        run_python_analysis,
        list_local_potentials,
        stage_local_potential,
        search_potentials,
        install_potential,
        web_search,
        fetch_web_result,
        list_skills,
        load_skill,
    ]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _tail(path: Path, max_chars: int = 12_000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def _is_lammps_input(path: str) -> bool:
    name = Path(path).name.casefold()
    return name.startswith("in.") or name.endswith((".in", ".lammps", ".lmp"))


def _reject_managed_artifact_write(workspace: Workspace, path: str) -> None:
    relative = workspace.relative_path(path)
    parts = tuple(part.casefold() for part in relative.parts)
    if len(parts) >= 3 and parts[0] == "runs" and parts[2] in {
        "executions",
        "latest.json",
    }:
        raise ValueError(
            "managed run artifacts are read-only; modify runs/<job>/inputs/ "
            "and create a new execution"
        )


def _minimal_process_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _scientific_preflight(script: str) -> list[str]:
    """Return conservative warnings without claiming scientific validation."""

    warnings: list[str] = []
    lowered = script.casefold()
    units_real = re.search(r"(?m)^\s*units\s+real\b", lowered) is not None
    if units_real:
        for match in re.finditer(r"(?m)^\s*lattice\s+\S+\s+([-+0-9.eE]+)", script):
            try:
                scale = float(match.group(1))
            except ValueError:
                continue
            if 0 < scale < 0.5:
                warnings.append(
                    "units real with lattice scale below 0.5 Angstrom is unusually "
                    "small; verify units and lattice semantics"
                )
                break
    if re.search(r"(?m)^\s*create_atoms\b", lowered) and not re.search(
        r"(?m)^\s*(region|read_data|read_restart)\b", lowered
    ):
        warnings.append("create_atoms appears without an explicit region or loaded simulation box")
    for match in re.finditer(
        r"(?m)^\s*create_atoms\s+\S+\s+random\s+([-+0-9]+)\b", script
    ):
        if int(match.group(1)) <= 0:
            warnings.append("create_atoms random requests a non-positive particle count")
    box = re.search(
        r"(?m)^\s*region\s+\S+\s+block\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        script,
    )
    cutoff = re.search(
        r"(?m)^\s*pair_style\s+\S*(?:cut|coul)\S*\s+([-+0-9.eE]+)", script
    )
    if box is not None and cutoff is not None:
        try:
            bounds = [float(value) for value in box.groups()]
            lengths = [bounds[index + 1] - bounds[index] for index in (0, 2, 4)]
            pair_cutoff = float(cutoff.group(1))
        except ValueError:
            pass
        else:
            if min(lengths) > 0 and pair_cutoff >= min(lengths) / 2:
                warnings.append(
                    "pair cutoff is at least half the smallest periodic box length; "
                    "verify minimum-image assumptions"
                )
    if "atom_style full" in lowered and "fix" in lowered and "shake" in lowered:
        if "pair_style" in lowered and "coul/long" in lowered and "kspace_style" not in lowered:
            warnings.append("long-range Coulomb water model has no kspace_style declaration")
    return warnings

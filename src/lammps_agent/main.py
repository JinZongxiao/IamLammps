from __future__ import annotations

import argparse
import json
import sys

from .agent import LammpsAgent
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LAMMPS Agent CLI")

    # --- Task mode (single task) ---
    parser.add_argument("--task", default=None, help="Natural language simulation request")
    parser.add_argument(
        "--run",
        dest="run",
        action="store_true",
        help="Generate script and execute LAMMPS",
    )
    parser.add_argument(
        "--no-run",
        dest="run",
        action="store_false",
        help="Generate script only",
    )
    parser.set_defaults(run=True)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Max auto-retry attempts on LAMMPS errors (overrides env MAX_RETRIES, default 3)",
    )

    # --- Workflow mode ---
    parser.add_argument(
        "--workflow",
        choices=["tensile", "nanoindent", "crack", "dbtt", "interface", "he_trapping"],
        default=None,
        help="Run a predefined mechanical test workflow",
    )
    parser.add_argument("--potential", type=str, default=None, help="Potential name (e.g. W_Zhou2004_EAM)")
    parser.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature in K")
    parser.add_argument("--supercell", type=str, default="10,10,20", help="Supercell nx,ny,nz")
    parser.add_argument("--strain-rate", type=float, default=1e8, help="Strain rate for tensile (/s)")

    # --- Production parameters (optional, defaults preserved) ---
    parser.add_argument("--runs-dir", type=str, default=None, help="Output directory for runs (overrides RUNS_DIR env var)")
    parser.add_argument("--equil-steps", type=int, default=None, help="Equilibration steps (default 20000)")
    parser.add_argument("--timestep", type=float, default=None, help="Timestep in ps (default 0.001)")
    parser.add_argument("--thermo-every", type=int, default=None, help="Output thermo every N steps (default 100)")
    parser.add_argument("--dump-every", type=int, default=None, help="Dump atoms every N steps (default: no dump)")

    # --- Runner options ---
    parser.add_argument("--mpi-np", type=int, default=None, help="Number of MPI processes")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per stage in seconds")

    return parser


def _run_single_task(args: argparse.Namespace) -> int:
    """Original single-task mode."""
    config = load_config()
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    if args.mpi_np is not None:
        config.mpi_np = args.mpi_np
    if args.timeout is not None:
        config.lammps_timeout = args.timeout

    agent = LammpsAgent(config)
    result = agent.run(task=args.task, execute=args.run)

    payload = {
        "run_dir": str(result.run_dir),
        "input_path": str(result.input_path),
        "executed": bool(result.lammps_result),
        "skills": [
            {"name": s.name, "summary": s.summary, "details": s.details}
            for s in result.skill_reports
        ],
        "memory_context_chars": len(result.compressed_memory_context),
    }

    if result.lammps_result:
        payload["lammps"] = {
            "return_code": result.lammps_result.return_code,
            "stdout_path": result.lammps_result.stdout_path,
            "stderr_path": result.lammps_result.stderr_path,
            "command": result.lammps_result.command,
        }
        payload["retry_count"] = result.retry_count

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_workflow(args: argparse.Namespace) -> int:
    """Run a predefined mechanical test workflow."""
    from .workflows import build_workflow, WorkflowConfig

    config = load_config()
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    if args.mpi_np is not None:
        config.mpi_np = args.mpi_np
    if args.timeout is not None:
        config.lammps_timeout = args.timeout

    # Apply production parameters to WorkflowConfig
    wf_config = WorkflowConfig(
        temperature=args.temperature,
        potential_name=args.potential or "W_Zhou2004_EAM",
        mpi_np=args.mpi_np,
        timeout_per_stage=args.timeout,
        equilibration_steps=args.equil_steps if args.equil_steps else None,
        timestep=args.timestep,
        thermo_every=args.thermo_every,
        dump_every=args.dump_every,
    )

    # Apply runs_dir to agent if specified
    from pathlib import Path
    runs_root = Path(args.runs_dir) if args.runs_dir else None
    agent = LammpsAgent(config, runs_root=runs_root)

    nx, ny, nz = (int(x) for x in args.supercell.split(","))
    workflow = build_workflow(
        name=args.workflow,
        config=wf_config,
        agent=agent,
        supercell=(nx, ny, nz),
        strain_rate=args.strain_rate,
        equilibration_steps=args.equil_steps,
        timestep=args.timestep,
        thermo_every=args.thermo_every,
        dump_every=args.dump_every,
    )

    # Inject report config so each workflow reporter has full CLI params
    try:
        from .report import (
            TensileReportConfig, NanoindentReportConfig,
            CrackReportConfig, DBTTReportConfig,
        )
        _pot = args.potential or "W_Zhou2004_EAM"
        _lmp = config.lammps_command
        if args.workflow == "tensile":
            workflow._report_config = TensileReportConfig(
                supercell=(nx, ny, nz),
                temperature=args.temperature,
                potential_name=_pot,
                equilibration_steps=args.equil_steps if args.equil_steps else wf_config.equilibration_steps,
                timestep=args.timestep if args.timestep else wf_config.timestep,
                thermo_every=args.thermo_every if args.thermo_every else wf_config.thermo_every,
                dump_every=args.dump_every,
                deform_direction="z",
                strain_rate=args.strain_rate,
                lammps_command=_lmp,
            )
        elif args.workflow == "nanoindent":
            workflow._report_config = NanoindentReportConfig(
                supercell=(nx, ny, nz),
                temperature=args.temperature,
                potential_name=_pot,
                lammps_command=_lmp,
            )
        elif args.workflow == "crack":
            workflow._report_config = CrackReportConfig(
                supercell=(nx, ny, nz),
                temperature=args.temperature,
                potential_name=_pot,
                lammps_command=_lmp,
            )
        elif args.workflow == "dbtt":
            workflow._report_config = DBTTReportConfig(
                temperatures=workflow.temperatures,
                supercell=(nx, ny, nz),
                potential_name=_pot,
                strain_rate=args.strain_rate,
                lammps_command=_lmp,
            )
    except Exception:
        pass  # non-critical

    pipeline_result = workflow.run()

    payload = {
        "workflow": args.workflow,
        "success": pipeline_result.success,
        "failed_stage": pipeline_result.failed_stage,
        "stages": [
            {
                "name": name,
                "run_dir": str(output.run_dir),
                "return_code": output.lammps_result.return_code if output.lammps_result else None,
                "retry_count": output.retry_count,
            }
            for name, output in pipeline_result.stages
        ],
    }

    # Append report path to payload
    if pipeline_result.success:
        from pathlib import Path as _Path
        _report_map = {
            "tensile":     ("deform",  "tensile_report.md"),
            "nanoindent":  ("indent",  "nanoindent_report.md"),
            "crack":       ("load",    "crack_report.md"),
        }
        if args.workflow in _report_map:
            stage_key, report_file = _report_map[args.workflow]
            stage_dir = next(
                (str(o.run_dir) for n, o in pipeline_result.stages if n == stage_key), None
            )
            if stage_dir:
                rp = _Path(stage_dir) / report_file
                payload["report"] = str(rp) if rp.exists() else "未生成"
        elif args.workflow == "dbtt" and pipeline_result.stages:
            # DBTT report lives one level up from the first deform stage
            first_deform = next(
                (str(o.run_dir) for n, o in pipeline_result.stages if n.endswith("_deform")), None
            )
            if first_deform:
                rp = _Path(first_deform).parent / "dbtt_report.md"
                payload["report"] = str(rp) if rp.exists() else "未生成"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if pipeline_result.success else 1


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.workflow:
            return _run_workflow(args)
        elif args.task:
            return _run_single_task(args)
        else:
            print("Error: either --task or --workflow is required.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

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

    agent = LammpsAgent(config)

    nx, ny, nz = (int(x) for x in args.supercell.split(","))
    wf_config = WorkflowConfig(
        temperature=args.temperature,
        potential_name=args.potential or "W_Zhou2004_EAM",
        mpi_np=args.mpi_np,
        timeout_per_stage=args.timeout,
    )

    workflow = build_workflow(
        name=args.workflow,
        config=wf_config,
        agent=agent,
        supercell=(nx, ny, nz),
        strain_rate=args.strain_rate,
    )
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

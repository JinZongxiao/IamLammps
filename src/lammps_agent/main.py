from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .project_resources import initialize_project, run_doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coding-first conversational agent for LAMMPS simulations"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="start the interactive shell even when stdin/stdout are not TTYs",
    )
    commands = parser.add_subparsers(dest="command")
    init_parser = commands.add_parser(
        "init", help="initialize an IamLAMMPS workspace from packaged templates"
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="workspace to initialize (default: current directory)",
    )
    doctor_parser = commands.add_parser(
        "doctor", help="check local configuration and simulation prerequisites"
    )
    doctor_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="workspace to inspect (default: current directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "init":
            result = initialize_project(args.directory)
            print(f"Initialized IamLAMMPS workspace: {result.root}")
            for path in result.created:
                print(f"  created {path.as_posix()}")
            for path in result.skipped:
                print(f"  kept    {path.as_posix()}")
            return 0
        if args.command == "doctor":
            root = Path(args.directory).expanduser().resolve(strict=False)
            print(f"IamLAMMPS doctor: {root}")
            checks = run_doctor(args.directory)
            for check in checks:
                label = "PASS" if check.ok else "FAIL"
                print(f"  [{label}] {check.name}: {check.detail}")
            passed = sum(check.ok for check in checks)
            print(f"{passed}/{len(checks)} checks passed")
            return 0 if passed == len(checks) else 1
        if args.interactive or (sys.stdin.isatty() and sys.stdout.isatty()):
            from .interactive import InteractiveShell

            return InteractiveShell(_load_config()).run()
        else:
            print(
                "Error: stdin is not a terminal. "
                "Use --interactive to start the shell.",
                file=sys.stderr,
            )
            return 2
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


def _load_config():
    from .config import load_config

    return load_config()


if __name__ == "__main__":
    raise SystemExit(main())

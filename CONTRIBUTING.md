# Contributing

IamLAMMPS has one product path: the OpenAI Agents SDK interactive simulation agent. Do not
add a parallel workflow runtime or provider-specific copies of orchestration behavior.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Keep changes cohesive, validate model-controlled input at subsystem boundaries, preserve
error context, and add focused success and failure tests. Do not commit `.env`, MCP secrets,
session state, downloaded potentials, or generated execution output.

Before submitting:

```bash
python -m pytest --cov=lammps_agent --cov-report=term-missing --cov-fail-under=85
python -m ruff check src tests
python -m mypy src/lammps_agent
python -m compileall -q src tests
python -m build
git diff --check
```

Run the non-skipping Linux SPC/E integration job when changing LAMMPS execution, script
policy, snapshots, analyzers, or scientific examples. A process smoke test is also required
on macOS. Scientific changes must state units, assumptions, potential compatibility,
sampling limitations, and the difference between generated code and executed evidence.

Public API breaks require a migration note and tests. v4.9 workflow imports and the
`--task`/`--workflow` CLI are intentionally removed and must not be restored as shims.

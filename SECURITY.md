# Security Policy

IamLAMMPS writes files, starts local processes, and can access approved external services.
Its controls are application-level guardrails, not an operating-system sandbox.

## Supported versions

Security fixes are applied to the latest v5 release and current `main`. The archived v4.9
line is retained for migration reference and does not receive security fixes.

## Report a vulnerability

Do not file a public issue for credential exposure, unapproved execution, workspace escape,
or private-network access. Use GitHub private vulnerability reporting when available, or
contact the repository owner privately. Include the affected commit, a reproduction using
fake secrets, expected versus observed behavior, impact, and any proposed mitigation.

## Trust boundaries

- Model output is untrusted and must pass typed validation, policy, budgets, and approval.
- Workspace and state writes reject symlinks and traversal. Session SQLite data and terminal
  history live in `.iamlammps/state/`, outside the model-visible run workspace.
- LAMMPS executes an immutable copy of validated input. Its child receives a minimal
  environment without model API keys.
- Run manifests and artifact paths are workspace-relative. Failed and timed-out evidence is
  retained rather than silently replaced by a later attempt.
- Model-facing generic file tools reject writes to managed execution records and latest
  pointers. Deterministic plotting writes only through its dedicated per-run artifact path.
- Python analysis requires explicit approval and can access the host as the current user.
  Use a container or scheduler sandbox for untrusted analysis.
- Project Skill `recommended-tools` values are advisory. MCP `allowed_tools` values are
  enforced, and MCP startup and tool calls have separate approvals.
- HTTP destinations must resolve to public addresses and remain bound to the validated peer.
  Remote HTTP MCP is disabled where peer revalidation is unavailable.
- Potential hashes establish artifact identity, not scientific suitability.

Never run IamLAMMPS as root. On shared systems, use an unprivileged container, restrict
mounts and networking, and enforce CPU, memory, process, and wall-time limits.

This project is distributed under the Apache License 2.0. The warranty disclaimer in the
[LICENSE](LICENSE) applies.

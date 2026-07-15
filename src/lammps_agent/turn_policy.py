"""Per-user-turn action budgets and fail-closed stop conditions.

The policy is intentionally independent from the Agents SDK.  The SDK remains
the sole orchestrator; callers consult this small state machine before resuming
an interrupted run and report tool outcomes after execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ActionKind(StrEnum):
    """Side-effect classes governed by a turn budget."""

    FILE_CHANGE = "file_change"
    LAMMPS_RUN = "lammps_run"
    PYTHON_RUN = "python_run"
    EXTERNAL = "external"


class PolicyCode(StrEnum):
    """Stable machine-readable reasons for a denied action."""

    TURN_REJECTED = "TURN_REJECTED"
    FILE_CHANGE_BUDGET = "FILE_CHANGE_BUDGET"
    LAMMPS_RUN_BUDGET = "LAMMPS_RUN_BUDGET"
    PYTHON_RUN_BUDGET = "PYTHON_RUN_BUDGET"
    SDK_TURN_BUDGET = "SDK_TURN_BUDGET"
    RUN_TIMED_OUT = "RUN_TIMED_OUT"
    REPEATED_RUN_FAILURE = "REPEATED_RUN_FAILURE"


@dataclass(frozen=True, slots=True)
class TurnLimits:
    """Maximum side effects permitted during one user turn."""

    file_changes: int = 4
    lammps_runs: int = 2
    python_runs: int = 1
    sdk_turns: int = 12

    def __post_init__(self) -> None:
        if min(
            self.file_changes,
            self.lammps_runs,
            self.python_runs,
            self.sdk_turns,
        ) < 1:
            raise ValueError("turn limits must be positive")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result returned by policy checks without performing an action."""

    allowed: bool
    code: PolicyCode | None = None
    message: str = ""


class TurnPolicy:
    """Track approval, execution, and failure state for one user turn.

    A rejection is terminal for side effects in the current turn.  This is
    deliberately stronger than comparing paths or tool arguments: changing a
    filename, shortening a script, or switching tools cannot turn a rejected
    operation into a new approval request.
    """

    def __init__(self, limits: TurnLimits | None = None) -> None:
        self.limits = limits or TurnLimits()
        self.file_changes = 0
        self.lammps_runs = 0
        self.python_runs = 0
        self.sdk_turns = 0
        self.consecutive_run_failures = 0
        self._stop: PolicyDecision | None = None

    @property
    def stopped(self) -> bool:
        return self._stop is not None

    @property
    def rejected(self) -> bool:
        """Whether a user rejection latched this turn closed."""

        return self._stop is not None and self._stop.code is PolicyCode.TURN_REJECTED

    @property
    def stop_decision(self) -> PolicyDecision | None:
        return self._stop

    def check_action(self, kind: ActionKind) -> PolicyDecision:
        """Return whether another action of *kind* may start."""

        if self._stop is not None:
            return self._stop
        used, limit, code = self._budget(kind)
        if used >= limit:
            return PolicyDecision(
                False,
                code,
                f"{kind.value} budget exhausted ({used}/{limit})",
            )
        return PolicyDecision(True)

    @classmethod
    def new_turn(cls, limits: TurnLimits | None = None) -> TurnPolicy:
        """Construct fresh policy state for one user message."""

        return cls(limits)

    def reset(self) -> None:
        """Reset this object for a new user turn while retaining its limits."""

        self.file_changes = 0
        self.lammps_runs = 0
        self.python_runs = 0
        self.sdk_turns = 0
        self.consecutive_run_failures = 0
        self._stop = None

    def approval_decision(
        self, tool_name: str, arguments: object | None = None
    ) -> PolicyDecision:
        """Check a tool request before displaying or granting approval."""

        del arguments  # Reserved for future argument-specific hard limits.
        kind = action_kind_for_tool(tool_name)
        return PolicyDecision(True) if kind is None else self.check_action(kind)

    def reserve(self, tool_name: str, arguments: object | None = None) -> PolicyDecision:
        """Reserve budget immediately before approving a side-effecting tool."""

        del arguments
        kind = action_kind_for_tool(tool_name)
        return PolicyDecision(True) if kind is None else self.record_action_started(kind)

    def record_result(self, tool_name: str, result: Any) -> PolicyDecision:
        """Interpret process-tool output and update retry stop conditions."""

        kind = action_kind_for_tool(tool_name)
        if kind not in {ActionKind.LAMMPS_RUN, ActionKind.PYTHON_RUN}:
            return self._stop or PolicyDecision(True)
        if isinstance(result, str):
            try:
                import json

                result = json.loads(result)
            except (TypeError, ValueError):
                return self.record_run_result(succeeded=False)
        if not isinstance(result, dict):
            return self.record_run_result(succeeded=False)
        timed_out = bool(result.get("timed_out")) or result.get("status") == "timed_out"
        status = result.get("status")
        succeeded = status == "succeeded" or (
            status is None and result.get("return_code") == 0 and not timed_out
        )
        return self.record_run_result(succeeded=succeeded, timed_out=timed_out)

    def record_action_started(self, kind: ActionKind) -> PolicyDecision:
        """Consume one action budget, returning a denial without over-counting."""

        decision = self.check_action(kind)
        if not decision.allowed:
            return decision
        if kind is ActionKind.FILE_CHANGE:
            self.file_changes += 1
        elif kind is ActionKind.LAMMPS_RUN:
            self.lammps_runs += 1
        elif kind is ActionKind.PYTHON_RUN:
            self.python_runs += 1
        return decision

    def record_rejection(self) -> PolicyDecision:
        """Stop all further side effects after a user rejects one action."""

        return self._stop_with(
            PolicyCode.TURN_REJECTED,
            "the user rejected an action; no further side effects are allowed this turn",
        )

    def record_sdk_turn(self) -> PolicyDecision:
        """Consume one model turn before invoking the SDK runner."""

        if self._stop is not None:
            return self._stop
        if self.sdk_turns >= self.limits.sdk_turns:
            return self._stop_with(
                PolicyCode.SDK_TURN_BUDGET,
                f"SDK turn budget exhausted ({self.sdk_turns}/{self.limits.sdk_turns})",
            )
        self.sdk_turns += 1
        return PolicyDecision(True)

    def record_run_result(self, *, succeeded: bool, timed_out: bool = False) -> PolicyDecision:
        """Update retry state after a LAMMPS or Python process finishes."""

        if timed_out:
            return self._stop_with(
                PolicyCode.RUN_TIMED_OUT,
                "an execution timed out; no retry is allowed this turn",
            )
        if succeeded:
            self.consecutive_run_failures = 0
            return PolicyDecision(True)
        self.consecutive_run_failures += 1
        if self.consecutive_run_failures >= 2:
            return self._stop_with(
                PolicyCode.REPEATED_RUN_FAILURE,
                "two consecutive executions failed; no retry is allowed this turn",
            )
        return PolicyDecision(True)

    def _budget(self, kind: ActionKind) -> tuple[int, int, PolicyCode]:
        if kind is ActionKind.FILE_CHANGE:
            return self.file_changes, self.limits.file_changes, PolicyCode.FILE_CHANGE_BUDGET
        if kind is ActionKind.LAMMPS_RUN:
            return self.lammps_runs, self.limits.lammps_runs, PolicyCode.LAMMPS_RUN_BUDGET
        if kind is ActionKind.PYTHON_RUN:
            return self.python_runs, self.limits.python_runs, PolicyCode.PYTHON_RUN_BUDGET
        # External actions always require approval but do not have a numeric
        # budget in the v5 policy.
        return 0, 1, PolicyCode.TURN_REJECTED

    def _stop_with(self, code: PolicyCode, message: str) -> PolicyDecision:
        if self._stop is None:
            self._stop = PolicyDecision(False, code, message)
        return self._stop


def action_kind_for_tool(tool_name: str) -> ActionKind | None:
    """Classify built-in tools; unknown names are external and fail closed."""

    normalized = tool_name.casefold()
    if normalized in {"write_file", "replace_file", "stage_local_potential"}:
        return ActionKind.FILE_CHANGE
    if normalized == "run_lammps_input":
        return ActionKind.LAMMPS_RUN
    if normalized == "run_python_analysis":
        return ActionKind.PYTHON_RUN
    if normalized in {
        "search_potentials",
        "install_potential",
        "web_search",
        "fetch_web_result",
    }:
        return ActionKind.EXTERNAL
    if normalized in {
        "list_files",
        "read_file",
        "parse_thermo",
        "analyze_run",
        "list_local_potentials",
        "list_skills",
        "load_skill",
        "list_runs",
        "inspect_run",
        "plot_run",
    }:
        return None
    return ActionKind.EXTERNAL

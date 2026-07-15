from __future__ import annotations

from lammps_agent.turn_policy import (
    ActionKind,
    PolicyCode,
    TurnLimits,
    TurnPolicy,
    action_kind_for_tool,
)


def test_default_action_budgets_are_enforced_without_overcounting() -> None:
    policy = TurnPolicy()

    for _ in range(4):
        assert policy.record_action_started(ActionKind.FILE_CHANGE).allowed
    denied = policy.record_action_started(ActionKind.FILE_CHANGE)

    assert denied.code is PolicyCode.FILE_CHANGE_BUDGET
    assert policy.file_changes == 4


def test_user_rejection_stops_path_and_tool_substitution() -> None:
    policy = TurnPolicy()
    policy.record_action_started(ActionKind.FILE_CHANGE)

    rejected = policy.record_rejection()

    assert rejected.code is PolicyCode.TURN_REJECTED
    assert policy.check_action(ActionKind.FILE_CHANGE) == rejected
    assert policy.check_action(ActionKind.LAMMPS_RUN) == rejected
    assert policy.check_action(ActionKind.EXTERNAL) == rejected


def test_timeout_and_two_consecutive_failures_stop_retries() -> None:
    timeout = TurnPolicy()
    decision = timeout.record_run_result(succeeded=False, timed_out=True)
    assert decision.code is PolicyCode.RUN_TIMED_OUT

    failures = TurnPolicy()
    assert failures.record_run_result(succeeded=False).allowed
    stopped = failures.record_run_result(succeeded=False)
    assert stopped.code is PolicyCode.REPEATED_RUN_FAILURE

    reset = TurnPolicy()
    assert reset.record_run_result(succeeded=False).allowed
    assert reset.record_run_result(succeeded=True).allowed
    assert reset.record_run_result(succeeded=False).allowed


def test_sdk_turn_budget_is_terminal() -> None:
    policy = TurnPolicy(TurnLimits(sdk_turns=2))

    assert policy.record_sdk_turn().allowed
    assert policy.record_sdk_turn().allowed
    assert policy.record_sdk_turn().code is PolicyCode.SDK_TURN_BUDGET
    assert policy.sdk_turns == 2


def test_tool_classification_fails_closed_for_unknown_tools() -> None:
    assert action_kind_for_tool("write_file") is ActionKind.FILE_CHANGE
    assert action_kind_for_tool("run_lammps_input") is ActionKind.LAMMPS_RUN
    assert action_kind_for_tool("run_python_analysis") is ActionKind.PYTHON_RUN
    assert action_kind_for_tool("read_file") is None
    assert action_kind_for_tool("mcp_materials_delete") is ActionKind.EXTERNAL


def test_integration_api_reserves_by_tool_and_interprets_results() -> None:
    policy = TurnPolicy.new_turn()

    assert policy.approval_decision("run_lammps_input", {"input_path": "a"}).allowed
    assert policy.reserve("run_lammps_input", {"input_path": "a"}).allowed
    assert policy.lammps_runs == 1
    assert policy.record_result("run_lammps_input", '{"return_code": 1}').allowed
    stopped = policy.record_result(
        "run_lammps_input", {"status": "failed", "return_code": 2}
    )
    assert stopped.code is PolicyCode.REPEATED_RUN_FAILURE

    policy.reset()
    assert not policy.stopped
    assert not policy.rejected
    assert policy.lammps_runs == 0
    policy.record_rejection()
    assert policy.rejected

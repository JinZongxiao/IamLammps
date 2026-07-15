"""Focused lifecycle tests for the SDK-native simulation coding agent."""

from __future__ import annotations

import asyncio
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import ModelSettings

from lammps_agent.config import AgentConfig
from lammps_agent.mcp_config import MCPConfig
from lammps_agent.simulation_agent import (
    SYSTEM_PROMPT,
    SimulationCodingAgent,
    _private_state_directory,
    _reject_symlink,
    _tool_result_summary,
    _ToolHooks,
)
from lammps_agent.turn_policy import PolicyCode, TurnLimits, TurnPolicy


class _RunState:
    def __init__(self) -> None:
        self.approved: list[object] = []
        self.rejected: list[tuple[object, str]] = []

    def approve(self, item: object) -> None:
        self.approved.append(item)

    def reject(self, item: object, *, rejection_message: str) -> None:
        self.rejected.append((item, rejection_message))


class _Result:
    def __init__(
        self,
        output: object,
        *,
        interruptions: list[object] | None = None,
        state: _RunState | None = None,
    ) -> None:
        self.final_output = output
        self.interruptions = interruptions or []
        self._state = state

    def to_state(self) -> _RunState:
        assert self._state is not None
        return self._state


def _bare_agent() -> SimulationCodingAgent:
    """Build only the state run_turn needs, avoiding model and MCP setup."""
    agent = object.__new__(SimulationCodingAgent)
    agent.agent = object()  # type: ignore[assignment]
    agent.context = object()  # type: ignore[assignment]
    agent.session = object()  # type: ignore[assignment]
    agent.hooks = object()  # type: ignore[assignment]
    agent.run_config = object()  # type: ignore[assignment]
    return agent


def _config() -> AgentConfig:
    return AgentConfig(
        llm_provider="deepseek",
        openai_base_url="https://api.openai.test/v1",
        openai_api_key="",
        openai_model="unused",
        lammps_command="lmp",
        enabled_analyzers=[],
        deepseek_api_key="test-key",
        openai_tracing=False,
    )


def test_system_prompt_keeps_identity_questions_conversational() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "answer conversationally" in prompt
    assert "Do not create or run a simulation" in prompt
    assert "run_lammps_input" in prompt
    assert "generic shell" in prompt


def test_plain_chat_returns_model_reply_without_approval_or_second_run() -> None:
    agent = _bare_agent()
    approval_handler = AsyncMock(return_value=True)
    completed = _Result("我是 IamLAMMPS，可以聊天，也可以协助模拟。")

    with patch(
        "lammps_agent.simulation_agent.Runner.run",
        new=AsyncMock(return_value=completed),
    ) as run:
        reply = asyncio.run(
            agent.run_turn(
                "你是什么",
                mode="ask",
                approval_handler=approval_handler,
            )
        )

    assert reply == "我是 IamLAMMPS，可以聊天，也可以协助模拟。"
    assert run.await_count == 1
    approval_handler.assert_not_awaited()
    first_call = run.await_args_list[0]
    assert first_call.args[1] == "你是什么"
    assert first_call.kwargs["context"] is agent.context


@pytest.mark.parametrize(
    ("mode", "tool_name", "handler_result", "expected_approved"),
    [
        ("ask", "write_file", True, True),
        ("ask", "write_file", False, False),
        ("auto", "run_lammps_input", False, False),
        ("auto", "install_potential", True, True),
        ("auto", "run_python_analysis", False, False),
        ("read-only", "write_file", True, False),
    ],
)
def test_interruption_resumes_same_state_with_mode_approval_policy(
    mode: str,
    tool_name: str,
    handler_result: bool,
    expected_approved: bool,
) -> None:
    agent = _bare_agent()
    item = SimpleNamespace(tool_name=tool_name)
    state = _RunState()
    interrupted = _Result(None, interruptions=[item], state=state)
    completed = _Result("done")
    approval_handler = AsyncMock(return_value=handler_result)

    with patch(
        "lammps_agent.simulation_agent.Runner.run",
        new=AsyncMock(side_effect=[interrupted, completed]),
    ) as run:
        output = asyncio.run(
            agent.run_turn(
                "do work",
                mode=mode,  # type: ignore[arg-type]
                approval_handler=approval_handler,
            )
        )

    assert output == "done"
    assert run.await_count == 2
    assert run.await_args_list[1].args[1] is state
    assert "context" not in run.await_args_list[1].kwargs
    if expected_approved:
        assert state.approved == [item]
        assert state.rejected == []
    else:
        assert state.approved == []
        expected_message = (
            "Rejected by read-only mode"
            if mode == "read-only"
            else "User declined this action"
        )
        assert state.rejected == [(item, expected_message)]

    handler_should_run = mode == "ask" or (
        mode == "auto" and tool_name not in agent.AUTO_APPROVED_TOOLS
    )
    assert approval_handler.await_count == int(handler_should_run)


def test_non_string_final_output_is_rendered_for_cli() -> None:
    agent = _bare_agent()
    with patch(
        "lammps_agent.simulation_agent.Runner.run",
        new=AsyncMock(return_value=_Result({"status": "ok"})),
    ):
        output = asyncio.run(
            agent.run_turn(
                "status",
                mode="ask",
                approval_handler=AsyncMock(return_value=False),
            )
        )

    assert output == "{'status': 'ok'}"


def test_rejection_cancels_other_pending_side_effects() -> None:
    agent = _bare_agent()
    first = SimpleNamespace(tool_name="write_file")
    second = SimpleNamespace(tool_name="run_lammps_input")
    state = _RunState()
    interrupted = _Result(None, interruptions=[first, second], state=state)
    completed = _Result("Stopped after rejection.")
    approval_handler = AsyncMock(side_effect=[True, False])

    with patch(
        "lammps_agent.simulation_agent.Runner.run",
        new=AsyncMock(side_effect=[interrupted, completed]),
    ):
        output = asyncio.run(
            agent.run_turn("do both", mode="ask", approval_handler=approval_handler)
        )

    assert output == "Stopped after rejection."
    assert state.approved == []
    assert [item for item, _ in state.rejected] == [first, second]


def test_tool_result_summary_renders_structured_errors_as_failed() -> None:
    assert _tool_result_summary('{"code":"FILE_EXISTS","path":"runs/a.in"}') == (
        '{"path": "runs/a.in", "status": "failed"}'
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("plain result", "completed"),
        ("execution failed badly", "failed"),
        (object(), "completed"),
        ("[]", "completed"),
        (
            {"return_code": 2, "run_id": "run-1", "timed_out": False},
            '{"return_code": 2, "run_id": "run-1", "status": "failed", '
            '"timed_out": false}',
        ),
    ],
)
def test_tool_result_summary_handles_plain_and_structured_results(
    result: object, expected: str
) -> None:
    assert _tool_result_summary(result) == expected


def test_tool_hooks_report_events_and_latch_timeout_policy() -> None:
    events: list[tuple[str, str]] = []
    hooks = _ToolHooks(lambda name, details: events.append((name, details)))
    hooks.policy = TurnPolicy()
    tool = SimpleNamespace(name="run_lammps_input")

    async def exercise() -> None:
        await hooks.on_llm_start(None, None, None, None)
        await hooks.on_tool_start(None, None, tool)
        await hooks.on_tool_end(
            None,
            None,
            tool,
            '{"status":"timed_out","return_code":-1,"artifact_dir":"runs/a"}',
        )

    asyncio.run(exercise())

    assert hooks.policy.sdk_turns == 1
    assert hooks.policy.stop_decision is not None
    assert hooks.policy.stop_decision.code is PolicyCode.RUN_TIMED_OUT
    assert events[0] == ("run_lammps_input", "started")
    assert '"status": "timed_out"' in events[1][1]


def test_tool_hooks_enforce_sdk_turn_budget() -> None:
    hooks = _ToolHooks(None)
    hooks.policy = TurnPolicy(TurnLimits(sdk_turns=1))

    async def exercise() -> None:
        await hooks.on_llm_start(None, None, None, None)
        with pytest.raises(RuntimeError, match="SDK turn budget exhausted"):
            await hooks.on_llm_start(None, None, None, None)
        await hooks.on_llm_start(None, None, None, None)

    asyncio.run(exercise())


def test_mcp_startup_fails_closed_without_preapproval() -> None:
    agent = object.__new__(SimulationCodingAgent)
    agent._entered = False
    agent._approved_mcp_names = frozenset()
    agent._mcp_manager = None
    agent._mcp_preflight_failed = []
    agent._mcp_config = MCPConfig.model_validate(
        {
            "servers": {
                "local": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["server"],
                    "allowed_tools": ["read"],
                }
            }
        }
    )
    agent.agent = SimpleNamespace(mcp_servers=[object()])
    agent.session = SimpleNamespace(close=lambda: None)

    asyncio.run(agent.__aenter__())

    assert agent.agent.mcp_servers == []
    assert agent._mcp_manager is None
    assert agent.mcp_status()["denied"] == ["local"]
    asyncio.run(agent.__aexit__(None, None, None))


def test_mcp_approval_rejects_unknown_names_and_freezes_loaded_config() -> None:
    agent = object.__new__(SimulationCodingAgent)
    agent._entered = False
    agent._mcp_config = MCPConfig.model_validate(
        {
            "servers": {
                "local": {
                    "transport": "stdio",
                    "command": "uvx",
                    "allowed_tools": ["read"],
                }
            }
        }
    )
    agent._approved_mcp_names = frozenset()
    agent._mcp_preflight_failed = []

    with pytest.raises(ValueError, match="Unknown or disabled"):
        agent.approve_mcp_startup({"other"})
    agent.approve_mcp_startup({"local"})

    assert agent._approved_mcp_names == frozenset({"local"})


def test_failed_http_mcp_preflight_does_not_abort_the_agent() -> None:
    agent = object.__new__(SimulationCodingAgent)
    agent._entered = False
    agent._approved_mcp_names = frozenset({"remote"})
    agent._mcp_manager = None
    agent._mcp_preflight_failed = []
    agent._mcp_config = MCPConfig.model_validate(
        {
            "servers": {
                "remote": {
                    "transport": "http",
                    "url": "https://example.test/mcp",
                    "allowed_tools": ["search"],
                }
            }
        }
    )
    agent.agent = SimpleNamespace(mcp_servers=[object()])
    agent.session = SimpleNamespace(close=lambda: None)

    with patch(
        "lammps_agent.simulation_agent.validate_mcp_http_destinations",
        side_effect=ValueError("blocked destination"),
    ):
        asyncio.run(agent.__aenter__())

    assert agent.agent.mcp_servers == []
    assert agent.mcp_status()["failed"] == ["remote"]
    asyncio.run(agent.__aexit__(None, None, None))


def test_real_initialization_uses_private_state_and_closes_client(tmp_path) -> None:
    client = SimpleNamespace(close=AsyncMock())

    with (
        patch(
            "lammps_agent.simulation_agent.build_agent_model",
            return_value=("deepseek-test", ModelSettings(), client),
        ),
        patch("lammps_agent.simulation_agent.build_simulation_tools", return_value=[]),
    ):
        agent = SimulationCodingAgent(
            _config(), project_root=tmp_path, session_id="session-test"
        )

    database = tmp_path / ".iamlammps" / "state" / "agent_sessions.db"
    assert agent.context.workspace.root == tmp_path
    assert agent.session_id == "session-test"
    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    assert database.is_file()
    assert stat.S_IMODE(database.stat().st_mode) == 0o600

    async def exercise() -> None:
        assert await agent.__aenter__() is agent
        with pytest.raises(RuntimeError, match="already running"):
            await agent.__aenter__()
        assert await agent.__aexit__(None, None, None) is None

    asyncio.run(exercise())
    client.close.assert_awaited_once()
    assert agent._entered is False


def test_approved_stdio_mcp_connects_and_closes_all_resources() -> None:
    agent = object.__new__(SimulationCodingAgent)
    agent._entered = False
    agent._approved_mcp_names = frozenset({"local"})
    agent._mcp_manager = None
    agent._mcp_preflight_failed = []
    agent._mcp_config = MCPConfig.model_validate(
        {
            "servers": {
                "local": {
                    "transport": "stdio",
                    "command": "uvx",
                    "allowed_tools": ["read"],
                }
            }
        }
    )
    model = SimpleNamespace(close=AsyncMock())
    agent.agent = SimpleNamespace(mcp_servers=[], model=model)
    agent.session = SimpleNamespace(close=Mock())
    agent._openai_client = SimpleNamespace(close=AsyncMock())
    active_server = SimpleNamespace(name="local")
    manager = SimpleNamespace(
        active_servers=[active_server],
        failed_servers=[],
        __aenter__=AsyncMock(),
        __aexit__=AsyncMock(return_value=False),
    )
    manager.__aenter__.return_value = manager

    async def exercise() -> None:
        with (
            patch(
                "lammps_agent.simulation_agent.build_mcp_servers",
                return_value=[active_server],
            ),
            patch(
                "lammps_agent.simulation_agent.MCPServerManager",
                return_value=manager,
            ),
        ):
            assert await agent.__aenter__() is agent
        assert agent.agent.mcp_servers == [active_server]
        assert agent.mcp_status()["active"] == ["local"]
        assert await agent.__aexit__(None, None, None) is False

    asyncio.run(exercise())
    manager.__aexit__.assert_awaited_once()
    agent.session.close.assert_called_once()
    model.close.assert_awaited_once()
    agent._openai_client.close.assert_awaited_once()


def test_mcp_start_failure_closes_session_and_client() -> None:
    agent = object.__new__(SimulationCodingAgent)
    agent._entered = False
    agent._approved_mcp_names = frozenset({"local"})
    agent._mcp_manager = None
    agent._mcp_preflight_failed = []
    agent._mcp_config = MCPConfig.model_validate(
        {
            "servers": {
                "local": {
                    "transport": "stdio",
                    "command": "uvx",
                    "allowed_tools": ["read"],
                }
            }
        }
    )
    agent.agent = SimpleNamespace(mcp_servers=[])
    agent.session = SimpleNamespace(close=Mock())
    agent._openai_client = SimpleNamespace(close=AsyncMock())
    manager = SimpleNamespace(__aenter__=AsyncMock(side_effect=RuntimeError("boom")))

    async def exercise() -> None:
        with (
            patch("lammps_agent.simulation_agent.build_mcp_servers", return_value=[object()]),
            patch(
                "lammps_agent.simulation_agent.MCPServerManager",
                return_value=manager,
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await agent.__aenter__()

    asyncio.run(exercise())
    agent.session.close.assert_called_once()
    agent._openai_client.close.assert_awaited_once()
    assert agent._entered is False


def test_private_state_rejects_symlinks(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".iamlammps").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="must not be a symlink"):
        _private_state_directory(tmp_path)

    target = tmp_path / "database"
    target.write_text("db", encoding="utf-8")
    link = tmp_path / "database-link"
    link.symlink_to(target)
    with pytest.raises(OSError, match="must not be a symlink"):
        _reject_symlink(link)


def test_clear_session_and_turns_are_serialized() -> None:
    agent = _bare_agent()
    agent.session = SimpleNamespace(clear_session=AsyncMock())  # type: ignore[assignment]
    agent._turn_lock = asyncio.Lock()

    asyncio.run(agent.clear_session())

    agent.session.clear_session.assert_awaited_once()  # type: ignore[attr-defined]

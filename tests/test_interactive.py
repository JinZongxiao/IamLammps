"""Behavioral tests for the coding-agent terminal shell."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lammps_agent.config import AgentConfig
from lammps_agent.interactive import InteractiveShell, _approval_view
from lammps_agent.mcp_config import MCPStartupRequest


class FakeSkillRegistry:
    def discover(self):
        return (SimpleNamespace(name="thermo", description="Analyze thermo output"),)


class FakeCodingAgent:
    def __init__(self) -> None:
        self.session_id = "test-session"
        self.turns: list[tuple[str, str]] = []
        self.clear_count = 0
        self.approved_mcp: set[str] | None = None
        self.context = SimpleNamespace(skill_registry=FakeSkillRegistry())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def run_turn(self, message, *, mode, approval_handler):
        self.turns.append((message, mode))
        return f"reply: {message}"

    async def clear_session(self):
        self.clear_count += 1

    def mcp_startup_requests(self):
        return ()

    def approve_mcp_startup(self, names):
        self.approved_mcp = set(names)

    def mcp_status(self):
        return {
            "configured": ["search", "broken"],
            "approved": ["search", "broken"],
            "denied": [],
            "active": ["search"],
            "failed": ["broken"],
        }


def _config() -> AgentConfig:
    return AgentConfig(
        llm_provider="deepseek",
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="openai-secret",
        openai_model="gpt-test",
        lammps_command="lmp",
        enabled_analyzers=[],
        deepseek_api_key="deepseek-secret",
        deepseek_model="deepseek-v4-flash",
    )


def _input(lines: list[str], prompts: list[str]):
    remaining = iter(lines)

    def read(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(remaining)
        except StopIteration as exc:
            raise EOFError from exc

    return read


def _shell(tmp_path: Path, lines: list[str]):
    outputs: list[str] = []
    prompts: list[str] = []
    agent = FakeCodingAgent()
    shell = InteractiveShell(
        _config(),
        coding_agent=agent,  # type: ignore[arg-type]
        input_fn=_input(lines, prompts),
        output_fn=outputs.append,
        error_fn=outputs.append,
    )
    shell._load_history = lambda: None  # type: ignore[method-assign]
    shell._save_history = lambda: None  # type: ignore[method-assign]
    return shell, agent, outputs, prompts


def test_normal_question_is_a_conversational_turn(tmp_path: Path) -> None:
    shell, agent, output, prompts = _shell(tmp_path, ["你是什么", "/exit"])

    assert shell.run() == 0
    assert agent.turns == [("你是什么", "ask")]
    assert "reply: 你是什么" in output
    assert all("Approve" not in prompt for prompt in prompts)


def test_blank_and_local_commands_do_not_call_model(tmp_path: Path) -> None:
    shell, agent, output, _ = _shell(
        tmp_path,
        ["", "/help", "/status", "/mcp", "/skills", "/exit"],
    )

    assert shell.run() == 0
    assert agent.turns == []
    rendered = "\n".join(output)
    assert "/mode ask|auto|read-only" in rendered
    assert "provider=deepseek/deepseek-v4-flash" in rendered
    assert "configured=search, broken" in rendered
    assert "active=search" in rendered
    assert "thermo: Analyze thermo output" in rendered
    assert "deepseek-secret" not in rendered
    assert "openai-secret" not in rendered


def test_modes_are_forwarded_to_agent(tmp_path: Path) -> None:
    shell, agent, output, _ = _shell(
        tmp_path,
        [
            "/mode auto",
            "run it",
            "/mode read-only",
            "inspect it",
            "/exit",
        ],
    )

    assert shell.run() == 0
    assert agent.turns == [("run it", "auto"), ("inspect it", "read-only")]
    assert "Auto approves workspace edits" in "\n".join(output)
    assert "Python analysis" in "\n".join(output)
    assert "still requires confirmation" in "\n".join(output)


def test_generate_only_aliases_read_only(tmp_path: Path) -> None:
    shell, agent, _, _ = _shell(
        tmp_path, ["/mode generate-only", "inspect", "/exit"]
    )

    assert shell.run() == 0
    assert agent.turns == [("inspect", "read-only")]


def test_new_clears_sdk_session(tmp_path: Path) -> None:
    shell, agent, output, _ = _shell(tmp_path, ["/new", "/exit"])

    assert shell.run() == 0
    assert agent.clear_count == 1
    assert "Conversation context cleared." in output


def test_unknown_command_is_reported_and_loop_continues(tmp_path: Path) -> None:
    shell, agent, output, _ = _shell(tmp_path, ["/nope", "hello", "/exit"])

    assert shell.run() == 0
    assert agent.turns == [("hello", "ask")]
    assert "Error: Unknown command: /nope. Use /help." in output


def test_tool_approval_displays_bounded_action_summary(tmp_path: Path) -> None:
    shell, _, output, _ = _shell(tmp_path, ["y"])
    item = SimpleNamespace(
        tool_name="run_lammps_input",
        raw_item={
            "name": "run_lammps_input",
            "arguments": '{"input_path":"runs/a/in.lammps"}',
        },
    )

    assert shell.run_async  # keep type checkers aware this is an async shell
    import asyncio

    assert asyncio.run(shell._approve_tool(item)) is True  # type: ignore[arg-type]
    rendered = "\n".join(output)
    assert "Approval required: run_lammps_input" in rendered
    assert '"path": "runs/a/in.lammps"' in rendered


def test_mcp_startup_requires_separate_explicit_approval(tmp_path: Path) -> None:
    shell, agent, output, prompts = _shell(tmp_path, ["y", "/exit"])
    agent.mcp_startup_requests = lambda: (
        MCPStartupRequest(
            name="materials",
            transport="stdio",
            summary='{"command": "uvx", "env_keys": ["TOKEN"]}',
        ),
    )

    assert shell.run() == 0
    assert agent.approved_mcp == {"materials"}
    assert prompts[0] == "Start this MCP server? [y/N] "
    rendered = "\n".join(output)
    assert "MCP server startup approval required: materials" in rendered
    assert "Each later MCP tool call will still require approval" in rendered


def test_mcp_startup_defaults_to_denied(tmp_path: Path) -> None:
    shell, agent, _, _ = _shell(tmp_path, ["", "/exit"])
    agent.mcp_startup_requests = lambda: (
        MCPStartupRequest("remote", "streamable-http", '{"url": "https://example.test"}'),
    )

    assert shell.run() == 0
    assert agent.approved_mcp == set()


def test_read_only_rejects_tool_without_prompt(tmp_path: Path) -> None:
    shell, _, output, prompts = _shell(tmp_path, [])
    shell.mode = "read-only"
    item = SimpleNamespace(tool_name="write_file", raw_item={"arguments": "{}"})
    import asyncio

    assert asyncio.run(shell._approve_tool(item)) is False  # type: ignore[arg-type]
    assert prompts == []
    assert "Rejected by read-only mode." in output


def test_approval_view_supports_object_and_mapping_calls() -> None:
    mapping = SimpleNamespace(
        tool_name=None,
        raw_item={"name": "download", "arguments": '{"id":"p1"}'},
    )
    obj = SimpleNamespace(
        tool_name="run",
        raw_item=SimpleNamespace(name="run", arguments='{"path":"in.lmp"}'),
    )

    assert _approval_view(mapping) == ("download", '{"action": "download"}')
    assert _approval_view(obj) == (
        "run",
        '{"action": "run", "path": "in.lmp"}',
    )

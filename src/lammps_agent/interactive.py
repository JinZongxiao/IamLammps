"""Codex-style terminal session for the simulation coding agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from agents.items import ToolApprovalItem

from .config import AgentConfig
from .simulation_agent import ApprovalMode, SimulationCodingAgent

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


class InteractiveShell:
    """Long-lived conversational shell with SDK-native tool approvals."""

    MODES = {"ask", "auto", "read-only"}

    def __init__(
        self,
        config: AgentConfig,
        coding_agent: SimulationCodingAgent | None = None,
        *,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
        error_fn: OutputFn = print,
    ) -> None:
        self.config = config
        self.input = input_fn
        self.output = output_fn
        self.error = error_fn
        self.mode: ApprovalMode = "ask"
        self.coding_agent = coding_agent or SimulationCodingAgent(
            config,
            tool_event_handler=self._render_tool_event,
        )

    def run(self) -> int:
        """Run one async event loop for the full REPL and all MCP lifecycles."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> int:
        self._load_history()
        self.output("IamLAMMPS simulation coding agent")
        self.output(
            f"{self._provider_label()} | mode={self.mode} | cwd={Path.cwd()}"
        )
        self.output("Chat, design, edit, run, and analyze simulations. /help for commands.")

        try:
            approved_mcp: set[str] = set()
            for request in self.coding_agent.mcp_startup_requests():
                if self._approve_mcp_startup(request):
                    approved_mcp.add(request.name)
            self.coding_agent.approve_mcp_startup(approved_mcp)
            async with self.coding_agent:
                while True:
                    try:
                        line = self.input("iamlammps> ").strip()
                    except EOFError:
                        self.output("")
                        return 0
                    except KeyboardInterrupt:
                        self.output("\nInterrupted. Use /exit to quit.")
                        continue
                    if not line:
                        continue
                    try:
                        if line.startswith("/"):
                            if not await self._handle_command(line):
                                return 0
                        else:
                            response = await self.coding_agent.run_turn(
                                line,
                                mode=self.mode,
                                approval_handler=self._approve_tool,
                            )
                            if response.strip():
                                self.output(response)
                    except KeyboardInterrupt:
                        self.error("Interrupted.")
                    except Exception as exc:
                        self.error(f"Error: {exc}")
        finally:
            self._save_history()

    async def _handle_command(self, line: str) -> bool:
        command, _, argument = line.partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self.output(
                "/mode ask|auto|read-only  Set tool approval policy\n"
                "/new                      Clear conversation context\n"
                "/status                   Show model and runtime state\n"
                "/mcp                      Show connected MCP servers\n"
                "/skills                   List registered project skills\n"
                "/help                     Show commands\n"
                "/exit                     Leave the session"
            )
            return True
        if command == "/mode":
            if argument == "generate-only":
                argument = "read-only"
            if argument not in self.MODES:
                raise ValueError("Usage: /mode ask|auto|read-only")
            self.mode = cast(ApprovalMode, argument)
            self.output(f"Mode set to {argument}.")
            if argument == "auto":
                self.output(
                    "Auto approves workspace edits and local potential staging. "
                    "Every LAMMPS launch, Python analysis, network access, download, "
                    "and external MCP action still requires confirmation."
                )
            return True
        if command == "/new":
            await self.coding_agent.clear_session()
            self.output("Conversation context cleared.")
            return True
        if command == "/status":
            self.output(
                f"provider={self._provider_label()}\n"
                f"mode={self.mode}\n"
                f"lammps={self.config.lammps_command}\n"
                f"mpi={self.config.mpi_np or 1}\n"
                f"timeout={self.config.lammps_timeout or 'none'}\n"
                f"session={self.coding_agent.session_id}"
            )
            return True
        if command == "/mcp":
            status = self.coding_agent.mcp_status()
            configured = ", ".join(status["configured"]) or "none"
            approved = ", ".join(status["approved"]) or "none"
            denied = ", ".join(status["denied"]) or "none"
            active = ", ".join(status["active"]) or "none"
            failed = ", ".join(status["failed"]) or "none"
            self.output(
                f"configured={configured}\napproved={approved}\ndenied={denied}\n"
                f"active={active}\nfailed={failed}"
            )
            return True
        if command == "/skills":
            skills = self.coding_agent.context.skill_registry.discover()
            if not skills:
                self.output("No project skills registered.")
            else:
                self.output("\n".join(f"{s.name}: {s.description}" for s in skills))
            return True
        raise ValueError(f"Unknown command: {command}. Use /help.")

    async def _approve_tool(self, item: ToolApprovalItem) -> bool:
        name, arguments = _approval_view(item)
        self.output(f"Approval required: {name}")
        if arguments:
            self.output(arguments)
        if self.mode == "read-only":
            self.output("Rejected by read-only mode.")
            return False
        try:
            answer = self.input("Approve this action? [y/N] ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            self.output("")
            return False
        return answer in {"y", "yes"}

    def _approve_mcp_startup(self, request) -> bool:
        self.output(
            f"MCP server startup approval required: {request.name} "
            f"({request.transport})"
        )
        self.output(request.summary)
        self.output(
            "Approving may start a local process or connect to an external service. "
            "Each later MCP tool call will still require approval."
        )
        try:
            answer = self.input("Start this MCP server? [y/N] ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            self.output("")
            return False
        return answer in {"y", "yes"}

    def _render_tool_event(self, name: str, details: str) -> None:
        self.output(f"[{name}] {details}".rstrip())

    def _provider_label(self) -> str:
        if self.config.llm_provider.casefold() == "deepseek":
            return f"deepseek/{self.config.deepseek_model}"
        return f"openai/{self.config.openai_model}"

    @staticmethod
    def _history_path() -> Path:
        return Path(".iamlammps/state/history")

    def _load_history(self) -> None:
        try:
            import readline

            path = self._history_path()
            if path.exists():
                readline.read_history_file(path)
        except (ImportError, OSError):
            pass

    def _save_history(self) -> None:
        try:
            import readline

            readline.set_history_length(500)
            path = self._history_path()
            _prepare_private_state(path.parent)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                readline.write_history_file(temporary)
                temporary.chmod(0o600)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        except (ImportError, OSError):
            pass


def _approval_view(item: ToolApprovalItem) -> tuple[str, str]:
    raw = item.raw_item
    name = item.tool_name
    if isinstance(raw, dict):
        name = name or raw.get("name")
        arguments = raw.get("arguments") or raw.get("input") or ""
    else:
        name = name or getattr(raw, "name", None)
        arguments = getattr(raw, "arguments", "") or getattr(raw, "input", "")
    tool_name = str(name or "unknown_tool")
    return tool_name, _summarize_tool_action(tool_name, str(arguments or ""))


def _summarize_tool_action(name: str, arguments: str) -> str:
    """Render a bounded approval summary without echoing file content or secrets."""

    try:
        payload = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return arguments[:1_000]
    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=True)[:1_000]

    summary: dict[str, object] = {"action": name}
    path_key = next(
        (
            key
            for key in (
                "path",
                "input_path",
                "script_path",
                "run_directory",
                "destination_directory",
            )
            if key in payload
        ),
        None,
    )
    if path_key is not None:
        summary["path"] = payload[path_key]
    if name == "write_file":
        content = str(payload.get("content", ""))
        summary.update(
            {
                "operation": "update" if payload.get("expected_sha256") else "create",
                "size": len(content.encode("utf-8")),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    elif name == "replace_file":
        summary.update(
            {
                "operation": "update",
                "expected_sha256": payload.get("expected_sha256"),
            }
        )
    elif name == "run_lammps_input":
        summary.update(
            {
                "expected_sha256": payload.get("expected_sha256"),
                "timeout_seconds": payload.get("timeout_seconds"),
                "mpi_processes": payload.get("mpi_processes"),
            }
        )
    else:
        for key in ("run_id", "candidate_id", "potential_name", "query", "limit"):
            if key in payload:
                summary[key] = payload[key]
    return json.dumps(summary, ensure_ascii=True, sort_keys=True)


def _prepare_private_state(path: Path) -> None:
    current = Path()
    for part in path.parts:
        current /= part
        if current.exists() and current.is_symlink():
            raise OSError(f"private state path must not be a symlink: {current}")
        current.mkdir(exist_ok=True)
        current.chmod(0o700)

"""Validated project MCP configuration for the OpenAI Agents SDK."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from agents.mcp import MCPServerStdio, MCPServerStreamableHttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHELL_EXECUTABLES = {
    "bash",
    "cmd",
    "cmd.exe",
    "dash",
    "fish",
    "ksh",
    "powershell",
    "powershell.exe",
    "pwsh",
    "sh",
    "zsh",
}


class MCPConfigError(ValueError):
    """Raised when ``.iamlammps/mcp.json`` is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class MCPStartupRequest:
    """Redacted description of an external server that may be started."""

    name: str
    transport: str
    summary: str


class _ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allowed_tools: list[str] = Field(min_length=1)
    approval: Literal["always"] = "always"
    cache_tools: bool = True
    session_timeout_seconds: float = Field(default=10.0, gt=0, le=300)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, tools: list[str]) -> list[str]:
        if len(set(tools)) != len(tools):
            raise ValueError("allowed_tools must not contain duplicates")
        for tool in tools:
            if not _TOOL_NAME_PATTERN.fullmatch(tool):
                raise ValueError(f"invalid MCP tool name: {tool!r}")
        return tools


class StdioServerConfig(_ServerConfig):
    transport: Literal["stdio"]
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    @field_validator("command")
    @classmethod
    def validate_executable(cls, command: str) -> str:
        if command != command.strip() or any(char.isspace() for char in command):
            raise ValueError("command must contain one executable; put arguments in args")
        if Path(command).name.lower() in _SHELL_EXECUTABLES:
            raise ValueError("shell executables are not allowed as MCP commands")
        return command

    @field_validator("args")
    @classmethod
    def validate_args(cls, args: list[str]) -> list[str]:
        if any("\x00" in arg for arg in args):
            raise ValueError("MCP arguments must not contain NUL bytes")
        return args

    @field_validator("env")
    @classmethod
    def validate_env(cls, env: dict[str, str] | None) -> dict[str, str] | None:
        if env is not None and any(not key or "=" in key or "\x00" in key for key in env):
            raise ValueError("invalid environment variable name")
        if env is not None and any("\x00" in value for value in env.values()):
            raise ValueError("environment values must not contain NUL bytes")
        return env


class HttpServerConfig(_ServerConfig):
    transport: Literal["http", "streamable-http"]
    url: str = Field(min_length=1)
    headers: dict[str, str] | None = None
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=300)
    sse_read_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("MCP URL must be an absolute https URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("MCP URL must not contain credentials or a fragment")
        try:
            literal_address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            literal_address = None
        if literal_address is not None and not literal_address.is_global:
            raise ValueError("MCP URL must not target a private or local address")
        return url


ServerConfig = Annotated[
    StdioServerConfig | HttpServerConfig,
    Field(discriminator="transport"),
]


class MCPConfig(BaseModel):
    """Root model for ``.iamlammps/mcp.json``."""

    model_config = ConfigDict(extra="forbid")
    servers: dict[str, ServerConfig] = Field(default_factory=dict)

    @field_validator("servers")
    @classmethod
    def validate_server_names(
        cls, servers: dict[str, ServerConfig]
    ) -> dict[str, ServerConfig]:
        for name in servers:
            if not _SERVER_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"invalid MCP server name: {name!r}")
        return servers


def load_mcp_config(project_root: str | Path = ".") -> MCPConfig:
    """Load the optional project MCP config without starting any server."""

    path = Path(project_root).resolve() / ".iamlammps" / "mcp.json"
    if not path.exists():
        return MCPConfig()
    if path.is_symlink() or not path.is_file():
        raise MCPConfigError(f"MCP config must be a regular file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"Cannot read MCP config {path}: {exc}") from exc
    try:
        return MCPConfig.model_validate(raw)
    except ValidationError as exc:
        raise MCPConfigError(f"Invalid MCP config {path}: {exc}") from exc


def build_mcp_servers(
    config: MCPConfig,
    *,
    approved_names: set[str] | frozenset[str],
) -> list[MCPServerStdio | MCPServerStreamableHttp]:
    """Translate approved validated config into native Agents SDK MCP servers."""

    servers: list[MCPServerStdio | MCPServerStreamableHttp] = []
    for name, server_config in config.servers.items():
        if not server_config.enabled or name not in approved_names:
            continue
        common = {
            "name": name,
            "cache_tools_list": server_config.cache_tools,
            "client_session_timeout_seconds": server_config.session_timeout_seconds,
            "tool_filter": {"allowed_tool_names": list(server_config.allowed_tools)},
            "require_approval": server_config.approval,
        }
        if isinstance(server_config, StdioServerConfig):
            params: dict[str, object] = {
                "command": server_config.command,
                "args": list(server_config.args),
            }
            if server_config.env is not None:
                params["env"] = dict(server_config.env)
            if server_config.cwd is not None:
                params["cwd"] = server_config.cwd
            servers.append(MCPServerStdio(params, **common))  # type: ignore[arg-type]
            continue
        params = {
            "url": server_config.url,
            "timeout": server_config.request_timeout_seconds,
            "sse_read_timeout": server_config.sse_read_timeout_seconds,
        }
        if server_config.headers is not None:
            params["headers"] = dict(server_config.headers)
        servers.append(MCPServerStreamableHttp(params, **common))  # type: ignore[arg-type]
    return servers


def mcp_startup_requests(config: MCPConfig) -> tuple[MCPStartupRequest, ...]:
    """Return stable, redacted startup approval descriptions."""

    requests: list[MCPStartupRequest] = []
    for name, server in config.servers.items():
        if not server.enabled:
            continue
        if isinstance(server, StdioServerConfig):
            details = {
                "command": server.command,
                "args": server.args,
                "cwd": server.cwd,
                "env_keys": sorted((server.env or {}).keys()),
                "allowed_tools": server.allowed_tools,
            }
        else:
            details = {
                "url": server.url,
                "header_keys": sorted((server.headers or {}).keys()),
                "allowed_tools": server.allowed_tools,
            }
        requests.append(
            MCPStartupRequest(
                name=name,
                transport=server.transport,
                summary=json.dumps(details, ensure_ascii=True, sort_keys=True),
            )
        )
    return tuple(requests)


def validate_mcp_http_destinations(
    config: MCPConfig,
    approved_names: set[str] | frozenset[str],
) -> None:
    """Reject remote MCP until the transport can verify the connected peer.

    Resolving a hostname before connection is insufficient because DNS may change
    between validation and connect. The SDK's streamable HTTP transport currently
    does not expose the actual peer address, so stdio remains the supported path.
    """

    for name in approved_names:
        server = config.servers[name]
        if not isinstance(server, HttpServerConfig):
            continue
        raise MCPConfigError(
            f"Remote HTTP MCP server {name!r} is disabled: the connection layer "
            "cannot revalidate the actual peer address"
        )

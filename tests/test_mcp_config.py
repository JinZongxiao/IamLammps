from __future__ import annotations

import json
from pathlib import Path

import pytest

from lammps_agent.mcp_config import (
    MCPConfigError,
    build_mcp_servers,
    load_mcp_config,
    mcp_startup_requests,
)


def _write_config(root: Path, value: object) -> Path:
    path = root / ".iamlammps" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_missing_config_has_no_servers(tmp_path: Path) -> None:
    assert load_mcp_config(tmp_path).servers == {}


def test_loads_typed_stdio_and_http_configs(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "servers": {
                "search": {
                    "transport": "http",
                    "url": "https://mcp.example.test/api",
                    "headers": {"Authorization": "Bearer test"},
                    "allowed_tools": ["search", "fetch-result"],
                },
                "local-data": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["data-server", "--readonly"],
                    "allowed_tools": ["read_data"],
                    "approval": "always",
                },
            }
        },
    )

    config = load_mcp_config(tmp_path)

    assert config.servers["search"].approval == "always"
    assert config.servers["local-data"].approval == "always"


@pytest.mark.parametrize(
    "server, message",
    [
        (
            {"transport": "stdio", "command": "npx server", "allowed_tools": ["read"]},
            "one executable",
        ),
        (
            {
                "transport": "stdio",
                "command": "bash",
                "args": ["-c", "anything"],
                "allowed_tools": ["read"],
            },
            "shell executables",
        ),
        ({"transport": "stdio", "command": "uvx", "allowed_tools": []}, "too_short"),
        (
            {
                "transport": "http",
                "url": "file:///etc/passwd",
                "allowed_tools": ["read"],
            },
            "absolute http",
        ),
        (
            {
                "transport": "http",
                "url": "https://user:password@example.test/mcp",
                "allowed_tools": ["read"],
            },
            "credentials",
        ),
        (
            {
                "transport": "http",
                "url": "https://example.test/mcp",
                "allowed_tools": ["bad tool"],
            },
            "invalid MCP tool",
        ),
    ],
)
def test_rejects_unsafe_or_invalid_server_config(
    tmp_path: Path, server: dict[str, object], message: str
) -> None:
    _write_config(tmp_path, {"servers": {"bad": server}})

    with pytest.raises(MCPConfigError, match=message):
        load_mcp_config(tmp_path)


def test_rejects_invalid_json_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / ".iamlammps" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(MCPConfigError, match="Cannot read"):
        load_mcp_config(tmp_path)

    _write_config(tmp_path, {"servers": {}, "surprise": True})
    with pytest.raises(MCPConfigError, match="extra_forbidden"):
        load_mcp_config(tmp_path)


def test_builds_native_sdk_servers_with_filters_and_approval(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "servers": {
                "local": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["server"],
                    "allowed_tools": ["read_data"],
                },
                "remote": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                    "allowed_tools": ["search"],
                    "enabled": False,
                },
            }
        },
    )

    servers = build_mcp_servers(load_mcp_config(tmp_path), approved_names={"local"})

    assert len(servers) == 1
    server = servers[0]
    assert server.name == "local"
    assert server.params.command == "uvx"
    assert server.params.args == ["server"]
    assert server.tool_filter == {"allowed_tool_names": ["read_data"]}
    assert server._needs_approval_policy is True


def test_build_only_includes_explicitly_approved_servers(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "servers": {
                "first": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["first-server"],
                    "allowed_tools": ["read"],
                },
                "second": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["second-server"],
                    "allowed_tools": ["read"],
                },
            }
        },
    )

    servers = build_mcp_servers(
        load_mcp_config(tmp_path), approved_names={"second"}
    )

    assert [server.name for server in servers] == ["second"]


def test_startup_summary_redacts_secret_values_and_escapes_controls(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "servers": {
                "remote": {
                    "transport": "http",
                    "url": "https://example.test/mcp",
                    "headers": {"Authorization": "Bearer do-not-print"},
                    "allowed_tools": ["search"],
                },
                "local": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["server", "line\nvalue"],
                    "env": {"API_TOKEN": "do-not-print-either"},
                    "allowed_tools": ["read"],
                },
            }
        },
    )

    requests = mcp_startup_requests(load_mcp_config(tmp_path))
    rendered = "\n".join(request.summary for request in requests)

    assert "Authorization" in rendered
    assert "API_TOKEN" in rendered
    assert "do-not-print" not in rendered
    assert "do-not-print-either" not in rendered
    assert "line\\nvalue" in rendered

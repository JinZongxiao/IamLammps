"""Fail-closed validation for LAMMPS scripts produced by a model.

This policy is a guardrail, not an operating-system sandbox.  It prevents
LAMMPS input from invoking the most direct host-code and external-script
escape hatches before the input is written or executed.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath

_DEFAULT_BLOCKED_COMMANDS = frozenset(
    {
        "include",
        "jump",
        "plugin",
        "python",
        "shell",
    }
)

_BLOCK_REASONS = {
    "include": "loading another input file bypasses validation of this script",
    "jump": "transferring control to another input file bypasses validation",
    "plugin": "loading native plugins can execute host code",
    "python": "executing Python can access the host process and filesystem",
    "shell": "executing shell commands can access the host system",
}

_STYLE_COMMANDS = frozenset(
    {
        "angle_style",
        "atom_style",
        "bond_style",
        "compute",
        "dihedral_style",
        "fix",
        "improper_style",
        "pair_style",
    }
)

# Commands whose arguments may name input or output files.  LAMMPS supports
# variable-expanded filenames and, for some styles/builds, external compressor
# processes.  Neither is permitted at this trust boundary because validation
# must identify every file before execution.
_SINGLE_FILE_ARGUMENTS = {
    "log": 1,
    "molecule": 2,
    "read_data": 1,
    "read_dump": 1,
    "read_restart": 1,
    "write_coeff": 1,
    "write_data": 1,
    "write_dump": 3,
    "write_restart": 1,
}

_COMPRESSED_SUFFIXES = (
    ".bz2",
    ".gz",
    ".xz",
    ".zst",
    ".zstd",
)


class ScriptPolicyViolation(ValueError):
    """A LAMMPS input command violated the script execution policy."""

    def __init__(self, line_number: int, command: str, reason: str) -> None:
        self.line_number = line_number
        self.command = command
        self.reason = reason
        super().__init__(
            f"unsafe LAMMPS script at line {line_number}: "
            f"command {command!r} is not allowed ({reason})"
        )


@dataclass(frozen=True)
class ScriptPolicy:
    """Commands and indirect execution mechanisms accepted by the validator."""

    blocked_commands: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_BLOCKED_COMMANDS
    )
    reject_dynamic_commands: bool = True
    reject_python_styles: bool = True
    reject_external_paths: bool = True

    def validate(self, script: str) -> None:
        """Validate *script*, raising :class:`ScriptPolicyViolation` if unsafe."""

        if not isinstance(script, str):
            raise TypeError("LAMMPS script must be a string")

        for line_number, command_text in _logical_commands(script):
            self._validate_command(command_text, line_number)

    def _validate_command(self, command_text: str, line_number: int) -> None:
        try:
            tokens = shlex.split(command_text, comments=False, posix=True)
        except ValueError as exc:
            raise ScriptPolicyViolation(
                line_number,
                "<parse>",
                f"the command cannot be parsed safely: {exc}",
            ) from exc

        if not tokens:
            return

        raw_command = tokens[0]
        command = raw_command.casefold()
        if self.reject_dynamic_commands and "$" in raw_command:
            raise ScriptPolicyViolation(
                line_number,
                raw_command,
                "a variable-expanded command name cannot be validated statically",
            )

        if command in self.blocked_commands:
            raise ScriptPolicyViolation(
                line_number,
                command,
                _BLOCK_REASONS.get(command, "the command is blocked by policy"),
            )

        if self.reject_external_paths:
            for token in tokens[1:]:
                normalized = token.replace("\\", "/")
                path = PurePosixPath(normalized)
                if normalized.startswith(("/", "~/")) or ".." in path.parts:
                    raise ScriptPolicyViolation(
                        line_number,
                        command,
                        "absolute paths and parent-directory traversal are not allowed",
                    )

        for token in _file_arguments(command, tokens):
            normalized = token.casefold().replace("\\", "/")
            if "|" in token:
                raise ScriptPolicyViolation(
                    line_number,
                    command,
                    "pipe-backed input and output paths are not allowed",
                )
            if normalized.endswith(_COMPRESSED_SUFFIXES):
                raise ScriptPolicyViolation(
                    line_number,
                    command,
                    "externally compressed input and output paths are not allowed",
                )

            if "$" in token:
                raise ScriptPolicyViolation(
                    line_number,
                    command,
                    "variable-expanded input and output paths are not allowed",
                )

        if self.reject_python_styles and command == "variable":
            if len(tokens) >= 3 and tokens[2].casefold() == "getenv":
                raise ScriptPolicyViolation(
                    line_number,
                    "variable ... getenv",
                    "reading process environment variables may expose credentials",
                )
            if len(tokens) >= 3 and tokens[2].casefold() == "python":
                raise ScriptPolicyViolation(
                    line_number,
                    "variable ... python",
                    "Python-backed variables execute host Python callbacks",
                )

        if self.reject_python_styles and command in _STYLE_COMMANDS:
            python_style = next(
                (
                    token
                    for token in tokens[1:]
                    if token.casefold() == "python"
                    or token.casefold().startswith("python/")
                ),
                None,
            )
            if python_style is not None:
                raise ScriptPolicyViolation(
                    line_number,
                    f"{command} {python_style}",
                    "Python-backed styles execute host Python callbacks",
                )

        for nested_command in _nested_commands(command, tokens, line_number):
            self._validate_command(nested_command, line_number)


DEFAULT_SCRIPT_POLICY = ScriptPolicy()


def validate_lammps_script(
    script: str, policy: ScriptPolicy = DEFAULT_SCRIPT_POLICY
) -> None:
    """Validate model-produced LAMMPS input with the default strict policy."""

    policy.validate(script)


def _logical_commands(script: str) -> Iterator[tuple[int, str]]:
    """Yield comment-free commands, joining LAMMPS ``&`` continuations."""

    parts: list[str] = []
    start_line = 1

    for line_number, physical_line in enumerate(script.splitlines(), start=1):
        line = _strip_comment(physical_line).rstrip()
        if not parts and not line.strip():
            continue
        if not parts:
            start_line = line_number

        continued = line.endswith("&")
        if continued:
            line = line[:-1]
        parts.append(line.strip())

        if not continued:
            command = " ".join(part for part in parts if part)
            if command:
                yield start_line, command
            parts.clear()

    if parts:
        raise ScriptPolicyViolation(
            start_line,
            "<parse>",
            "the script ends with an unfinished '&' continuation",
        )


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None:
            return line[:index]
    return line


def _file_arguments(command: str, tokens: list[str]) -> Iterator[str]:
    index = _SINGLE_FILE_ARGUMENTS.get(command)
    if index is not None and len(tokens) > index:
        yield tokens[index]
    if command == "dump" and len(tokens) > 5:
        yield tokens[5]
    if command == "pair_coeff" and len(tokens) > 3:
        # Whether this token is a filename depends on pair_style.  Applying
        # dynamic-path restrictions is harmless for numeric coefficients and
        # prevents a model from deferring potential selection to LAMMPS.
        yield tokens[3]
    if command == "restart":
        yield from tokens[2:]
    # Many fix styles use a ``file NAME`` option.  Validate only the value,
    # leaving ordinary variable-expanded scientific parameters usable.
    for index, token in enumerate(tokens[:-1]):
        if token.casefold() == "file":
            yield tokens[index + 1]


def _nested_commands(
    command: str, tokens: list[str], line_number: int
) -> Iterator[str]:
    """Extract commands evaluated later by LAMMPS control-flow commands."""

    if command == "run":
        lowered = [token.casefold() for token in tokens]
        try:
            every_index = lowered.index("every")
        except ValueError:
            return
        # ``run N every M cmd...``; NULL explicitly means no nested command.
        for token in tokens[every_index + 2 :]:
            if token.casefold() != "null":
                yield token
        return

    if command != "if":
        return

    # Syntax: if CONDITION then CMD... [elif CONDITION CMD...] [else CMD...]
    # Conditions and commands are quoted into individual shlex tokens.  An
    # ``elif`` does not have a second ``then`` keyword, which is why a simple
    # then/else flag misses its deferred commands.
    try:
        then_index = next(
            index for index, token in enumerate(tokens[1:], start=1)
            if token.casefold() == "then"
        )
    except StopIteration as exc:
        raise ScriptPolicyViolation(
            line_number,
            "if",
            "an if command must contain a then branch",
        ) from exc

    index = then_index + 1
    branch_commands = 0
    while index < len(tokens):
        keyword = tokens[index].casefold()
        if keyword == "elif":
            if branch_commands == 0 or index + 1 >= len(tokens):
                raise ScriptPolicyViolation(line_number, "if", "malformed elif branch")
            # The condition is one quoted LAMMPS argument.
            index += 2
            branch_commands = 0
            continue
        if keyword == "else":
            if branch_commands == 0:
                raise ScriptPolicyViolation(line_number, "if", "malformed else branch")
            index += 1
            branch_commands = 0
            continue
        if keyword == "then":
            raise ScriptPolicyViolation(line_number, "if", "unexpected then keyword")
        yield tokens[index]
        branch_commands += 1
        index += 1

    if branch_commands == 0:
        raise ScriptPolicyViolation(
            line_number, "if", "an if branch must contain a command"
        )

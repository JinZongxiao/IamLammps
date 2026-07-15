"""Declarative project skill discovery.

Project skills are instruction files, not Python plugins.  Reading a skill never
imports or executes code from the skill directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TOOL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_SKILL_BYTES = 256 * 1024
_FRONTMATTER_KEYS = {"name", "description", "recommended-tools"}


class SkillRegistryError(ValueError):
    """Raised when a project skill is malformed or unsafe to read."""


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """Validated instructions and metadata for one project skill."""

    name: str
    description: str
    recommended_tools: tuple[str, ...]
    instructions: str
    path: Path


class SkillRegistry:
    """Discover instruction-only skills below ``.iamlammps/skills``."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root).resolve()
        self.skills_root = self.project_root / ".iamlammps" / "skills"

    def discover(self) -> tuple[SkillDefinition, ...]:
        """Return all skills in stable name order.

        A malformed skill is an explicit configuration error.  Silently ignoring
        it would make an agent's available capabilities differ from its config.
        """

        if not self.skills_root.exists():
            return ()
        if self.skills_root.is_symlink() or not self.skills_root.is_dir():
            raise SkillRegistryError(
                f"Skills root must be a real directory: {self.skills_root}"
            )

        skills: list[SkillDefinition] = []
        seen: set[str] = set()
        for child in sorted(self.skills_root.iterdir(), key=lambda path: path.name):
            if child.is_symlink() or not child.is_dir():
                continue
            skill_path = child / "SKILL.md"
            if not skill_path.exists():
                continue
            skill = self._read_skill(skill_path, expected_name=child.name)
            if skill.name in seen:
                raise SkillRegistryError(f"Duplicate skill name: {skill.name}")
            seen.add(skill.name)
            skills.append(skill)
        return tuple(skills)

    def get(self, name: str) -> SkillDefinition:
        """Return a named skill or raise ``KeyError`` when it is not registered."""

        for skill in self.discover():
            if skill.name == name:
                return skill
        raise KeyError(name)

    def _read_skill(self, path: Path, *, expected_name: str) -> SkillDefinition:
        if path.is_symlink() or not path.is_file():
            raise SkillRegistryError(f"SKILL.md must be a regular file: {path}")
        resolved_path = path.resolve()
        resolved_root = self.skills_root.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise SkillRegistryError(f"Skill path escapes the registry: {path}")
        size = path.stat().st_size
        if size > _MAX_SKILL_BYTES:
            raise SkillRegistryError(
                f"Skill file is too large ({size} bytes, max {_MAX_SKILL_BYTES}): {path}"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillRegistryError(f"Skill must be UTF-8: {path}") from exc

        metadata, instructions = _parse_skill_document(text, path)
        name = metadata["name"]
        if not _NAME_PATTERN.fullmatch(name):
            raise SkillRegistryError(f"Invalid skill name {name!r} in {path}")
        if expected_name != name:
            raise SkillRegistryError(
                f"Skill name {name!r} must match directory {expected_name!r}"
            )
        description = metadata["description"].strip()
        if not description:
            raise SkillRegistryError(f"Skill description must not be empty: {path}")
        tools = _parse_recommended_tools(metadata["recommended-tools"], path)
        if not instructions.strip():
            raise SkillRegistryError(f"Skill instructions must not be empty: {path}")
        return SkillDefinition(
            name=name,
            description=description,
            recommended_tools=tools,
            instructions=instructions.strip(),
            path=resolved_path,
        )


def discover_skills(project_root: str | Path = ".") -> tuple[SkillDefinition, ...]:
    """Convenience wrapper for project skill discovery."""

    return SkillRegistry(project_root).discover()


def _parse_skill_document(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillRegistryError(f"Skill must start with YAML-style frontmatter: {path}")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillRegistryError(f"Unterminated skill frontmatter: {path}") from exc

    metadata: dict[str, str] = {}
    current_list_key: str | None = None
    for line_number, raw_line in enumerate(lines[1:end], start=2):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            if current_list_key != "recommended-tools":
                raise SkillRegistryError(
                    f"Unexpected list item at {path}:{line_number}"
                )
            item = stripped[1:].strip()
            if not item:
                raise SkillRegistryError(f"Empty tool name at {path}:{line_number}")
            previous = metadata.get(current_list_key, "")
            metadata[current_list_key] = f"{previous},{item}".strip(",")
            continue
        if ":" not in raw_line:
            raise SkillRegistryError(f"Invalid frontmatter at {path}:{line_number}")
        key, value = (part.strip() for part in raw_line.split(":", 1))
        if key not in _FRONTMATTER_KEYS:
            raise SkillRegistryError(f"Unknown skill field {key!r} at {path}:{line_number}")
        if key in metadata:
            raise SkillRegistryError(f"Duplicate skill field {key!r} at {path}:{line_number}")
        metadata[key] = _unquote(value)
        current_list_key = key if not value else None

    missing = _FRONTMATTER_KEYS - metadata.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise SkillRegistryError(f"Missing skill fields ({fields}): {path}")
    return metadata, "\n".join(lines[end + 1 :])


def _parse_recommended_tools(value: str, path: Path) -> tuple[str, ...]:
    raw = value.strip()
    if raw.startswith("["):
        if not raw.endswith("]"):
            raise SkillRegistryError(f"Invalid recommended-tools list: {path}")
        raw = raw[1:-1]
    tools = tuple(_unquote(item.strip()) for item in raw.split(",") if item.strip())
    if len(set(tools)) != len(tools):
        raise SkillRegistryError(f"Duplicate tool in recommended-tools: {path}")
    for tool in tools:
        if not _TOOL_PATTERN.fullmatch(tool):
            raise SkillRegistryError(f"Invalid recommended tool {tool!r}: {path}")
    return tools


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value

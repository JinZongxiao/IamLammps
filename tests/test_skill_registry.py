from __future__ import annotations

from pathlib import Path

import pytest

from lammps_agent.skill_registry import SkillRegistry, SkillRegistryError


def _write_skill(root: Path, name: str, content: str) -> Path:
    path = root / ".iamlammps" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_registry_is_empty(tmp_path: Path) -> None:
    assert SkillRegistry(tmp_path).discover() == ()


def test_discovers_declarative_skills_in_stable_order(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "web-research",
        """---
name: web-research
description: Find scientific sources
recommended-tools:
  - web.search
  - web.fetch
---
Find primary sources and cite them.
""",
    )
    _write_skill(
        tmp_path,
        "analyze",
        """---
name: analyze
description: Analyze a completed run
recommended-tools: [read_run, plot.data]
---
Inspect thermo data before drawing conclusions.
""",
    )

    skills = SkillRegistry(tmp_path).discover()

    assert [skill.name for skill in skills] == ["analyze", "web-research"]
    assert skills[0].recommended_tools == ("read_run", "plot.data")
    assert skills[1].instructions == "Find primary sources and cite them."
    assert skills[1].path.is_absolute()


def test_get_returns_named_skill(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "analyze",
        """---
name: analyze
description: Analyze data
recommended-tools: []
---
Read the result.
""",
    )
    registry = SkillRegistry(tmp_path)

    assert registry.get("analyze").description == "Analyze data"
    with pytest.raises(KeyError):
        registry.get("missing")


@pytest.mark.parametrize(
    "document, message",
    [
        ("No frontmatter", "frontmatter"),
        ("---\nname: x\n---\nDo x", "Missing skill fields"),
        (
            "---\nname: wrong\ndescription: X\nrecommended-tools: []\n---\nDo x",
            "must match directory",
        ),
        (
            "---\nname: x\ndescription: X\nrecommended-tools: [bad tool]\n---\nDo x",
            "Invalid recommended tool",
        ),
        (
            "---\nname: x\ndescription: X\nrecommended-tools: []\npython: run.py\n---\nDo x",
            "Unknown skill field",
        ),
    ],
)
def test_rejects_malformed_or_executable_metadata(
    tmp_path: Path, document: str, message: str
) -> None:
    _write_skill(tmp_path, "x", document)

    with pytest.raises(SkillRegistryError, match=message):
        SkillRegistry(tmp_path).discover()


def test_does_not_follow_skill_file_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nname: x\ndescription: X\nrecommended-tools: []\n---\nDo x", encoding="utf-8"
    )
    skill_dir = tmp_path / ".iamlammps" / "skills" / "x"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").symlink_to(outside)

    with pytest.raises(SkillRegistryError, match="regular file"):
        SkillRegistry(tmp_path).discover()


def test_ignores_non_skill_files_without_importing_them(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".iamlammps" / "skills" / "safe"
    _write_skill(
        tmp_path,
        "safe",
        "---\nname: safe\ndescription: Safe\nrecommended-tools: []\n---\nFollow instructions.",
    )
    (skill_dir / "plugin.py").write_text("raise RuntimeError('executed')", encoding="utf-8")

    assert SkillRegistry(tmp_path).get("safe").name == "safe"

"""Tests for the enhanced memory system."""

from __future__ import annotations

import json
from pathlib import Path

from src.lammps_agent.memory import SimulationMemoryEntry, SimulationMemoryStore


def test_new_fields_defaults():
    """New fields should have sensible defaults."""
    entry = SimulationMemoryEntry(
        timestamp="2026-04-17T10:00:00",
        task="test",
        run_dir="/tmp/test",
        return_code=0,
        summary="ok",
    )
    assert entry.reward_score is None
    assert entry.error_type is None
    assert entry.retry_count == 0
    assert entry.script_hash is None


def test_new_fields_serialization(tmp_path: Path):
    """Entries with new fields should round-trip through JSONL."""
    mem_file = tmp_path / "memory.jsonl"
    store = SimulationMemoryStore(memory_file=mem_file)

    entry = store.make_entry(
        task="test task",
        run_dir=tmp_path,
        return_code=0,
        summary="ok",
        reward_score=0.85,
        error_type=None,
        retry_count=1,
        script_hash="abc123def456",
    )
    store.append(entry)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].reward_score == 0.85
    assert loaded[0].retry_count == 1
    assert loaded[0].script_hash == "abc123def456"


def test_backward_compat_old_jsonl(tmp_path: Path):
    """Old JSONL entries without new fields should load without error."""
    mem_file = tmp_path / "memory.jsonl"
    old_entry = {
        "timestamp": "2026-04-08T16:11:14",
        "task": "old task",
        "run_dir": "/tmp/old",
        "return_code": 1,
        "summary": "return_code=1",
    }
    mem_file.write_text(json.dumps(old_entry) + "\n", encoding="utf-8")

    store = SimulationMemoryStore(memory_file=mem_file)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].reward_score is None
    assert loaded[0].retry_count == 0


def test_compressed_context_reward_ordering(tmp_path: Path):
    """High-reward entries should appear before low-reward entries in context."""
    mem_file = tmp_path / "memory.jsonl"
    store = SimulationMemoryStore(memory_file=mem_file)

    # Append low-reward entry first, then high-reward.
    store.append(store.make_entry(
        task="bad task", run_dir=tmp_path, return_code=1,
        summary="failed", reward_score=0.1,
    ))
    store.append(store.make_entry(
        task="good task", run_dir=tmp_path, return_code=0,
        summary="success", reward_score=0.9,
    ))

    context = store.compressed_context()
    good_pos = context.find("good task")
    bad_pos = context.find("bad task")
    # High-reward ("good task") should appear before low-reward ("bad task").
    assert good_pos < bad_pos

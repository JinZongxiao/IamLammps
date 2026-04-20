from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class SimulationMemoryEntry:
    timestamp: str
    task: str
    run_dir: str
    return_code: int | None
    summary: str
    # --- New fields (optional for backward compat with older JSONL) ---
    reward_score: float | None = None
    error_type: str | None = None
    retry_count: int = 0
    script_hash: str | None = None


class SimulationMemoryStore:
    def __init__(self, memory_file: Path, max_entries: int = 200) -> None:
        self.memory_file = memory_file
        self.max_entries = max_entries
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: SimulationMemoryEntry) -> None:
        entries = self.load_all()
        entries.append(entry)
        entries = entries[-self.max_entries :]

        with self.memory_file.open("w", encoding="utf-8") as f:
            for item in entries:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    def load_all(self) -> list[SimulationMemoryEntry]:
        if not self.memory_file.exists():
            return []

        # Known field names for backward compatibility filtering.
        _KNOWN_FIELDS = {f.name for f in SimulationMemoryEntry.__dataclass_fields__.values()}

        result: list[SimulationMemoryEntry] = []
        with self.memory_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Strip unknown keys from old/future entries.
                    filtered = {k: v for k, v in data.items() if k in _KNOWN_FIELDS}
                    result.append(SimulationMemoryEntry(**filtered))
                except Exception:
                    continue
        return result

    def load_recent(self, limit: int = 20) -> list[SimulationMemoryEntry]:
        return self.load_all()[-limit:]

    def compressed_context(self, max_chars: int = 2400, recent_limit: int = 20) -> str:
        entries = self.load_recent(limit=recent_limit)
        if not entries:
            return ""

        # Sort: high-reward successes first, then by timestamp (recent last).
        def _sort_key(e: SimulationMemoryEntry) -> tuple[float, str]:
            score = e.reward_score if e.reward_score is not None else -1.0
            return (-score, e.timestamp)

        entries_sorted = sorted(entries, key=_sort_key)

        lines = ["Recent simulation memory (compressed):"]
        for e in entries_sorted:
            code = "n/a" if e.return_code is None else str(e.return_code)
            reward_str = f" reward={e.reward_score:.2f}" if e.reward_score is not None else ""
            retry_str = f" retries={e.retry_count}" if e.retry_count > 0 else ""
            lines.append(
                f"- [{e.timestamp}] rc={code}{reward_str}{retry_str} "
                f"task={self._shorten(e.task, 140)} summary={self._shorten(e.summary, 180)}"
            )

        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text

        # Keep the newest memories under context budget.
        kept = [lines[0]]
        for line in reversed(lines[1:]):
            candidate = "\n".join(kept + [line])
            if len(candidate) > max_chars:
                break
            kept.append(line)
        kept = [kept[0]] + list(reversed(kept[1:]))
        return "\n".join(kept)

    @staticmethod
    def make_entry(
        task: str,
        run_dir: Path,
        return_code: int | None,
        summary: str,
        reward_score: float | None = None,
        error_type: str | None = None,
        retry_count: int = 0,
        script_hash: str | None = None,
    ) -> SimulationMemoryEntry:
        return SimulationMemoryEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            task=task,
            run_dir=str(run_dir),
            return_code=return_code,
            summary=summary,
            reward_score=reward_score,
            error_type=error_type,
            retry_count=retry_count,
            script_hash=script_hash,
        )

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."

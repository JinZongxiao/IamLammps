from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _parse_optional_int(value: str | None) -> int | None:
    if not value or not value.strip():
        return None
    return int(value.strip())


@dataclass
class AgentConfig:
    llm_provider: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    local_base_url: str
    local_model: str
    lammps_command: str
    enabled_skills: list[str]
    memory_file: str
    memory_max_entries: int
    memory_context_chars: int
    max_retries: int
    # --- Phase B additions ---
    lammps_timeout: int | None = None
    mpi_np: int | None = None
    potentials_dir: str | None = None
    # --- Backend abstraction ---
    backend: str = "local"  # "local" | "slurm"


def load_config() -> AgentConfig:
    # Prefer .env; fall back to .env.example for convenience.
    load_dotenv(dotenv_path=Path(".env"), override=False)
    if not Path(".env").exists() and Path(".env.example").exists():
        load_dotenv(dotenv_path=Path(".env.example"), override=False)

    raw_skills = os.getenv("ENABLED_SKILLS", "thermo_tail,convergence,thermo_validator").strip()
    enabled_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]

    return AgentConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        local_base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434").strip(),
        local_model=os.getenv("LOCAL_MODEL", "qwen2.5:7b").strip(),
        lammps_command=os.getenv("LAMMPS_COMMAND", "lmp_mpi").strip(),
        enabled_skills=enabled_skills,
        memory_file=os.getenv("MEMORY_FILE", "runs/simulation_memory.jsonl").strip(),
        memory_max_entries=int(os.getenv("MEMORY_MAX_ENTRIES", "200").strip()),
        memory_context_chars=int(os.getenv("MEMORY_CONTEXT_CHARS", "2400").strip()),
        max_retries=int(os.getenv("MAX_RETRIES", "3").strip()),
        lammps_timeout=_parse_optional_int(os.getenv("LAMMPS_TIMEOUT")),
        mpi_np=_parse_optional_int(os.getenv("MPI_NP")),
        potentials_dir=os.getenv("LAMMPS_POTENTIALS_DIR", "").strip() or None,
        backend=os.getenv("BACKEND", "local").strip(),
    )

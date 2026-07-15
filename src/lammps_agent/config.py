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
    lammps_command: str
    enabled_analyzers: list[str]
    lammps_timeout: int | None = None
    mpi_np: int | None = None
    potentials_dir: str | None = None
    reasoning_effort: str = "medium"
    openai_tracing: bool = True
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"

def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_config() -> AgentConfig:
    # Prefer .env; fall back to .env.example for convenience.
    load_dotenv(dotenv_path=Path(".env"), override=False)
    if not Path(".env").exists() and Path(".env.example").exists():
        load_dotenv(dotenv_path=Path(".env.example"), override=False)

    raw_analyzers = os.getenv(
        "ENABLED_ANALYZERS", "thermo_tail,convergence,thermo_validator"
    ).strip()
    enabled_analyzers = [
        name.strip() for name in raw_analyzers.split(",") if name.strip()
    ]

    return AgentConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai_agents").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "").strip(),
        lammps_command=os.getenv("LAMMPS_COMMAND", "lmp_mpi").strip(),
        enabled_analyzers=enabled_analyzers,
        lammps_timeout=_parse_optional_int(os.getenv("LAMMPS_TIMEOUT")),
        mpi_np=_parse_optional_int(os.getenv("MPI_NP")),
        potentials_dir=os.getenv("LAMMPS_POTENTIALS_DIR", "").strip() or None,
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium").strip(),
        openai_tracing=_parse_bool(os.getenv("OPENAI_TRACING"), True),
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).strip(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
    )

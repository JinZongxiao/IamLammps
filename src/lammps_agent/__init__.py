"""IamLAMMPS simulation coding agent."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation_agent import SimulationCodingAgent

__all__ = ["SimulationCodingAgent"]


def __getattr__(name: str):
    if name == "SimulationCodingAgent":
        from .simulation_agent import SimulationCodingAgent

        return SimulationCodingAgent
    raise AttributeError(name)

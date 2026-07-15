"""Interatomic potential registry and manager."""

from .manager import PotentialManager
from .registry import PotentialEntry, PotentialRegistry

__all__ = ["PotentialEntry", "PotentialRegistry", "PotentialManager"]

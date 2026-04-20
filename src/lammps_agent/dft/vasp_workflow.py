"""DFT workflow placeholder for VASP (or similar DFT code).

Intended use: generate DFT training data for MLIP fitting.

All methods raise :class:`NotImplementedError` — this module defines
the **interface** that a future VASP integration should implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DFTResult:
    """Result of a single DFT calculation."""

    energy: float = 0.0                          # eV
    forces: list[list[float]] = field(default_factory=list)  # Nx3
    stress: list[float] = field(default_factory=list)        # 6 (Voigt)
    converged: bool = False
    output_dir: Path | None = None


class VASPWorkflow:
    """Placeholder for VASP DFT calculations.

    Parameters
    ----------
    vasp_command : str
        VASP executable (e.g. ``"vasp_std"``).
    potcar_dir : str | Path | None
        Directory containing POTCAR files.
    """

    def __init__(
        self,
        vasp_command: str = "vasp_std",
        potcar_dir: str | Path | None = None,
    ) -> None:
        self.vasp_command = vasp_command
        self.potcar_dir = Path(potcar_dir) if potcar_dir else None

    def generate_input(
        self,
        structure: Any,
        calculation_type: str = "static",
        encut: float = 500.0,
        kpoints: tuple[int, int, int] = (4, 4, 4),
    ) -> dict[str, str]:
        """Generate VASP input files (POSCAR, INCAR, KPOINTS, POTCAR).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "VASPWorkflow.generate_input() is not yet implemented."
        )

    def submit(
        self,
        input_dir: Path,
        backend: str = "local",
    ) -> str:
        """Submit a VASP job and return a job ID.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "VASPWorkflow.submit() is not yet implemented."
        )

    def parse_output(self, output_dir: Path) -> DFTResult:
        """Parse VASP output (OUTCAR, vasprun.xml) into a DFTResult.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "VASPWorkflow.parse_output() is not yet implemented."
        )

    def build_training_set(
        self,
        structures: list[Any],
        calculation_type: str = "static",
    ) -> list[DFTResult]:
        """Run DFT on multiple structures to build MLIP training data.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "VASPWorkflow.build_training_set() is not yet implemented."
        )

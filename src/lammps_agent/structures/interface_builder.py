"""Build oxide/W interface models for LAMMPS.

Generates LAMMPS command strings or data-file content for bilayer slab
models used in interface adhesion and mechanical testing.

.. note::

   This module provides **placeholder** implementations.  Real interface
   construction will require:

   1. Actual oxide crystal structures (from Materials Project or CIF files).
   2. Lattice-matching algorithms (CSL or strain-based).
   3. Interface termination selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OxidePhase:
    """Minimal descriptor for an oxide crystal phase."""

    name: str                          # e.g. "Y2O3", "Al2O3"
    structure: str = "cubic"           # "cubic" | "hexagonal" | ...
    lattice_a: float = 0.0             # Angstrom
    lattice_c: float | None = None     # Angstrom (for non-cubic)
    atom_types: list[str] = field(default_factory=list)


# Placeholder oxide data — real values would come from a crystal database.
_OXIDE_REGISTRY: dict[str, OxidePhase] = {
    "Y2O3": OxidePhase(
        name="Y2O3",
        structure="cubic",
        lattice_a=10.604,
        atom_types=["Y", "O"],
    ),
    "Al2O3": OxidePhase(
        name="Al2O3",
        structure="hexagonal",
        lattice_a=4.759,
        lattice_c=12.991,
        atom_types=["Al", "O"],
    ),
}


class InterfaceBuilder:
    """Construct oxide/W interface structures.

    All ``build_*`` methods are currently **placeholders** that raise
    :class:`NotImplementedError`.  They document the intended API and
    parameter signatures so downstream code can be written against them.
    """

    def __init__(self, oxide_name: str = "Y2O3") -> None:
        if oxide_name not in _OXIDE_REGISTRY:
            raise ValueError(
                f"Unknown oxide: {oxide_name!r}. "
                f"Available: {list(_OXIDE_REGISTRY)}"
            )
        self.oxide = _OXIDE_REGISTRY[oxide_name]

    @staticmethod
    def list_oxides() -> list[str]:
        """Return names of registered oxide phases."""
        return list(_OXIDE_REGISTRY)

    def build_bilayer_slab(
        self,
        w_layers: int = 10,
        oxide_layers: int = 5,
        orientation: str = "(110)",
        vacuum: float = 20.0,
    ) -> str:
        """Build a W/oxide bilayer slab model.

        Parameters
        ----------
        w_layers : int
            Number of W atomic layers.
        oxide_layers : int
            Number of oxide atomic layers.
        orientation : str
            W surface orientation.
        vacuum : float
            Vacuum gap in Angstrom (for surface boundary).

        Returns
        -------
        str
            LAMMPS commands or data-file content.

        Raises
        ------
        NotImplementedError
            Always — this is a placeholder.
        """
        raise NotImplementedError(
            f"InterfaceBuilder.build_bilayer_slab() for W/{self.oxide.name} "
            f"is not yet implemented.  Requires crystal structure data and "
            f"lattice-matching algorithm."
        )

    def build_coherent_interface(
        self,
        w_supercell: tuple[int, int, int] = (5, 5, 10),
        rotation_matrix: list[list[float]] | None = None,
    ) -> str:
        """Build a coherent W/oxide interface with minimal lattice mismatch.

        Raises
        ------
        NotImplementedError
            Always — this is a placeholder.
        """
        raise NotImplementedError(
            "InterfaceBuilder.build_coherent_interface() is not yet implemented."
        )

    def compute_interface_area(
        self,
        lx: float,
        ly: float,
    ) -> float:
        """Return the interface area in Angstrom² given box dimensions."""
        return lx * ly

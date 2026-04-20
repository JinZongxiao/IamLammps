"""Tungsten model construction utilities.

Generates LAMMPS command strings for various BCC tungsten model geometries.
"""

from __future__ import annotations


class TungstenModelBuilder:
    """Generate LAMMPS command strings for BCC tungsten models."""

    LATTICE_CONSTANT = 3.165  # Angstrom
    ATOMIC_MASS = 183.84

    def bcc_supercell_commands(
        self,
        nx: int,
        ny: int,
        nz: int,
        lattice_constant: float = 3.165,
    ) -> str:
        """Return LAMMPS commands to create a BCC W supercell."""
        return (
            f"lattice bcc {lattice_constant}\n"
            f"region box block 0 {nx} 0 {ny} 0 {nz}\n"
            f"create_box 1 box\n"
            f"create_atoms 1 box\n"
            f"mass 1 {self.ATOMIC_MASS}"
        )

    def slab_for_indentation(
        self,
        nx: int,
        ny: int,
        nz: int,
        lattice_constant: float = 3.165,
        fixed_layer_fraction: float = 0.15,
        thermostat_layer_fraction: float = 0.15,
    ) -> str:
        """Return LAMMPS commands for an indentation slab model.

        Creates 3 groups:
        - fixed (bottom): atoms held rigid
        - thermostat (middle-bottom): Langevin thermostat layer
        - mobile (rest): Newtonian dynamics
        """
        lz = nz * lattice_constant
        fixed_z = fixed_layer_fraction * nz * lattice_constant
        thermo_z = (fixed_layer_fraction + thermostat_layer_fraction) * nz * lattice_constant

        lines = [
            f"lattice bcc {lattice_constant}",
            f"region box block 0 {nx} 0 {ny} 0 {nz}",
            "create_box 1 box",
            "create_atoms 1 box",
            f"mass 1 {self.ATOMIC_MASS}",
            "",
            "# Define layers for indentation boundary conditions",
            f"region fixed_region block INF INF INF INF INF {fixed_z:.4f}",
            f"region thermo_region block INF INF INF INF {fixed_z:.4f} {thermo_z:.4f}",
            "group fixed_atoms region fixed_region",
            "group thermo_atoms region thermo_region",
            "group mobile_atoms subtract all fixed_atoms",
        ]
        return "\n".join(lines)

    def precracked_specimen(
        self,
        nx: int,
        ny: int,
        nz: int,
        lattice_constant: float = 3.165,
        crack_length_fraction: float = 0.3,
        crack_width_fraction: float = 0.02,
    ) -> str:
        """Return LAMMPS commands for a specimen with a pre-crack.

        The crack is a rectangular notch on the left side of the xz-plane,
        extending into the middle of the y-direction.
        """
        lx = nx * lattice_constant
        ly = ny * lattice_constant
        crack_len = crack_length_fraction * lx
        half_width = crack_width_fraction * ly / 2.0
        y_mid = ly / 2.0

        lines = [
            f"lattice bcc {lattice_constant}",
            f"region box block 0 {nx} 0 {ny} 0 {nz}",
            "create_box 1 box",
            "create_atoms 1 box",
            f"mass 1 {self.ATOMIC_MASS}",
            "",
            "# Create pre-crack by deleting atoms in a thin notch region",
            f"region crack block 0.0 {crack_len:.4f} "
            f"{y_mid - half_width:.4f} {y_mid + half_width:.4f} INF INF",
            "delete_atoms region crack",
        ]
        return "\n".join(lines)

    def full_setup_header(self) -> str:
        """Return standard header commands for W simulations."""
        return (
            "units metal\n"
            "atom_style atomic\n"
            "boundary p p p"
        )

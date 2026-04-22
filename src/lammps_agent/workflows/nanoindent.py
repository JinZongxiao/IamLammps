"""Nanoindentation workflow for BCC tungsten — deterministic template edition."""

from __future__ import annotations

import random
import sys

from ..pipeline import PipelineResult, PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig

_A0 = 3.165  # BCC-W lattice constant (Angstrom)


class NanoindentWorkflow(BaseWorkflow):
    """3-stage pipeline: minimize → equilibrate → indent.

    All stages use deterministic script_template — no LLM free-generation.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        supercell: tuple[int, int, int] = (20, 20, 30),
        indenter_radius: float = 30.0,
        indenter_velocity: float = 1e-3,
        dump_every: int | None = None,
        dump_split: bool = False,
    ) -> None:
        super().__init__(config, agent)
        self.supercell = supercell
        self.indenter_radius = indenter_radius
        self.indenter_velocity = indenter_velocity
        self.dump_every = dump_every if dump_every is not None else config.dump_every
        self.dump_split = dump_split

    # Report config — set by main.py after construction
    _report_config: "NanoindentReportConfig | None" = None  # type: ignore[name-defined]

    def run(self) -> PipelineResult:
        pipeline = self.build_pipeline()
        result = pipeline.run()
        if result.success:
            self._try_generate_report(result)
        return result

    def _try_generate_report(self, result: PipelineResult) -> None:
        try:
            from ..report import NanoindentReporter, NanoindentReportConfig
            cfg = self._report_config or NanoindentReportConfig(
                supercell=self.supercell,
                temperature=self.config.temperature,
                potential_name=self.config.potential_name,
                indenter_radius=self.indenter_radius,
                indenter_velocity=self.indenter_velocity,
                lammps_command=self.agent.config.lammps_command,
            )
            NanoindentReporter(cfg).generate(result)
        except Exception as exc:
            print(f"[REPORT WARNING] Nanoindent report failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def build_pipeline(self) -> SimulationPipeline:
        pot_path = self._resolve_pot_path()
        stages = [
            PipelineStage(
                name="minimize",
                task_description="minimize (template)",
                script_template=self._minimize_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
            PipelineStage(
                name="equilibrate",
                task_description="equilibrate (template)",
                depends_on="minimize",
                input_files=["restart.minimize"],
                script_template=self._equilibrate_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
            PipelineStage(
                name="indent",
                task_description="indent (template)",
                depends_on="equilibrate",
                input_files=["restart.equilibrate"],
                script_template=self._indent_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_pot_path(self) -> str:
        pm = PotentialManager()
        pot = pm.resolve_potential(self.config.potential_name or "W_Zhou2004_EAM")
        return str(pot.file_path)

    def _geometry(self) -> dict:
        nx, ny, nz = self.supercell
        lx = nx * _A0
        ly = ny * _A0
        lz = nz * _A0
        z_fixed = 0.15 * lz   # bottom 15 %: rigid layer
        z_thermo = 0.30 * lz  # 15 %–30 %: thermostat layer
        return dict(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz,
                    z_fixed=z_fixed, z_thermo=z_thermo)

    # ------------------------------------------------------------------
    # Stage scripts
    # ------------------------------------------------------------------

    def _minimize_script(self, pot_path: str) -> str:
        g = self._geometry()
        return f"""\
# === minimize stage: BCC-W slab {g['nx']}x{g['ny']}x{g['nz']} for nanoindentation ===
units metal
atom_style atomic
boundary p p p

lattice bcc {_A0}
region simbox block 0 {g['nx']} 0 {g['ny']} 0 {g['nz']} units lattice
create_box 1 simbox
create_atoms 1 box

mass 1 183.84
pair_style eam/alloy
pair_coeff * * {pot_path} W

# Bottom 15 % → fixed; next 15 % → thermostat; rest → mobile
region fixed_region  block INF INF INF INF INF {g['z_fixed']:.6f} units box
region thermo_region block INF INF INF INF {g['z_fixed']:.6f} {g['z_thermo']:.6f} units box
group fixed_atoms  region fixed_region
group thermo_atoms region thermo_region
group mobile_atoms subtract all fixed_atoms

thermo 100
thermo_style custom step pe ke etotal fmax fnorm press

min_style cg
minimize 1e-10 1e-10 10000 100000

write_restart restart.minimize
"""

    def _equilibrate_script(self, pot_path: str) -> str:
        g = self._geometry()
        T = self.config.temperature
        seed = random.randint(10000, 99999)
        return f"""\
# === equilibrate stage: NVT at {T} K ===
read_restart restart.minimize

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

# Re-create groups (same numeric bounds as minimize)
region fixed_region  block INF INF INF INF INF {g['z_fixed']:.6f} units box
region thermo_region block INF INF INF INF {g['z_fixed']:.6f} {g['z_thermo']:.6f} units box
group fixed_atoms  region fixed_region
group thermo_atoms region thermo_region
group mobile_atoms subtract all fixed_atoms

fix freeze  fixed_atoms  setforce 0 0 0
velocity mobile_atoms create {T} {seed} dist gaussian

fix nvt_eq thermo_atoms nvt temp {T} {T} 0.1
fix nve_eq mobile_atoms nve

timestep 0.001
thermo 100
thermo_style custom step temp pe ke etotal press

run 10000

write_restart restart.equilibrate
"""

    def _indent_script(self, pot_path: str) -> str:
        g = self._geometry()
        T = self.config.temperature
        seed = random.randint(10000, 99999)
        R = self.indenter_radius
        vel = self.indenter_velocity
        K = 10.0                        # eV/Angstrom^2 — indenter stiffness
        indent_depth = 0.1 * g['lz']   # indent to 10 % of slab height
        nsteps = max(1, int(indent_depth / vel))

        # Indenter starts just above the slab surface + one radius clearance
        z_start = g['lz'] + R + 5.0

        # Indenter x/y centred on slab
        x_centre = g['lx'] / 2.0
        y_centre = g['ly'] / 2.0

        if self.dump_every is not None:
            if self.dump_split:
                dump_line = f"dump 1 all custom {self.dump_every} dump.indent.*.lammpstrj id type x y z"
            else:
                dump_line = f"dump 1 all custom {self.dump_every} dump.indent.lammpstrj id type x y z"
        else:
            dump_line = "# dump disabled"

        return f"""\
# === indent stage: spherical indenter R={R} A, vel={vel} A/step ===
# indent_depth = {indent_depth:.4f} A  →  nsteps = {nsteps}
read_restart restart.equilibrate

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

# Re-create groups (same numeric bounds as previous stages)
region fixed_region  block INF INF INF INF INF {g['z_fixed']:.6f} units box
region thermo_region block INF INF INF INF {g['z_fixed']:.6f} {g['z_thermo']:.6f} units box
group fixed_atoms  region fixed_region
group thermo_atoms region thermo_region
group mobile_atoms subtract all fixed_atoms

fix freeze  fixed_atoms  setforce 0 0 0
velocity mobile_atoms create {T} {seed} dist gaussian

# Indenter z position decreases at constant velocity (equal-style variable)
variable indenter_z equal {z_start:.6f}-{vel}*step*dt

# Spherical indenter: fix indent K sphere x0 y0 v_var R
fix indent_fix mobile_atoms indent {K} sphere {x_centre:.6f} {y_centre:.6f} v_indenter_z {R}

fix langevin_th thermo_atoms langevin {T} {T} 0.1 {seed+1}
fix nve_mob     mobile_atoms nve

timestep 0.001
thermo 100
thermo_style custom step temp f_indent_fix[1] f_indent_fix[2] f_indent_fix[3] v_indenter_z pe

{dump_line}

run {nsteps}
"""

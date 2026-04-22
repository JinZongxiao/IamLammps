"""Nanoindentation workflow for BCC tungsten."""

from __future__ import annotations

import random
import sys

from ..pipeline import PipelineResult, PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig


class NanoindentWorkflow(BaseWorkflow):
    """3-stage pipeline: build slab + minimize → equilibrate → indent."""

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        supercell: tuple[int, int, int] = (20, 20, 30),
        indenter_radius: float = 30.0,
        indenter_velocity: float = 1e-3,
        dump_every: int | None = None,
        dump_split: bool = False,   # False = single-file dump (default)
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

    def build_pipeline(self) -> SimulationPipeline:
        stages = [
            PipelineStage(
                name="minimize",
                task_description=self._minimize_task(),
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="equilibrate",
                task_description=self._equilibrate_task(),
                depends_on="minimize",
                input_files=["restart.minimize"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="indent",
                task_description=self._indent_task(),
                depends_on="equilibrate",
                input_files=["restart.equilibrate"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    def _minimize_task(self) -> str:
        nx, ny, nz = self.supercell
        return (
            f"Create a BCC tungsten slab for nanoindentation ({nx}x{ny}x{nz} unit cells). "
            f"Metal units, atom_style atomic, lattice bcc 3.165 Angstrom.\n"
            f"pair_style eam/alloy, pair_coeff * * W_zhou.eam.alloy W, mass 1 183.84.\n"
            f"Create 3 groups for boundary conditions:\n"
            f"- 'fixed' group: bottom 15% of z (rigid layer, no forces)\n"
            f"- 'thermo' group: next 15% of z (thermostat layer)\n"
            f"- 'mobile' group: everything else\n"
            f"Use: region fixed_region block INF INF INF INF INF <z_fixed>\n"
            f"      group fixed_atoms region fixed_region\n"
            f"      group mobile_atoms subtract all fixed_atoms\n"
            f"Minimize with CG: etol=1e-10, ftol=1e-10, maxiter=10000, maxeval=100000.\n"
            f"write_restart restart.minimize"
        )

    def _equilibrate_task(self) -> str:
        T = self.config.temperature
        seed = random.randint(10000, 99999)
        nx, ny, nz = self.supercell
        lz = nz * 3.165
        z_fixed = 0.15 * lz   # bottom 15 % of slab height
        z_thermo = 0.30 * lz  # next 15 % (15 %–30 %)
        return (
            f"read_restart restart.minimize\n"
            f"pair_style eam/alloy\n"
            f"pair_coeff * * W_zhou.eam.alloy W\n"
            f"mass 1 183.84\n"
            f"\n"
            f"# Re-create groups using hard-coded z boundaries\n"
            f"# (compute/reduce cannot be used before run 0 for region definitions)\n"
            f"region fixed_region  block INF INF INF INF INF {z_fixed:.4f}\n"
            f"region thermo_region block INF INF INF INF {z_fixed:.4f} {z_thermo:.4f}\n"
            f"group fixed_atoms  region fixed_region\n"
            f"group thermo_atoms region thermo_region\n"
            f"group mobile_atoms subtract all fixed_atoms\n"
            f"\n"
            f"fix freeze  fixed_atoms  setforce 0 0 0\n"
            f"velocity mobile_atoms create {T} {seed} dist gaussian\n"
            f"\n"
            f"fix nvt_eq thermo_atoms nvt temp {T} {T} 0.1\n"
            f"fix nve_eq mobile_atoms nve\n"
            f"\n"
            f"timestep 0.001\n"
            f"thermo 100\n"
            f"thermo_style custom step temp pe ke etotal press\n"
            f"run 10000\n"
            f"\n"
            f"write_restart restart.equilibrate"
        )

    def _indent_task(self) -> str:
        T = self.config.temperature
        nx, ny, nz = self.supercell
        lz = nz * 3.165
        R = self.indenter_radius
        vel = self.indenter_velocity
        z_top = lz + R + 5.0
        K = 10.0  # eV/Angstrom^2
        indent_depth = 0.1 * lz
        nsteps = int(indent_depth / vel)

        # Dump: single-file by default (dump_split=False)
        if self.dump_every is not None:
            if self.dump_split:
                dump_cmd = f"dump 1 all custom {self.dump_every} dump.indent.*.lammpstrj id type x y z"
            else:
                dump_cmd = f"dump 1 all custom {self.dump_every} dump.indent.lammpstrj id type x y z"
        else:
            dump_cmd = ""

        return (
            f"read_restart restart.equilibrate\n"
            f"pair_style eam/alloy, pair_coeff * * W_zhou.eam.alloy W, mass 1 183.84.\n"
            f"Re-create boundary groups (fixed bottom 15%, mobile rest).\n"
            f"fix freeze fixed_atoms setforce 0 0 0\n"
            f"Apply spherical indenter from above using fix indent:\n"
            f"  fix indent_fix mobile_atoms indent {K} sphere "
            f"{nx*3.165/2:.2f} {ny*3.165/2:.2f} v_indenter_z {R}\n"
            f"where v_indenter_z decreases from {z_top:.2f} at velocity {vel} Ang/timestep.\n"
            f"Use variable: variable indenter_z equal {z_top:.2f}-{vel}*step*dt\n"
            f"NVE integration for mobile atoms, Langevin thermostat on thermo layer.\n"
            f"timestep 0.001\n"
            f"thermo_style custom step temp f_indent_fix[1] f_indent_fix[2] f_indent_fix[3] "
            f"v_indenter_z pe\n"
            f"thermo 100\n"
            f"Run {nsteps} steps.\n"
            f"{dump_cmd}"
        )

"""Uniaxial tensile test workflow for BCC tungsten — deterministic template edition."""

from __future__ import annotations

import random
import sys

from ..pipeline import PipelineResult, PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig

_A0 = 3.165  # BCC-W lattice constant (Angstrom)


class TensileWorkflow(BaseWorkflow):
    """3-stage pipeline: minimize → equilibrate (NPT) → uniaxial deform.

    All three stages use deterministic script_template — no LLM free-generation
    for core LAMMPS commands.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        strain_rate: float = 1e8,
        max_strain: float = 0.3,
        supercell: tuple[int, int, int] = (10, 10, 20),
        deform_direction: str = "z",
        equilibration_steps: int | None = None,
        timestep: float | None = None,
        thermo_every: int | None = None,
        dump_every: int | None = None,
    ) -> None:
        super().__init__(config, agent)
        self.strain_rate = strain_rate
        self.max_strain = max_strain
        self.supercell = supercell
        self.deform_direction = deform_direction
        self.equilibration_steps = (
            equilibration_steps if equilibration_steps is not None
            else config.equilibration_steps
        )
        self.timestep = timestep if timestep is not None else config.timestep
        self.thermo_every = thermo_every if thermo_every is not None else config.thermo_every
        self.dump_every = dump_every if dump_every is not None else config.dump_every

    # Report config — set by main.py after construction (optional)
    _report_config: "TensileReportConfig | None" = None  # type: ignore[name-defined]

    def run(self) -> PipelineResult:
        pipeline = self.build_pipeline()
        result = pipeline.run()
        if result.success:
            self._try_generate_report(result)
        return result

    def _try_generate_report(self, result: PipelineResult) -> None:
        try:
            from ..report import TensileReporter, TensileReportConfig
            cfg = self._report_config or TensileReportConfig(
                supercell=self.supercell,
                temperature=self.config.temperature,
                potential_name=self.config.potential_name,
                equilibration_steps=self.equilibration_steps,
                timestep=self.timestep,
                thermo_every=self.thermo_every,
                dump_every=self.dump_every,
                deform_direction=self.deform_direction,
                strain_rate=self.strain_rate,
                max_strain=self.max_strain,
                lammps_command=self.agent.config.lammps_command,
            )
            TensileReporter(cfg).generate(result)
        except Exception as exc:
            print(f"[REPORT WARNING] Report generation failed: {exc}", file=sys.stderr)

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
                name="deform",
                task_description="deform (template)",
                depends_on="equilibrate",
                input_files=["restart.equilibrate"],
                script_template=self._deform_script(pot_path),
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

    # ------------------------------------------------------------------
    # Stage scripts
    # ------------------------------------------------------------------

    def _minimize_script(self, pot_path: str) -> str:
        nx, ny, nz = self.supercell
        thermo = self.thermo_every
        return f"""\
# === minimize stage: BCC-W supercell {nx}x{ny}x{nz} ===
units metal
atom_style atomic
boundary p p p

lattice bcc {_A0}
region simbox block 0 {nx} 0 {ny} 0 {nz} units lattice
create_box 1 simbox
create_atoms 1 box

mass 1 183.84
pair_style eam/alloy
pair_coeff * * {pot_path} W

thermo {thermo}
thermo_style custom step pe ke etotal fmax fnorm press

min_style cg
minimize 1e-10 1e-10 10000 100000

write_restart restart.minimize
"""

    def _equilibrate_script(self, pot_path: str) -> str:
        T = self.config.temperature
        steps = self.equilibration_steps
        dt = self.timestep
        thermo = self.thermo_every
        seed = random.randint(10000, 99999)
        return f"""\
# === equilibrate stage: NPT at {T} K ===
read_restart restart.minimize

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

velocity all create {T} {seed} dist gaussian

# NPT: couple all axes, Tdamp = 100*dt, Pdamp = 1000*dt
fix npt_eq all npt temp {T} {T} {100*dt:.6f} iso 0.0 0.0 {1000*dt:.6f}

timestep {dt}
thermo {thermo}
thermo_style custom step temp press pe ke etotal vol lx ly lz

run {steps}

write_restart restart.equilibrate
"""

    def _deform_script(self, pot_path: str) -> str:
        T = self.config.temperature
        d = self.deform_direction
        dt = self.timestep
        thermo = self.thermo_every

        # erate for fix deform: strain_rate [1/s] → [1/ps] (metal units)
        erate = self.strain_rate * 1e-12
        nsteps = int(self.max_strain / (erate * dt))

        seed = random.randint(10000, 99999)

        # Transverse axes
        trans = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
        t1, t2 = trans.get(d, ("x", "y"))

        # Dump command
        if self.dump_every is not None:
            dump_line = (
                f"dump 1 all custom {self.dump_every} "
                f"dump.tensile.lammpstrj id type x y z"
            )
        else:
            dump_line = "# dump disabled (no --dump-every specified)"

        return f"""\
# === deform stage: uniaxial tensile along {d} ===
# strain_rate = {self.strain_rate:.3e} /s  →  erate = {erate:.6e} /ps
# max_strain  = {self.max_strain}  →  nsteps = {nsteps}
read_restart restart.equilibrate

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

velocity all create {T} {seed} dist gaussian

# fix deform on loading axis; fix npt relaxes transverse axes at zero pressure
fix deform_fix all deform 1 {d} erate {erate:.6e} remap x
fix npt_trans  all npt temp {T} {T} {100*dt:.6f} &
    {t1} 0.0 0.0 {1000*dt:.6f} {t2} 0.0 0.0 {1000*dt:.6f}

timestep {dt}
thermo {thermo}
thermo_style custom step temp pxx pyy pzz pxy pxz pyz lx ly lz pe ke etotal

{dump_line}

run {nsteps}
"""

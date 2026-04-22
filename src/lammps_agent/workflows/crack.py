"""Crack propagation workflow for BCC tungsten — deterministic template edition.

All LAMMPS scripts are generated from Python templates with pre-computed
geometry values. No LLM free-generation for core commands. Stages run
fail-fast on error (no retry).
"""

from __future__ import annotations

import random
import sys

from ..pipeline import PipelineResult, PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig

# BCC tungsten lattice constant (Angstrom)
_A0 = 3.165


class CrackWorkflow(BaseWorkflow):
    """4-stage pipeline: build_crack → minimize → load → analyze.

    All geometry is pre-computed in Python; LAMMPS scripts contain only
    numeric literals (no expression strings like '5*3.165').
    """

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        supercell: tuple[int, int, int] = (30, 15, 5),
        crack_length_fraction: float = 0.3,
        dump_every: int | None = None,
        dump_split: bool = False,
    ) -> None:
        super().__init__(config, agent)
        self.supercell = supercell
        self.crack_length_fraction = crack_length_fraction
        self.dump_every = dump_every if dump_every is not None else config.dump_every
        self.dump_split = dump_split

    # Report config — set by main.py after construction
    _report_config: "CrackReportConfig | None" = None  # type: ignore[name-defined]

    def run(self) -> PipelineResult:
        pipeline = self.build_pipeline()
        result = pipeline.run()
        if result.success:
            self._try_generate_report(result)
        return result

    def _try_generate_report(self, result: PipelineResult) -> None:
        try:
            from ..report import CrackReporter, CrackReportConfig
            cfg = self._report_config or CrackReportConfig(
                supercell=self.supercell,
                temperature=self.config.temperature,
                potential_name=self.config.potential_name,
                crack_length_fraction=self.crack_length_fraction,
                lammps_command=self.agent.config.lammps_command,
            )
            CrackReporter(cfg).generate(result)
        except Exception as exc:
            print(f"[REPORT WARNING] Crack report failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def build_pipeline(self) -> SimulationPipeline:
        pot_path = self._resolve_pot_path()
        stages = [
            PipelineStage(
                name="build_crack",
                task_description="build_crack (template)",
                script_template=self._build_crack_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
            PipelineStage(
                name="minimize",
                task_description="minimize (template)",
                depends_on="build_crack",
                input_files=["restart.cracked"],
                script_template=self._minimize_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
            PipelineStage(
                name="load",
                task_description="load (template)",
                depends_on="minimize",
                input_files=["restart.minimized"],
                script_template=self._load_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
            PipelineStage(
                name="analyze",
                task_description="analyze (template)",
                depends_on="load",
                input_files=["restart.loaded"],
                script_template=self._analyze_script(pot_path),
                timeout=self.config.timeout_per_stage,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _resolve_pot_path(self) -> str:
        """Return the absolute path to W_zhou.eam.alloy."""
        pm = PotentialManager()
        pot = pm.resolve_potential(self.config.potential_name or "W_Zhou2004_EAM")
        return str(pot.file_path)

    def _geometry(self) -> dict:
        """Pre-compute all geometry values used across stages."""
        nx, ny, nz = self.supercell
        lx = nx * _A0
        ly = ny * _A0
        lz = nz * _A0

        frac = self.crack_length_fraction
        crack_xhi = frac * lx          # crack extends from x=0 to crack_xhi
        crack_notch_hw = 0.5 * _A0     # half-width of notch slit (one lattice spacing)
        crack_ylo = ly / 2.0 - crack_notch_hw
        crack_yhi = ly / 2.0 + crack_notch_hw

        upper_ylo = 0.9 * ly           # top 10 % → loading grip
        lower_yhi = 0.1 * ly           # bottom 10 % → loading grip

        return dict(
            nx=nx, ny=ny, nz=nz,
            lx=lx, ly=ly, lz=lz,
            crack_xhi=crack_xhi,
            crack_ylo=crack_ylo, crack_yhi=crack_yhi,
            upper_ylo=upper_ylo,
            lower_yhi=lower_yhi,
        )

    # ------------------------------------------------------------------
    # Stage scripts
    # ------------------------------------------------------------------

    def _build_crack_script(self, pot_path: str) -> str:
        g = self._geometry()
        return f"""\
# === build_crack stage: BCC-W crack specimen ===
# Supercell {g['nx']}x{g['ny']}x{g['nz']}, crack fraction {self.crack_length_fraction}
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

# Pre-crack notch: delete atoms in thin slit at y = ly/2
# All coordinates are numeric (pre-computed in Python)
region crack_notch block 0.0 {g['crack_xhi']:.6f} {g['crack_ylo']:.6f} {g['crack_yhi']:.6f} INF INF units box
delete_atoms region crack_notch

# Boundary grip groups (top/bottom 10 % in y)
region upper_grip block INF INF {g['upper_ylo']:.6f} INF INF INF units box
region lower_grip block INF INF INF {g['lower_yhi']:.6f} INF INF units box
group upper_atoms region upper_grip
group lower_atoms region lower_grip
group mobile_atoms subtract all upper_atoms lower_atoms

thermo 10
thermo_style custom step pe ke etotal press vol
write_restart restart.cracked
"""

    def _minimize_script(self, pot_path: str) -> str:
        return f"""\
# === minimize stage: relax cracked specimen ===
read_restart restart.cracked

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

thermo 100
thermo_style custom step pe ke etotal fmax fnorm press

min_style cg
minimize 1e-8 1e-8 20000 200000

write_restart restart.minimized
"""

    def _load_script(self, pot_path: str) -> str:
        g = self._geometry()
        T = self.config.temperature
        seed = random.randint(10000, 99999)

        if self.dump_every is not None:
            if self.dump_split:
                dump_line = f"dump 1 all custom {self.dump_every} dump.crack.*.lammpstrj id type x y z"
            else:
                dump_line = f"dump 1 all custom {self.dump_every} dump.crack.lammpstrj id type x y z"
        else:
            dump_line = "dump 1 all custom 1000 dump.crack.lammpstrj id type x y z"

        return f"""\
# === load stage: Mode-I displacement loading ===
read_restart restart.minimized

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

# Re-create grip groups (same numeric bounds as build_crack)
region upper_grip block INF INF {g['upper_ylo']:.6f} INF INF INF units box
region lower_grip block INF INF INF {g['lower_yhi']:.6f} INF INF units box
group upper_atoms region upper_grip
group lower_atoms region lower_grip
group mobile_atoms subtract all upper_atoms lower_atoms

# Initialise velocities for mobile atoms only
velocity mobile_atoms create {T} {seed} dist gaussian

# Mode-I loading: constant-force pull on grip layers
fix pull_up   upper_atoms aveforce 0.0  0.001 0.0
fix pull_down lower_atoms aveforce 0.0 -0.001 0.0

# NVE for mobile atoms
fix nve_mob mobile_atoms nve

timestep 0.001
thermo 100
thermo_style custom step temp pyy ly pe

{dump_line}

run 50000

write_restart restart.loaded
"""

    def _analyze_script(self, pot_path: str) -> str:
        return f"""\
# === analyze stage: identify crack tip via CNA + coord number ===
read_restart restart.loaded

pair_style eam/alloy
pair_coeff * * {pot_path} W
mass 1 183.84

compute coord all coord/atom cutoff 3.5
compute cna   all cna/atom   3.5

dump 1 all custom 1 dump.crack_final.lammpstrj id type x y z c_coord c_cna

run 0
"""

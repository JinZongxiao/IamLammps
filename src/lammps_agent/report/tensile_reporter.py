"""Tensile workflow report generator.

Produces tensile_report.md, tensile_report.json, stress_strain.png,
stress_strain.csv in the final (deform) run_dir after a successful
tensile workflow run.

Design principles:
- Zero new required dependencies beyond numpy/matplotlib (already in env).
- If any section fails, the failure is recorded in the report as "未提供/未提取"
  rather than raising an exception and aborting the workflow.
- OVITO analysis is attempted via ovito_analyzer; gracefully degraded.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..pipeline import PipelineResult
from ..skills.stress_strain import StressStrainSkill
from ..skills.convergence import ConvergenceCheckSkill
from ..skills.thermo_validator import ThermoValidatorSkill
from ..skills.thermo_tail import ThermoTailSkill
from ..thermo_parser import parse_thermo_table
from .ovito_analyzer import analyze_tensile_dump


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TensileReportConfig:
    """Parameters from the workflow that go into the report header."""
    supercell: tuple[int, int, int] = (10, 10, 20)
    temperature: float = 300.0
    potential_name: str = "W_Zhou2004_EAM"
    equilibration_steps: int = 20000
    timestep: float = 0.001
    thermo_every: int = 100
    dump_every: Optional[int] = None
    deform_direction: str = "z"
    strain_rate: float = 1e8
    max_strain: float = 0.3
    lammps_command: str = "lmp"
    run_timestamp: str = ""


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class _MechanicsResult:
    youngs_modulus_GPa: Optional[float] = None
    yield_strength_GPa: Optional[float] = None
    uts_GPa: Optional[float] = None
    max_strain: Optional[float] = None
    data_points: int = 0
    peak_strain: Optional[float] = None
    post_peak_softening: Optional[bool] = None
    deform_direction_note: str = ""
    note: str = ""


@dataclass
class _StabilityResult:
    convergence_summary: str = "未提取"
    convergence_details: dict = field(default_factory=dict)
    thermo_validator_summary: str = "未提取"
    thermo_validator_warnings: list = field(default_factory=list)
    temperature_stable: Optional[bool] = None
    has_nan_inf: bool = False
    credibility: str = "未评估"


@dataclass
class _StageInfo:
    name: str
    run_dir: str
    return_code: Optional[int]
    retry_count: int
    success: bool


# ---------------------------------------------------------------------------
# Main reporter class
# ---------------------------------------------------------------------------

class TensileReporter:
    """Generate a standardized research report for a tensile workflow run."""

    def __init__(self, config: TensileReportConfig) -> None:
        self.config = config

    def generate(self, pipeline_result: PipelineResult) -> Optional[Path]:
        """Entry point. Returns path to tensile_report.md, or None on failure."""
        if not pipeline_result.success:
            _warn("Pipeline did not succeed; report not generated.")
            return None

        # Find the deform stage output (last stage)
        deform_output = None
        stage_infos: list[_StageInfo] = []
        for name, output in pipeline_result.stages:
            rc = output.lammps_result.return_code if output.lammps_result else None
            stage_infos.append(_StageInfo(
                name=name,
                run_dir=str(output.run_dir),
                return_code=rc,
                retry_count=output.retry_count,
                success=(rc == 0),
            ))
            if name == "deform":
                deform_output = output

        if deform_output is None:
            _warn("No 'deform' stage found in pipeline result; report not generated.")
            return None

        run_dir = deform_output.run_dir
        _info(f"Generating tensile report in {run_dir}")

        # ----------------------------------------------------------------
        # 1. Mechanics from stress_strain skill (re-run for fresh data)
        # ----------------------------------------------------------------
        mechanics = self._extract_mechanics(run_dir)

        # ----------------------------------------------------------------
        # 2. Thermo data for plotting
        # ----------------------------------------------------------------
        strain_arr, stress_arr = self._load_stress_strain_data(run_dir)

        # ----------------------------------------------------------------
        # 3. Stability / physical checks (re-run skills for fresh data)
        # ----------------------------------------------------------------
        stability = self._extract_stability(run_dir)

        # ----------------------------------------------------------------
        # 4. Generate plots + CSV
        # ----------------------------------------------------------------
        plot_files = self._generate_plots(
            run_dir, strain_arr, stress_arr,
            mechanics.yield_strength_GPa,
            mechanics.uts_GPa,
            mechanics.youngs_modulus_GPa,
        )

        # ----------------------------------------------------------------
        # 5. File artifacts inventory
        # ----------------------------------------------------------------
        artifacts = self._inventory_files(run_dir, plot_files)

        # ----------------------------------------------------------------
        # 6. OVITO analysis
        # ----------------------------------------------------------------
        yield_strain = mechanics.peak_strain  # reuse peak_strain as proxy if yield not set
        uts_strain = self._find_uts_strain(strain_arr, stress_arr)
        ovito_result = analyze_tensile_dump(
            run_dir,
            list(strain_arr) if strain_arr is not None else [],
            mechanics.yield_strength_GPa and self._find_yield_strain(
                strain_arr, stress_arr, mechanics.youngs_modulus_GPa
            ),
            uts_strain,
        )

        # Update artifact list with OVITO images
        for img_name in ovito_result.keyframe_images.values():
            artifacts.append(img_name)

        # ----------------------------------------------------------------
        # 7. Count retry info
        # ----------------------------------------------------------------
        any_retry = any(s.retry_count > 0 for s in stage_infos)
        retry_detail = [
            f"{s.name}阶段 retry={s.retry_count} 次"
            for s in stage_infos if s.retry_count > 0
        ]

        # ----------------------------------------------------------------
        # 8. Atom count estimate
        # ----------------------------------------------------------------
        nx, ny, nz = self.config.supercell
        atom_count = nx * ny * nz * 2  # BCC has 2 atoms/unit cell

        # ----------------------------------------------------------------
        # 9. Assemble JSON payload
        # ----------------------------------------------------------------
        report_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json_payload = {
            "report_generated_at": report_ts,
            "workflow": "tensile",
            "run_dir": str(run_dir),
            "success": True,
            # --- A. Run info ---
            "run_info": {
                "supercell": list(self.config.supercell),
                "atom_count_estimated": atom_count,
                "temperature_K": self.config.temperature,
                "timestep_ps": self.config.timestep,
                "equilibration_steps": self.config.equilibration_steps,
                "thermo_every": self.config.thermo_every,
                "dump_every": self.config.dump_every,
                "potential": self.config.potential_name,
                "lammps_command": self.config.lammps_command,
                "deform_direction": self.config.deform_direction,
                "strain_rate_per_s": self.config.strain_rate,
                "any_retry": any_retry,
                "retry_detail": retry_detail,
            },
            # --- B. Mechanics ---
            "mechanics": {
                "youngs_modulus_GPa": mechanics.youngs_modulus_GPa,
                "yield_strength_GPa": mechanics.yield_strength_GPa,
                "uts_GPa": mechanics.uts_GPa,
                "max_strain": mechanics.max_strain,
                "data_points": mechanics.data_points,
                "peak_strain": mechanics.peak_strain,
                "post_peak_softening": mechanics.post_peak_softening,
                "deform_direction_note": mechanics.deform_direction_note,
                "note": mechanics.note,
            },
            # --- C. Files ---
            "output_files": {
                "stress_strain_png": "stress_strain.png" if "stress_strain.png" in artifacts else "未生成",
                "stress_strain_csv": "stress_strain.csv" if "stress_strain.csv" in artifacts else "未生成",
                "tensile_report_md": "tensile_report.md",
                "tensile_report_json": "tensile_report.json",
                "all_artifacts": artifacts,
            },
            # --- D. Stability ---
            "stability": {
                "convergence_summary": stability.convergence_summary,
                "convergence_details": stability.convergence_details,
                "thermo_validator_summary": stability.thermo_validator_summary,
                "thermo_validator_warnings": stability.thermo_validator_warnings,
                "temperature_stable": stability.temperature_stable,
                "has_nan_inf": stability.has_nan_inf,
                "credibility": stability.credibility,
            },
            # --- E. Stages ---
            "stages": [asdict(s) for s in stage_infos],
            # --- F. OVITO ---
            "ovito_analysis": {
                "available": ovito_result.available,
                "unavailable_reason": ovito_result.unavailable_reason,
                "keyframe_images": ovito_result.keyframe_images,
                "structural_stats": ovito_result.structural_stats,
                "summary": ovito_result.summary,
                "error": ovito_result.error,
            },
        }

        # ----------------------------------------------------------------
        # 10. Write JSON
        # ----------------------------------------------------------------
        json_path = run_dir / "tensile_report.json"
        try:
            json_path.write_text(
                json.dumps(json_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            _warn(f"Failed to write tensile_report.json: {exc}")

        # ----------------------------------------------------------------
        # 11. Write Markdown
        # ----------------------------------------------------------------
        md_path = run_dir / "tensile_report.md"
        try:
            md_content = self._render_markdown(
                json_payload, mechanics, stability, ovito_result, stage_infos, atom_count
            )
            md_path.write_text(md_content, encoding="utf-8")
        except Exception as exc:
            _warn(f"Failed to write tensile_report.md: {exc}")
            return None

        _info(f"Report written: {md_path}")
        return md_path

    # ------------------------------------------------------------------
    # Mechanics extraction
    # ------------------------------------------------------------------

    def _extract_mechanics(self, run_dir: Path) -> _MechanicsResult:
        try:
            skill = StressStrainSkill()
            report = skill.analyze(run_dir)
            d = report.details
            m = _MechanicsResult(
                youngs_modulus_GPa=d.get("youngs_modulus_GPa"),
                yield_strength_GPa=d.get("yield_strength_GPa"),
                uts_GPa=d.get("ultimate_tensile_strength_GPa"),
                max_strain=d.get("max_strain"),
                data_points=d.get("data_points", 0),
                deform_direction_note=(
                    f"Tensile axis {self.config.deform_direction.upper()}, "
                    f"stress component P{self.config.deform_direction.lower()}{self.config.deform_direction.lower()}"
                ),
            )
            # Detect post-peak softening
            strain_arr, stress_arr = self._load_stress_strain_data(run_dir)
            if strain_arr is not None and len(stress_arr) > 5:
                peak_idx = int(np.argmax(stress_arr))
                m.peak_strain = round(float(strain_arr[peak_idx]), 4)
                if peak_idx < len(stress_arr) - 3:
                    tail_mean = float(np.mean(stress_arr[peak_idx:]))
                    if tail_mean < float(stress_arr[peak_idx]) * 0.95:
                        m.post_peak_softening = True
                    else:
                        m.post_peak_softening = False
            return m
        except Exception as exc:
            return _MechanicsResult(note=f"力学数据提取失败: {exc}")

    # ------------------------------------------------------------------
    # Thermo data loading
    # ------------------------------------------------------------------

    def _load_stress_strain_data(self, run_dir: Path):
        """Returns (strain_arr, stress_arr) as numpy arrays, or (None, None)."""
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return None, None
        try:
            thermo = parse_thermo_table(stdout_path)
            lz = thermo.get("Lz", [])
            pzz = thermo.get("Pzz", [])
            if len(lz) < 5 or len(pzz) < 5:
                return None, None
            lz0 = lz[0]
            strain = np.array([(l - lz0) / lz0 for l in lz])
            stress = np.array([-p * 1e-4 for p in pzz])  # bar → GPa, tensile positive
            return strain, stress
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Stability checks
    # ------------------------------------------------------------------

    def _extract_stability(self, run_dir: Path) -> _StabilityResult:
        s = _StabilityResult()
        try:
            conv = ConvergenceCheckSkill().analyze(run_dir)
            s.convergence_summary = conv.summary
            s.convergence_details = conv.details.get("convergence", {})
        except Exception as exc:
            s.convergence_summary = f"检查失败: {exc}"

        try:
            val = ThermoValidatorSkill().analyze(run_dir)
            s.thermo_validator_summary = val.summary
            s.thermo_validator_warnings = val.details.get("warnings", [])
            s.has_nan_inf = any("NaN" in w or "Inf" in w for w in s.thermo_validator_warnings)
        except Exception as exc:
            s.thermo_validator_summary = f"检查失败: {exc}"

        # Temperature stability: check if Temp tail is within ±20% of target
        try:
            stdout_path = run_dir / "stdout.log"
            if stdout_path.exists():
                thermo = parse_thermo_table(stdout_path)
                temps = thermo.get("Temp", [])
                if temps and len(temps) >= 10:
                    tail = temps[int(len(temps) * 0.8):]
                    mean_t = float(np.mean(tail))
                    target_t = self.config.temperature
                    if target_t > 0:
                        s.temperature_stable = abs(mean_t - target_t) / target_t < 0.20
        except Exception:
            pass

        # Credibility assessment
        issues = []
        if s.has_nan_inf:
            issues.append("能量/压力含NaN/Inf")
        if s.thermo_validator_warnings:
            issues.append(f"{len(s.thermo_validator_warnings)}个物理量警告")
        if s.temperature_stable is False:
            issues.append("温度偏离目标值 >20%")

        if not issues:
            s.credibility = "结果可信（无物理异常，热力学量正常）"
        else:
            s.credibility = f"结果可疑，存在以下问题: {'; '.join(issues)}"

        return s

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _generate_plots(
        self, run_dir: Path, strain_arr, stress_arr,
        yield_strength, uts, youngs_modulus
    ) -> list[str]:
        """Generate stress_strain.png (with annotations) and stress_strain.csv."""
        produced: list[str] = []
        if strain_arr is None or len(strain_arr) < 5:
            _warn("Insufficient data for plotting; skipping plots.")
            return produced

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(strain_arr * 100, stress_arr, color="#1f77b4", lw=1.5, label="Stress-Strain")

            # Mark yield point
            if youngs_modulus is not None and youngs_modulus > 0 and yield_strength is not None:
                yield_strain_val = self._find_yield_strain(strain_arr, stress_arr, youngs_modulus)
                if yield_strain_val is not None:
                    ax.axvline(yield_strain_val * 100, color="orange", ls="--", lw=1,
                               label=f"Yield sigma_y={yield_strength:.2f} GPa")
                    ax.scatter([yield_strain_val * 100], [yield_strength],
                               color="orange", zorder=5, s=60)

            # Mark UTS
            if uts is not None:
                uts_strain_val = self._find_uts_strain(strain_arr, stress_arr)
                if uts_strain_val is not None:
                    ax.axvline(uts_strain_val * 100, color="red", ls="--", lw=1,
                               label=f"UTS={uts:.2f} GPa")
                    ax.scatter([uts_strain_val * 100], [uts],
                               color="red", zorder=5, s=60)

            ax.set_xlabel("Engineering Strain (%)", fontsize=12)
            ax.set_ylabel("True Stress (GPa)", fontsize=12)
            title_parts = [f"W BCC Uniaxial Tensile  T={self.config.temperature:.0f} K"]
            if youngs_modulus:
                title_parts.append(f"E={youngs_modulus:.0f} GPa")
            ax.set_title("  |  ".join(title_parts), fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            png_path = run_dir / "stress_strain.png"
            fig.savefig(str(png_path), dpi=150)
            plt.close(fig)
            produced.append("stress_strain.png")
        except Exception as exc:
            _warn(f"Plot generation failed: {exc}")

        # CSV export
        try:
            csv_path = run_dir / "stress_strain.csv"
            lines = ["strain,stress_GPa"]
            for eps, sig in zip(strain_arr, stress_arr):
                lines.append(f"{eps:.6f},{sig:.6f}")
            csv_path.write_text("\n".join(lines), encoding="utf-8")
            produced.append("stress_strain.csv")
        except Exception as exc:
            _warn(f"CSV export failed: {exc}")

        return produced

    # ------------------------------------------------------------------
    # File inventory
    # ------------------------------------------------------------------

    def _inventory_files(self, run_dir: Path, extra_files: list[str]) -> list[str]:
        """List key files present in run_dir."""
        candidates = [
            "in.lammps", "stdout.log", "stderr.log", "log.lammps", "result.json",
            "restart.equilibrate", "restart.minimize",
            "stress_strain.png", "stress_strain.csv",
            "tensile_report.md", "tensile_report.json",
        ]
        # Add retry scripts
        for retry_f in run_dir.glob("in.lammps.retry*"):
            candidates.append(retry_f.name)
        # Add dump files (just count them)
        dump_count = len(list(run_dir.glob("dump.tensile.*.lammpstrj")))

        present = [f for f in candidates if (run_dir / f).exists()]
        for ef in extra_files:  # OVITO images etc. — avoid duplicates
            if ef not in present:
                present.append(ef)
        if dump_count > 0:
            present.append(f"dump.tensile.*.lammpstrj (共 {dump_count} 帧)")
        return present

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def _render_markdown(
        self, payload: dict, mechanics: _MechanicsResult,
        stability: _StabilityResult, ovito_result,
        stage_infos: list[_StageInfo], atom_count: int
    ) -> str:
        cfg = self.config
        ri = payload["run_info"]
        ts = payload["report_generated_at"]
        nx, ny, nz = cfg.supercell

        def _val(v, unit="", fmt=".3f"):
            if v is None:
                return "未提取"
            return f"{v:{fmt}} {unit}".strip()

        lines: list[str] = []

        lines += [
            f"# IamLAMMPS 拉伸仿真自动报告",
            f"",
            f"> **生成时间：** {ts}  ",
            f"> **工作流：** tensile (W BCC 单轴拉伸)  ",
            f"> **run_dir：** `{payload['run_dir']}`",
            f"",
        ]

        # --- A. 基本运行信息 ---
        lines += [
            f"## A. 基本运行信息",
            f"",
            f"| 参数 | 值 |",
            f"|------|-----|",
            f"| 超胞 | {nx}×{ny}×{nz} |",
            f"| 原子数（估算，BCC 2个/晶胞） | {atom_count} |",
            f"| 温度 | {cfg.temperature:.0f} K |",
            f"| 时间步长 | {cfg.timestep} ps |",
            f"| 平衡步数 | {cfg.equilibration_steps} |",
            f"| 热力学输出频率 | 每 {cfg.thermo_every} 步 |",
            f"| dump 频率 | {'每 ' + str(cfg.dump_every) + ' 步' if cfg.dump_every else '未启用'} |",
            f"| 势函数 | {cfg.potential_name} |",
            f"| LAMMPS 命令 | `{cfg.lammps_command}` |",
            f"| 拉伸方向 | {cfg.deform_direction.upper()} 轴 |",
            f"| 应变速率 | {cfg.strain_rate:.2e} /s |",
            f"| 是否发生 retry | {'是' if ri['any_retry'] else '否'} |",
        ]
        if ri["retry_detail"]:
            lines.append(f"| retry 详情 | {'; '.join(ri['retry_detail'])} |")
        lines.append("")

        # Stage table
        lines += [
            f"### 各阶段状态",
            f"",
            f"| 阶段 | run_dir | 返回码 | retry 次数 | 状态 |",
            f"|------|---------|--------|-----------|------|",
        ]
        for s in stage_infos:
            status = "✓ 成功" if s.success else "✗ 失败"
            lines.append(
                f"| {s.name} | `{s.run_dir}` | {s.return_code} | {s.retry_count} | {status} |"
            )
        lines.append("")

        # --- B. 力学结果 ---
        lines += [
            f"## B. 拉伸力学结果",
            f"",
            f"| 物理量 | 值 |",
            f"|--------|-----|",
            f"| 杨氏模量 E | {_val(mechanics.youngs_modulus_GPa, 'GPa')} |",
            f"| 屈服强度 σ_y（0.2% offset） | {_val(mechanics.yield_strength_GPa, 'GPa')} |",
            f"| 极限抗拉强度 UTS | {_val(mechanics.uts_GPa, 'GPa')} |",
            f"| 最大应变 | {_val(mechanics.max_strain, fmt='.4f')} |",
            f"| 峰值应变 | {_val(mechanics.peak_strain, fmt='.4f')} |",
            f"| 数据点数 | {mechanics.data_points} |",
            f"| 峰值后软化 | {'是' if mechanics.post_peak_softening else ('否' if mechanics.post_peak_softening is False else '未判断')} |",
            f"| 拉伸方向说明 | {mechanics.deform_direction_note or '未提供'} |",
        ]
        if mechanics.note:
            lines.append(f"| 备注 | {mechanics.note} |")
        lines.append("")

        # --- C. 曲线与数据导出 ---
        of = payload["output_files"]
        lines += [
            f"## C. 曲线与数据导出",
            f"",
            f"- **应力-应变曲线图：** `{of['stress_strain_png']}`",
            f"- **原始数据 CSV：** `{of['stress_strain_csv']}`",
            f"",
            f"> 图表已标注屈服点（橙色虚线）与 UTS（红色虚线）。",
            f"",
        ]

        # --- D. 稳定性检查 ---
        lines += [
            f"## D. 运行稳定性与物理性检查",
            f"",
            f"| 检查项 | 结果 |",
            f"|--------|------|",
            f"| 收敛性检查 | {stability.convergence_summary} |",
            f"| 物理量范围检查 | {stability.thermo_validator_summary} |",
            f"| 温度稳定性 | {'稳定（尾段均值偏差 <20%）' if stability.temperature_stable else ('不稳定' if stability.temperature_stable is False else '未评估')} |",
            f"| 含 NaN/Inf | {'是（⚠ 模拟不稳定）' if stability.has_nan_inf else '否'} |",
            f"",
            f"**可信度判断：** {stability.credibility}",
            f"",
        ]
        if stability.thermo_validator_warnings:
            lines.append("**物理量警告列表：**")
            for w in stability.thermo_validator_warnings:
                lines.append(f"- {w}")
            lines.append("")

        # --- E. 文件产物清单 ---
        lines += [
            f"## E. 文件产物清单",
            f"",
        ]
        for f_name in of["all_artifacts"]:
            lines.append(f"- `{f_name}`")
        lines.append("")

        # --- F. OVITO 结构演化分析 ---
        lines += [
            f"## F. 原子结构演化分析（OVITO）",
            f"",
        ]
        if not ovito_result.available:
            lines += [
                f"> **OVITO analysis unavailable**",
                f">",
                f"> 原因：{ovito_result.unavailable_reason}",
                f">",
                f"> 如需启用，请在 iam-lammps 环境中安装：`pip install ovito`",
                f"",
            ]
        elif ovito_result.error:
            lines += [
                f"> ⚠ OVITO 分析执行失败：{ovito_result.error}",
                f"",
            ]
        else:
            lines += [
                f"**分析方法：** PTM（Polyhedral Template Matching）BCC 结构分类",
                f"",
            ]
            if ovito_result.keyframe_images:
                lines.append("**关键帧图像：**")
                label_map = {
                    "initial": "初始态", "yield": "屈服附近",
                    "uts": "UTS附近", "final": "最终态",
                }
                for label, fname in ovito_result.keyframe_images.items():
                    lines.append(f"- {label_map.get(label, label)}: `{fname}`")
                lines.append("")
            if ovito_result.structural_stats:
                lines += [
                    "**结构演化统计：**",
                    "",
                    "| 时刻 | 近似应变 | BCC 比例 | 非BCC原子数 |",
                    "|------|---------|---------|------------|",
                ]
                label_map = {
                    "initial": "初始态", "yield": "屈服附近",
                    "uts": "UTS附近", "final": "最终态",
                }
                for label, stats in ovito_result.structural_stats.items():
                    eps_str = f"{stats.get('strain_approx', '?'):.3f}" if stats.get("strain_approx") is not None else "?"
                    bcc_frac = stats.get("bcc_fraction")
                    bcc_str = f"{bcc_frac*100:.1f}%" if bcc_frac is not None else "?"
                    n_def = stats.get("n_non_bcc", "?")
                    lines.append(f"| {label_map.get(label, label)} | {eps_str} | {bcc_str} | {n_def} |")
                lines.append("")
            lines += [
                f"**结构演化摘要：**",
                f"",
                f"{ovito_result.summary}",
                f"",
            ]

        # --- 报告结束 ---
        lines += [
            f"---",
            f"",
            f"*本报告由 IamLAMMPS 自动生成。*",
        ]

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _find_uts_strain(strain_arr, stress_arr) -> Optional[float]:
        if strain_arr is None or len(stress_arr) < 2:
            return None
        idx = int(np.argmax(stress_arr))
        return float(strain_arr[idx])

    @staticmethod
    def _find_yield_strain(strain_arr, stress_arr, youngs_modulus) -> Optional[float]:
        if strain_arr is None or youngs_modulus is None or youngs_modulus <= 0:
            return None
        offset_line = youngs_modulus * (strain_arr - 0.002)
        diff = stress_arr - offset_line
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            return float(strain_arr[sign_changes[0]])
        return None


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _info(msg: str) -> None:
    print(f"[REPORT] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    warnings.warn(f"[REPORT] {msg}", stacklevel=2)
    print(f"[REPORT WARNING] {msg}", file=sys.stderr)

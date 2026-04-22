"""Nanoindentation workflow report generator.

Produces:
  nanoindent_report.md / .json
  indent_ph_curve.png
  indent_ph_curve.csv
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..pipeline import PipelineResult
from ..skills.indent_analysis import IndentAnalysisSkill
from ..thermo_parser import parse_thermo_table
from .base_reporter import (
    _StabilityResult, _StageInfo,
    run_stability_checks, inventory_files,
    write_json, write_md, extract_stage_infos,
    report_info, report_warn,
)


@dataclass
class NanoindentReportConfig:
    supercell: tuple[int, int, int] = (20, 20, 30)
    temperature: float = 300.0
    potential_name: str = "W_Zhou2004_EAM"
    indenter_radius: float = 30.0
    indenter_velocity: float = 1e-3
    lammps_command: str = "lmp"


class NanoindentReporter:
    def __init__(self, config: NanoindentReportConfig) -> None:
        self.config = config

    def generate(self, pipeline_result: PipelineResult) -> Optional[Path]:
        if not pipeline_result.success:
            report_warn("Pipeline did not succeed; nanoindent report not generated.")
            return None

        indent_output = next(
            (o for n, o in pipeline_result.stages if n == "indent"), None
        )
        if indent_output is None:
            report_warn("No 'indent' stage found; report skipped.")
            return None

        run_dir = indent_output.run_dir
        report_info(f"Generating nanoindent report in {run_dir}")

        stage_infos = extract_stage_infos(pipeline_result)
        stability = run_stability_checks(run_dir, self.config.temperature)

        # ── Oliver-Pharr results ──────────────────────────────────────────
        indent_result = self._extract_indent(run_dir)

        # ── Raw P-h data ─────────────────────────────────────────────────
        force_arr, disp_arr = self._load_ph_data(run_dir)

        # ── Plots + CSV ───────────────────────────────────────────────────
        plot_files = self._generate_plots(run_dir, force_arr, disp_arr, indent_result)

        # ── File inventory ────────────────────────────────────────────────
        artifacts = inventory_files(run_dir, plot_files)

        # ── Retry info ────────────────────────────────────────────────────
        any_retry = any(s.retry_count > 0 for s in stage_infos)
        retry_detail = [f"{s.name} retry={s.retry_count}" for s in stage_infos if s.retry_count > 0]
        nx, ny, nz = self.config.supercell
        atom_count = nx * ny * nz * 2

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "report_generated_at": ts,
            "workflow": "nanoindent",
            "run_dir": str(run_dir),
            "success": True,
            "run_info": {
                "supercell": list(self.config.supercell),
                "atom_count_estimated": atom_count,
                "temperature_K": self.config.temperature,
                "indenter_radius_A": self.config.indenter_radius,
                "indenter_velocity_A_per_step": self.config.indenter_velocity,
                "potential": self.config.potential_name,
                "lammps_command": self.config.lammps_command,
                "any_retry": any_retry,
                "retry_detail": retry_detail,
            },
            "mechanics": indent_result,
            "output_files": {
                "ph_curve_png": "indent_ph_curve.png" if "indent_ph_curve.png" in artifacts else "未生成",
                "ph_curve_csv": "indent_ph_curve.csv" if "indent_ph_curve.csv" in artifacts else "未生成",
                "report_md": "nanoindent_report.md",
                "report_json": "nanoindent_report.json",
                "all_artifacts": artifacts,
            },
            "stability": {
                "convergence_summary": stability.convergence_summary,
                "convergence_details": stability.convergence_details,
                "thermo_validator_summary": stability.thermo_validator_summary,
                "thermo_validator_warnings": stability.thermo_validator_warnings,
                "temperature_stable": stability.temperature_stable,
                "has_nan_inf": stability.has_nan_inf,
                "credibility": stability.credibility,
            },
            "stages": [
                {"name": s.name, "run_dir": s.run_dir, "return_code": s.return_code,
                 "retry_count": s.retry_count, "success": s.success}
                for s in stage_infos
            ],
        }

        try:
            write_json(run_dir, "nanoindent_report.json", payload)
        except Exception as exc:
            report_warn(f"Failed to write nanoindent_report.json: {exc}")

        try:
            md = self._render_markdown(payload, indent_result, stability, stage_infos, atom_count)
            md_path = write_md(run_dir, "nanoindent_report.md", md)
        except Exception as exc:
            report_warn(f"Failed to write nanoindent_report.md: {exc}")
            return None

        report_info(f"Nanoindent report written: {md_path}")
        return md_path

    # ── Data extraction ───────────────────────────────────────────────────

    def _extract_indent(self, run_dir: Path) -> dict:
        try:
            skill = IndentAnalysisSkill()
            report = skill.analyze(run_dir)
            d = report.details
            return {
                "P_max_nN": d.get("P_max_nN"),
                "h_max_Angstrom": d.get("h_max_Angstrom"),
                "contact_stiffness_nN_per_A": d.get("contact_stiffness_nN_per_A"),
                "hardness_GPa": d.get("hardness_GPa"),
                "reduced_modulus_GPa": d.get("reduced_modulus_GPa"),
                "h_c_Angstrom": "未提取",
                "residual_depth_Angstrom": "未提取",
                "plastic_work_ratio": "未提取",
                "note": report.summary,
            }
        except Exception as exc:
            return {"note": f"Oliver-Pharr 提取失败: {exc}"}

    def _load_ph_data(self, run_dir: Path):
        stdout = run_dir / "stdout.log"
        if not stdout.exists():
            return None, None
        try:
            thermo = parse_thermo_table(stdout)
            force_col = next(
                (c for c in thermo if "indent" in c.lower() and ("[3]" in c or "z" in c.lower())), None
            )
            pos_col = next(
                (c for c in thermo if "indenter_z" in c.lower() or "v_indenter" in c.lower()), None
            )
            if not force_col:
                return None, None
            force_nN = np.abs(np.array(thermo[force_col])) * 1.602
            if pos_col and pos_col in thermo and len(thermo[pos_col]) == len(force_nN):
                pos = np.array(thermo[pos_col])
                disp = pos[0] - pos
            else:
                steps = np.array(thermo.get("Step", list(range(len(force_nN)))))
                disp = steps * self.config.indenter_velocity * 0.001
            return force_nN, disp
        except Exception:
            return None, None

    # ── Plots ─────────────────────────────────────────────────────────────

    def _generate_plots(self, run_dir: Path, force_arr, disp_arr, indent_result: dict) -> list[str]:
        produced = []
        if force_arr is None or len(force_arr) < 5:
            report_warn("Insufficient P-h data; skipping nanoindent plot.")
            return produced
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(disp_arr, force_arr, color="#2ca02c", lw=1.5, label="P-h curve")
            p_max = indent_result.get("P_max_nN")
            h_max = indent_result.get("h_max_Angstrom")
            if p_max and h_max:
                ax.scatter([h_max], [p_max], color="red", zorder=5, s=60,
                           label=f"P_max={p_max:.2f} nN")
            ax.set_xlabel("Displacement h (Ang)", fontsize=12)
            ax.set_ylabel("Load P (nN)", fontsize=12)
            ax.set_title(
                f"W BCC Nanoindentation  T={self.config.temperature:.0f} K  R={self.config.indenter_radius:.0f} A",
                fontsize=10,
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(run_dir / "indent_ph_curve.png"), dpi=150)
            plt.close(fig)
            produced.append("indent_ph_curve.png")
        except Exception as exc:
            report_warn(f"P-h plot failed: {exc}")

        try:
            lines = ["displacement_A,force_nN"]
            for h, p in zip(disp_arr, force_arr):
                lines.append(f"{h:.6f},{p:.6f}")
            (run_dir / "indent_ph_curve.csv").write_text("\n".join(lines), encoding="utf-8")
            produced.append("indent_ph_curve.csv")
        except Exception as exc:
            report_warn(f"P-h CSV export failed: {exc}")

        return produced

    # ── Markdown ──────────────────────────────────────────────────────────

    def _render_markdown(self, payload, indent_result, stability: _StabilityResult,
                         stage_infos: list[_StageInfo], atom_count: int) -> str:
        cfg = self.config
        ts = payload["report_generated_at"]
        nx, ny, nz = cfg.supercell

        def _v(v, unit="", fmt=".3f"):
            return "未提取" if v is None or v == "未提取" else f"{v:{fmt}} {unit}".strip()

        lines = [
            "# IamLAMMPS 纳米压痕仿真自动报告",
            "",
            f"> **生成时间：** {ts}  ",
            f"> **工作流：** nanoindent (W BCC 球形压头)  ",
            f"> **run_dir：** `{payload['run_dir']}`",
            "",
            "## A. 基本运行信息",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
            f"| 超胞 | {nx}×{ny}×{nz} |",
            f"| 原子数（估算） | {atom_count} |",
            f"| 温度 | {cfg.temperature:.0f} K |",
            f"| 压头半径 | {cfg.indenter_radius:.1f} Å |",
            f"| 压头速度 | {cfg.indenter_velocity} Å/step |",
            f"| 势函数 | {cfg.potential_name} |",
            f"| LAMMPS 命令 | `{cfg.lammps_command}` |",
            f"| 是否发生 retry | {'是' if payload['run_info']['any_retry'] else '否'} |",
            "",
            "### 各阶段状态",
            "",
            "| 阶段 | 返回码 | retry | 状态 |",
            "|------|--------|-------|------|",
        ]
        for s in stage_infos:
            lines.append(f"| {s.name} | {s.return_code} | {s.retry_count} | {'✓ 成功' if s.success else '✗ 失败'} |")

        lines += [
            "",
            "## B. Oliver-Pharr 力学结果",
            "",
            "| 物理量 | 值 |",
            "|--------|-----|",
            f"| P_max | {_v(indent_result.get('P_max_nN'), 'nN')} |",
            f"| h_max（压痕深度） | {_v(indent_result.get('h_max_Angstrom'), 'Å')} |",
            f"| 接触刚度 S | {_v(indent_result.get('contact_stiffness_nN_per_A'), 'nN/Å')} |",
            f"| 硬度 H | {_v(indent_result.get('hardness_GPa'), 'GPa')} |",
            f"| 折合模量 E_r | {_v(indent_result.get('reduced_modulus_GPa'), 'GPa')} |",
            f"| 接触深度 h_c | {_v(indent_result.get('h_c_Angstrom'), 'Å')} |",
            f"| 残余压入深度 | {_v(indent_result.get('residual_depth_Angstrom'), 'Å')} |",
            f"| 塑性功/总功比 | {_v(indent_result.get('plastic_work_ratio'))} |",
            "",
            "## C. 载荷-位移曲线与数据导出",
            "",
            f"- **P-h 曲线图：** `{payload['output_files']['ph_curve_png']}`",
            f"- **原始数据 CSV：** `{payload['output_files']['ph_curve_csv']}`",
            "",
            "## D. 运行稳定性检查",
            "",
            "| 检查项 | 结果 |",
            "|--------|------|",
            f"| 收敛性 | {stability.convergence_summary} |",
            f"| 物理量范围 | {stability.thermo_validator_summary} |",
            f"| 温度稳定性 | {'稳定' if stability.temperature_stable else ('不稳定' if stability.temperature_stable is False else '未评估')} |",
            f"| 含 NaN/Inf | {'是 ⚠' if stability.has_nan_inf else '否'} |",
            "",
            f"**可信度：** {stability.credibility}",
            "",
            "## E. 文件产物清单",
            "",
        ]
        for f in payload["output_files"]["all_artifacts"]:
            lines.append(f"- `{f}`")
        lines += ["", "---", "", "*本报告由 IamLAMMPS 自动生成。*", ""]
        return "\n".join(lines)

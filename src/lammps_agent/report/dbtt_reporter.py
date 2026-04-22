"""DBTT sweep workflow report generator.

Aggregates tensile results across all temperature points and produces:
  dbtt_report.md / .json
  dbtt_summary.png
  dbtt_summary.csv
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..pipeline import PipelineResult
from ..skills.stress_strain import StressStrainSkill
from ..skills.dbtt_fit import DBTTFitSkill
from ..thermo_parser import parse_thermo_table
from .base_reporter import (
    write_json, write_md,
    report_info, report_warn,
)


@dataclass
class DBTTReportConfig:
    temperatures: list[float] = field(default_factory=lambda: list(range(200, 1300, 100)))
    supercell: tuple[int, int, int] = (10, 10, 20)
    potential_name: str = "W_Zhou2004_EAM"
    strain_rate: float = 1e8
    lammps_command: str = "lmp"
    # Where to write the aggregated report (caller sets this)
    report_dir: Optional[Path] = None


@dataclass
class _TempResult:
    temperature: float
    youngs_modulus_GPa: Optional[float]
    yield_strength_GPa: Optional[float]
    uts_GPa: Optional[float]
    failure_strain: Optional[float]
    ductility_label: str   # brittle / transition / ductile
    run_dir: str
    deform_success: bool
    retry_count: int


class DBTTReporter:
    def __init__(self, config: DBTTReportConfig) -> None:
        self.config = config

    def generate(self, pipeline_result: PipelineResult) -> Optional[Path]:
        """Generate aggregated DBTT report from a DBTTSweepWorkflow result."""

        # Collect per-temperature results from the flat stage list
        # Stage names are like "T300_deform", "T300_minimize", etc.
        temp_results = self._collect_temp_results(pipeline_result)

        if not temp_results:
            report_warn("No temperature results found; DBTT report skipped.")
            return None

        # Choose report directory: prefer config.report_dir, else first deform dir's parent
        report_dir = self.config.report_dir
        if report_dir is None:
            first_deform = next((r for r in temp_results if r.deform_success), None)
            if first_deform:
                report_dir = Path(first_deform.run_dir).parent
            else:
                report_dir = Path(temp_results[0].run_dir).parent
        report_dir.mkdir(parents=True, exist_ok=True)

        report_info(f"Generating DBTT report in {report_dir}")

        # DBTT sigmoid fit
        fit_result = self._fit_dbtt(temp_results)

        # Plots + CSV
        plot_files = self._generate_plots(report_dir, temp_results, fit_result)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nx, ny, nz = self.config.supercell
        atom_count = nx * ny * nz * 2

        payload = {
            "report_generated_at": ts,
            "workflow": "dbtt",
            "report_dir": str(report_dir),
            "success": pipeline_result.success,
            "run_info": {
                "supercell": list(self.config.supercell),
                "atom_count_estimated": atom_count,
                "temperatures_K": self.config.temperatures,
                "strain_rate_per_s": self.config.strain_rate,
                "potential": self.config.potential_name,
                "lammps_command": self.config.lammps_command,
            },
            "dbtt_fit": fit_result,
            "temperature_results": [
                {
                    "T_K": r.temperature,
                    "E_GPa": r.youngs_modulus_GPa,
                    "sigma_y_GPa": r.yield_strength_GPa,
                    "UTS_GPa": r.uts_GPa,
                    "failure_strain": r.failure_strain,
                    "ductility_label": r.ductility_label,
                    "deform_success": r.deform_success,
                    "retry_count": r.retry_count,
                    "run_dir": r.run_dir,
                }
                for r in temp_results
            ],
            "output_files": {
                "summary_png": "dbtt_summary.png" if (report_dir / "dbtt_summary.png").exists() else "未生成",
                "summary_csv": "dbtt_summary.csv" if (report_dir / "dbtt_summary.csv").exists() else "未生成",
                "report_md": "dbtt_report.md",
                "report_json": "dbtt_report.json",
            },
        }

        try:
            write_json(report_dir, "dbtt_report.json", payload)
        except Exception as exc:
            report_warn(f"Failed to write dbtt_report.json: {exc}")

        try:
            md = self._render_markdown(payload, temp_results, fit_result)
            md_path = write_md(report_dir, "dbtt_report.md", md)
        except Exception as exc:
            report_warn(f"Failed to write dbtt_report.md: {exc}")
            return None

        report_info(f"DBTT report written: {md_path}")
        return md_path

    # ── Per-temperature extraction ────────────────────────────────────────

    def _collect_temp_results(self, pipeline_result: PipelineResult) -> list[_TempResult]:
        """Parse flat stage list into per-temperature mechanics results."""
        # Group stages by temperature prefix T{T}_stagename
        from collections import defaultdict
        groups: dict[float, dict[str, any]] = defaultdict(dict)

        for stage_name, output in pipeline_result.stages:
            if not stage_name.startswith("T"):
                continue
            parts = stage_name.split("_", 1)
            if len(parts) < 2:
                continue
            try:
                T = float(parts[0][1:])
            except ValueError:
                continue
            sname = parts[1]
            groups[T][sname] = output

        results = []
        for T in sorted(groups.keys()):
            g = groups[T]
            deform_out = g.get("deform")
            deform_success = (
                deform_out is not None
                and deform_out.lammps_result is not None
                and deform_out.lammps_result.return_code == 0
            )
            retry_count = deform_out.retry_count if deform_out else 0
            run_dir_str = str(deform_out.run_dir) if deform_out else "未知"

            E, sy, uts, fs = None, None, None, None
            if deform_success and deform_out is not None:
                E, sy, uts, fs = self._extract_mechanics(deform_out.run_dir)

            ductility_label = self._classify_ductility(fs, uts)

            results.append(_TempResult(
                temperature=T,
                youngs_modulus_GPa=E,
                yield_strength_GPa=sy,
                uts_GPa=uts,
                failure_strain=fs,
                ductility_label=ductility_label,
                run_dir=run_dir_str,
                deform_success=deform_success,
                retry_count=retry_count,
            ))
        return results

    def _extract_mechanics(self, run_dir: Path):
        """Return (E, sigma_y, UTS, failure_strain) from stress_strain skill."""
        try:
            skill = StressStrainSkill()
            report = skill.analyze(run_dir)
            d = report.details
            E = d.get("youngs_modulus_GPa")
            sy = d.get("yield_strength_GPa")
            uts = d.get("ultimate_tensile_strength_GPa")
            ms = d.get("max_strain")
            # Use peak_strain as failure_strain proxy
            thermo = parse_thermo_table(run_dir / "stdout.log")
            lz = thermo.get("Lz", [])
            pzz = thermo.get("Pzz", [])
            fs = None
            if len(lz) > 1 and len(pzz) > 1:
                lz0 = lz[0]
                strain = [(l - lz0) / lz0 for l in lz]
                stress = [-p * 1e-4 for p in pzz]
                peak_idx = int(np.argmax(stress))
                fs = round(float(strain[peak_idx]), 4)
            return E, sy, uts, fs
        except Exception:
            return None, None, None, None

    @staticmethod
    def _classify_ductility(failure_strain: Optional[float], uts: Optional[float]) -> str:
        if failure_strain is None:
            return "unknown"
        if failure_strain < 0.02:
            return "brittle"
        elif failure_strain < 0.08:
            return "transition"
        else:
            return "ductile"

    # ── DBTT fit ─────────────────────────────────────────────────────────

    def _fit_dbtt(self, temp_results: list[_TempResult]) -> dict:
        valid = [(r.temperature, r.failure_strain) for r in temp_results
                 if r.failure_strain is not None and r.deform_success]
        if len(valid) < 3:
            return {
                "T_dbtt_K": None,
                "transition_width_K": None,
                "note": f"数据点不足（{len(valid)} / {len(temp_results)}），无法拟合",
            }
        temps = [v[0] for v in valid]
        ductilities = [v[1] for v in valid]
        try:
            skill = DBTTFitSkill()
            report = skill.analyze_sweep(temps, ductilities)
            return {
                "T_dbtt_K": report.details.get("T_dbtt_K"),
                "transition_width_K": report.details.get("transition_width_K"),
                "note": report.summary,
            }
        except Exception as exc:
            return {"T_dbtt_K": None, "note": f"拟合失败: {exc}"}

    # ── Plots ─────────────────────────────────────────────────────────────

    def _generate_plots(self, report_dir: Path, temp_results: list[_TempResult],
                        fit_result: dict) -> list[str]:
        produced = []
        valid = [r for r in temp_results if r.deform_success]
        if len(valid) < 2:
            report_warn("Insufficient data points for DBTT plot.")
            return produced

        temps = [r.temperature for r in valid]
        uts_vals = [r.uts_GPa for r in valid]
        fs_vals = [r.failure_strain for r in valid]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: UTS vs T
            ax = axes[0]
            ax.plot(temps, [u if u else 0 for u in uts_vals],
                    "o-", color="#1f77b4", lw=1.5, label="UTS")
            ax.set_xlabel("Temperature (K)", fontsize=11)
            ax.set_ylabel("UTS (GPa)", fontsize=11)
            ax.set_title("UTS vs Temperature", fontsize=11)
            ax.grid(True, alpha=0.3)

            # Right: failure strain vs T + sigmoid fit
            ax = axes[1]
            fs_clean = [f if f is not None else 0 for f in fs_vals]
            ax.plot(temps, fs_clean, "s-", color="#ff7f0e", lw=1.5, label="Failure strain")

            # Draw sigmoid fit if DBTT available
            T_dbtt = fit_result.get("T_dbtt_K")
            width = fit_result.get("transition_width_K")
            if T_dbtt and width:
                t_fit = np.linspace(min(temps) - 50, max(temps) + 50, 200)
                d_min, d_max = min(fs_clean), max(fs_clean)
                sigmoid = d_min + (d_max - d_min) / (1 + np.exp(-(t_fit - T_dbtt) / max(width, 1)))
                ax.plot(t_fit, sigmoid, "--", color="gray", lw=1, label=f"Sigmoid fit")
                ax.axvline(T_dbtt, color="red", ls=":", lw=1.5,
                           label=f"DBTT ≈ {T_dbtt:.0f} K")

            ax.set_xlabel("Temperature (K)", fontsize=11)
            ax.set_ylabel("Failure strain", fontsize=11)
            ax.set_title("Ductility vs Temperature (DBTT)", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.suptitle(
                f"W BCC DBTT Sweep  {self.config.supercell[0]}x{self.config.supercell[1]}x{self.config.supercell[2]}",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(str(report_dir / "dbtt_summary.png"), dpi=150)
            plt.close(fig)
            produced.append("dbtt_summary.png")
        except Exception as exc:
            report_warn(f"DBTT plot failed: {exc}")

        # CSV
        try:
            lines = ["T_K,E_GPa,sigma_y_GPa,UTS_GPa,failure_strain,ductility_label,success"]
            for r in temp_results:
                def _s(v):
                    return "" if v is None else str(v)
                lines.append(
                    f"{r.temperature},{_s(r.youngs_modulus_GPa)},{_s(r.yield_strength_GPa)},"
                    f"{_s(r.uts_GPa)},{_s(r.failure_strain)},{r.ductility_label},{r.deform_success}"
                )
            (report_dir / "dbtt_summary.csv").write_text("\n".join(lines), encoding="utf-8")
            produced.append("dbtt_summary.csv")
        except Exception as exc:
            report_warn(f"DBTT CSV export failed: {exc}")

        return produced

    # ── Markdown ──────────────────────────────────────────────────────────

    def _render_markdown(self, payload, temp_results: list[_TempResult], fit_result: dict) -> str:
        ts = payload["report_generated_at"]
        ri = payload["run_info"]
        nx, ny, nz = self.config.supercell

        def _v(v, unit="", fmt=".3f"):
            return "未提取" if (v is None or v == "未提取") else f"{v:{fmt}} {unit}".strip()

        lines = [
            "# IamLAMMPS DBTT 温度扫描自动报告",
            "",
            f"> **生成时间：** {ts}  ",
            f"> **工作流：** dbtt (W BCC 多温度拉伸)  ",
            f"> **报告目录：** `{payload['report_dir']}`",
            "",
            "## A. 基本运行信息",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
            f"| 超胞 | {nx}×{ny}×{nz} |",
            f"| 原子数（估算） | {ri['atom_count_estimated']} |",
            f"| 扫描温度 | {', '.join(str(int(t)) for t in ri['temperatures_K'])} K |",
            f"| 应变速率 | {ri['strain_rate_per_s']:.2e} /s |",
            f"| 势函数 | {ri['potential']} |",
            f"| LAMMPS 命令 | `{ri['lammps_command']}` |",
            "",
            "## B. 各温度点力学结果汇总",
            "",
            "| T (K) | E (GPa) | σ_y (GPa) | UTS (GPa) | Failure strain | 韧脆倾向 | 状态 |",
            "|-------|---------|-----------|-----------|----------------|---------|------|",
        ]
        for r in temp_results:
            status = "✓" if r.deform_success else "✗"
            label_zh = {"brittle": "脆性", "transition": "过渡", "ductile": "韧性", "unknown": "未知"}
            lines.append(
                f"| {r.temperature:.0f} | {_v(r.youngs_modulus_GPa)} | "
                f"{_v(r.yield_strength_GPa)} | {_v(r.uts_GPa)} | "
                f"{_v(r.failure_strain, fmt='.4f')} | "
                f"{label_zh.get(r.ductility_label, r.ductility_label)} | {status} |"
            )

        T_dbtt = fit_result.get("T_dbtt_K")
        width = fit_result.get("transition_width_K")
        lines += [
            "",
            "## C. DBTT 拟合结果",
            "",
            "| 物理量 | 值 |",
            "|--------|-----|",
            f"| DBTT（Boltzmann sigmoid 拟合） | {_v(T_dbtt, 'K', '.1f')} |",
            f"| 过渡区宽度 | {_v(width, 'K', '.1f')} |",
            f"| 备注 | {fit_result.get('note', '')} |",
            "",
            "## D. 温度-响应曲线与数据导出",
            "",
            f"- **汇总图（UTS & Ductility vs T）：** `{payload['output_files']['summary_png']}`",
            f"- **汇总数据 CSV：** `{payload['output_files']['summary_csv']}`",
            "",
            "## E. 温度点运行状态",
            "",
            "| T (K) | deform run_dir | retry | 成功 |",
            "|-------|----------------|-------|------|",
        ]
        for r in temp_results:
            lines.append(f"| {r.temperature:.0f} | `{r.run_dir}` | {r.retry_count} | {'✓' if r.deform_success else '✗'} |")

        lines += [
            "",
            "---",
            "",
            "*本报告由 IamLAMMPS 自动生成。*",
            "",
        ]
        return "\n".join(lines)

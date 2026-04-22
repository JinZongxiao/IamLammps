"""Crack propagation workflow report generator.

Produces:
  crack_report.md / .json
  crack_curve.png
  crack_curve.csv
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
from ..skills.crack_analysis import CrackAnalysisSkill
from ..thermo_parser import parse_thermo_table
from .base_reporter import (
    _StabilityResult, _StageInfo,
    run_stability_checks, inventory_files,
    write_json, write_md, extract_stage_infos,
    report_info, report_warn,
)


@dataclass
class CrackReportConfig:
    supercell: tuple[int, int, int] = (30, 15, 5)
    temperature: float = 300.0
    potential_name: str = "W_Zhou2004_EAM"
    crack_length_fraction: float = 0.3
    lammps_command: str = "lmp"


class CrackReporter:
    def __init__(self, config: CrackReportConfig) -> None:
        self.config = config

    def generate(self, pipeline_result: PipelineResult) -> Optional[Path]:
        if not pipeline_result.success:
            report_warn("Pipeline did not succeed; crack report not generated.")
            return None

        # Use the 'load' stage for thermo/stress data
        load_output = next(
            (o for n, o in pipeline_result.stages if n == "load"), None
        )
        if load_output is None:
            report_warn("No 'load' stage found; crack report skipped.")
            return None

        run_dir = load_output.run_dir
        report_info(f"Generating crack report in {run_dir}")

        stage_infos = extract_stage_infos(pipeline_result)
        stability = run_stability_checks(run_dir, self.config.temperature)

        crack_result = self._extract_crack(run_dir)
        steps_arr, stress_arr, strain_arr = self._load_curve_data(run_dir)
        fracture_mode = self._classify_fracture_mode(stress_arr, crack_result)
        plot_files = self._generate_plots(run_dir, steps_arr, stress_arr, crack_result)
        artifacts = inventory_files(run_dir, plot_files)

        any_retry = any(s.retry_count > 0 for s in stage_infos)
        retry_detail = [f"{s.name} retry={s.retry_count}" for s in stage_infos if s.retry_count > 0]
        nx, ny, nz = self.config.supercell
        atom_count = nx * ny * nz * 2

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "report_generated_at": ts,
            "workflow": "crack",
            "run_dir": str(run_dir),
            "success": True,
            "run_info": {
                "supercell": list(self.config.supercell),
                "atom_count_estimated": atom_count,
                "temperature_K": self.config.temperature,
                "crack_length_fraction": self.config.crack_length_fraction,
                "potential": self.config.potential_name,
                "lammps_command": self.config.lammps_command,
                "any_retry": any_retry,
                "retry_detail": retry_detail,
            },
            "fracture": {**crack_result, "fracture_mode": fracture_mode},
            "output_files": {
                "crack_curve_png": "crack_curve.png" if "crack_curve.png" in artifacts else "未生成",
                "crack_curve_csv": "crack_curve.csv" if "crack_curve.csv" in artifacts else "未生成",
                "report_md": "crack_report.md",
                "report_json": "crack_report.json",
                "all_artifacts": artifacts,
            },
            "stability": {
                "convergence_summary": stability.convergence_summary,
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
            write_json(run_dir, "crack_report.json", payload)
        except Exception as exc:
            report_warn(f"Failed to write crack_report.json: {exc}")

        try:
            md = self._render_markdown(payload, crack_result, fracture_mode, stability, stage_infos, atom_count)
            md_path = write_md(run_dir, "crack_report.md", md)
        except Exception as exc:
            report_warn(f"Failed to write crack_report.md: {exc}")
            return None

        report_info(f"Crack report written: {md_path}")
        return md_path

    # ── Data extraction ───────────────────────────────────────────────────

    def _extract_crack(self, run_dir: Path) -> dict:
        try:
            skill = CrackAnalysisSkill()
            report = skill.analyze(run_dir)
            d = report.details
            return {
                "max_stress_GPa": d.get("max_stress_GPa"),
                "max_stress_step": d.get("max_stress_step"),
                "fracture_strain": d.get("fracture_strain"),
                "data_points": d.get("data_points", 0),
                "crack_extension_observed": "未提取",
                "crack_propagation_rate": "未提取",
                "energy_trend": "未提取",
                "note": report.summary,
            }
        except Exception as exc:
            return {"note": f"裂纹分析失败: {exc}"}

    def _load_curve_data(self, run_dir: Path):
        stdout = run_dir / "stdout.log"
        if not stdout.exists():
            return None, None, None
        try:
            thermo = parse_thermo_table(stdout)
            pyy = thermo.get("Pyy", [])
            steps = thermo.get("Step", [])
            ly = thermo.get("Ly", [])
            if len(pyy) < 5:
                return None, None, None
            stress = np.array([-p * 1e-4 for p in pyy])
            steps_arr = np.array(steps) if steps else np.arange(len(pyy))
            strain_arr = None
            if len(ly) >= 2:
                ly0 = ly[0]
                strain_arr = np.array([(l - ly0) / ly0 for l in ly])
            return steps_arr, stress, strain_arr
        except Exception:
            return None, None, None

    def _classify_fracture_mode(self, stress_arr, crack_result: dict) -> str:
        """Simple rule-based fracture mode classification."""
        if stress_arr is None or len(stress_arr) < 5:
            return "unknown"
        max_s = crack_result.get("max_stress_GPa")
        if max_s is None:
            return "unknown"
        # Find peak index
        peak_idx = int(np.argmax(stress_arr))
        post = stress_arr[peak_idx:]
        if len(post) < 3:
            return "unknown"
        drop = (stress_arr[peak_idx] - stress_arr[-1]) / (stress_arr[peak_idx] + 1e-12)
        slope_after = (stress_arr[-1] - stress_arr[peak_idx]) / max(len(post), 1)
        if drop > 0.5:
            return "brittle-like"
        elif drop < 0.1:
            return "crack blunting"
        else:
            return "crack deflection / partial"

    # ── Plots ─────────────────────────────────────────────────────────────

    def _generate_plots(self, run_dir: Path, steps_arr, stress_arr, crack_result: dict) -> list[str]:
        produced = []
        if stress_arr is None or len(stress_arr) < 5:
            report_warn("Insufficient crack data; skipping plot.")
            return produced
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(steps_arr, stress_arr, color="#d62728", lw=1.5, label="Pyy stress (Mode I)")
            max_s = crack_result.get("max_stress_GPa")
            max_step = crack_result.get("max_stress_step")
            if max_s and max_step:
                ax.axvline(max_step, color="orange", ls="--", lw=1,
                           label=f"Peak stress={max_s:.2f} GPa")
                ax.scatter([max_step], [max_s], color="orange", zorder=5, s=60)
            ax.set_xlabel("Timestep", fontsize=12)
            ax.set_ylabel("Stress Pyy (GPa)", fontsize=12)
            ax.set_title(
                f"W BCC Mode-I Crack  T={self.config.temperature:.0f} K  "
                f"a/W={self.config.crack_length_fraction:.1f}",
                fontsize=10,
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(run_dir / "crack_curve.png"), dpi=150)
            plt.close(fig)
            produced.append("crack_curve.png")
        except Exception as exc:
            report_warn(f"Crack curve plot failed: {exc}")

        try:
            lines = ["step,stress_GPa"]
            for st, s in zip(steps_arr, stress_arr):
                lines.append(f"{int(st)},{s:.6f}")
            (run_dir / "crack_curve.csv").write_text("\n".join(lines), encoding="utf-8")
            produced.append("crack_curve.csv")
        except Exception as exc:
            report_warn(f"Crack CSV export failed: {exc}")

        return produced

    # ── Markdown ──────────────────────────────────────────────────────────

    def _render_markdown(self, payload, crack_result, fracture_mode,
                         stability: _StabilityResult, stage_infos: list[_StageInfo],
                         atom_count: int) -> str:
        cfg = self.config
        ts = payload["report_generated_at"]
        nx, ny, nz = cfg.supercell

        def _v(v, unit="", fmt=".3f"):
            return "未提取" if (v is None or v == "未提取") else f"{v:{fmt}} {unit}".strip()

        mode_desc = {
            "brittle-like": "脆性断裂（峰值后应力急剧下降 >50%）",
            "crack blunting": "裂纹钝化（应力平缓，无明显断裂）",
            "crack deflection / partial": "部分裂纹扩展/偏转",
            "unknown": "未知（数据不足或未分类）",
        }

        lines = [
            "# IamLAMMPS 裂纹扩展仿真自动报告",
            "",
            f"> **生成时间：** {ts}  ",
            f"> **工作流：** crack (W BCC Mode-I)  ",
            f"> **run_dir：** `{payload['run_dir']}`",
            "",
            "## A. 基本运行信息",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
            f"| 超胞 | {nx}×{ny}×{nz} |",
            f"| 原子数（估算） | {atom_count} |",
            f"| 温度 | {cfg.temperature:.0f} K |",
            f"| 预裂纹长度比 a/W | {cfg.crack_length_fraction} |",
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
            "## B. 断裂力学结果",
            "",
            "| 物理量 | 值 |",
            "|--------|-----|",
            f"| 最大应力（Mode I Pyy） | {_v(crack_result.get('max_stress_GPa'), 'GPa')} |",
            f"| 峰值应力出现步数 | {crack_result.get('max_stress_step', '未提取')} |",
            f"| 断裂应变 | {_v(crack_result.get('fracture_strain'), fmt='.4f')} |",
            f"| 数据点数 | {crack_result.get('data_points', '未提取')} |",
            f"| 裂纹扩展观测 | {crack_result.get('crack_extension_observed', '未提取')} |",
            f"| 裂纹扩展速率 | {crack_result.get('crack_propagation_rate', '未提取')} |",
            f"| 能量演化趋势 | {crack_result.get('energy_trend', '未提取')} |",
            "",
            "### 断裂模式分类",
            "",
            f"**判定：** `{fracture_mode}`  ",
            f"**说明：** {mode_desc.get(fracture_mode, '未知')}",
            "",
            "> 分类依据：峰值后应力降幅（>50% → 脆性；<10% → 钝化；其余 → 偏转/部分）",
            "",
            "## C. 应力-步数曲线与数据导出",
            "",
            f"- **应力-步数曲线：** `{payload['output_files']['crack_curve_png']}`",
            f"- **原始数据 CSV：** `{payload['output_files']['crack_curve_csv']}`",
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

# CHANGELOG

本文件记录 IamLAMMPS 每次重要更新，最新条目在顶部。

---

## [v4.6] — 2026-04-22

### 新增：Tensile Workflow 自动科研报告生成 + OVITO 结构演化分析

**新增文件**
- `src/lammps_agent/report/__init__.py` — report 包入口
- `src/lammps_agent/report/tensile_reporter.py` — 核心报告生成器
- `src/lammps_agent/report/ovito_analyzer.py` — OVITO PTM 分析模块（含优雅降级）
- `src/lammps_agent/reports/` — 早期实验性报告模块（存档）

**修改文件**
- `src/lammps_agent/workflows/tensile.py` — 覆盖 `run()`，pipeline 成功后自动触发报告生成；失败仅 warning，不影响 workflow 结果
- `src/lammps_agent/main.py` — 向 tensile workflow 注入 `TensileReportConfig`（CLI 参数完整透传）；payload 输出 `tensile_report` 路径

**报告产物（自动写入 deform run_dir）**

| 文件 | 内容 |
|------|------|
| `tensile_report.md` | 中文科研报告（A～F 六节） |
| `tensile_report.json` | 结构化 JSON，字段稳定，便于程序读取 |
| `stress_strain.png` | 应力-应变曲线，标注 yield（橙）/ UTS（红） |
| `stress_strain.csv` | 原始数据导出（strain, stress_GPa） |
| `ovito_*.png` | OVITO 关键帧图（需安装 ovito 包，否则自动跳过） |

**报告章节**
- A. 基本运行信息（超胞/原子数/T/dt/势函数/retry 详情/各阶段状态）
- B. 拉伸力学结果（E / σ_y / UTS / 最大应变 / 峰值后软化判断）
- C. 曲线与数据导出（文件路径）
- D. 运行稳定性与物理性检查（convergence / thermo_validator / NaN / 可信度评估）
- E. 文件产物清单
- F. 原子结构演化分析（OVITO PTM，不可用时输出 `OVITO analysis unavailable` 及安装指引）

**OVITO 降级机制**
`ovito_analyzer.py` 在 `import ovito` 失败时立即返回 `available=False`，整个报告链路不受影响。
未来安装 OVITO 后无需修改代码，自动启用 PTM 结构分类 + 4 关键帧渲染。

**运行命令**
```bash
conda run -n iam-lammps python -m src.lammps_agent.main \
  --workflow tensile \
  --supercell 10,10,20 \
  --temperature 300 \
  --equil-steps 20000 \
  --timestep 0.001 \
  --thermo-every 1000 \
  --dump-every 1000 \
  --runs-dir /ssd1/jinzongxiao/iam_runs/runs \
  --run
```

---

## [v4.5] — 2026-04-20

### 初始提交：IamLAMMPS v4.5

- 核心 Agent（生成 → 执行 → 错误重试 → 记忆）
- Pipeline 多阶段流水线（minimize → equilibrate → deform）
- Skills 框架（stress_strain / convergence / thermo_validator / thermo_tail 等 9 种）
- Workflows（Tensile / Nanoindent / Crack / DBTT / Interface / HeTrapping）
- Backends 抽象层（LocalBackend / SlurmBackend 占位）
- DFT / MLIP / Structures 领域扩展占位
- 72 个单元测试 + 5 个端到端测试，全部通过

---

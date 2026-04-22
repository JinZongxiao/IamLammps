# CHANGELOG

本文件记录 IamLAMMPS 每次重要更新，最新条目在顶部。

---

## [v4.7.1] — 2026-04-22

### 修复：Nanoindent equilibrate 阶段反复 retry 失败 + `[RETRY N/M]` 显示越界

**问题一：equilibrate 阶段 100% 重试、始终失败**

`_equilibrate_task()` 只给出自然语言描述（"Re-create the fixed group (bottom 15% of z)"），LLM 在翻译时自行推断，生成了以下非法脚本模式：

```lammps
compute zcoord all property/atom z
compute zmin all reduce min c_zcoord   # ← 'reduce' 是合法 style，但放在 run 0 之后无效
run 0
variable zmin_val equal c_zmin
variable zcut equal ${zmin_val}+0.15*(${zmax_val}-${zmin_val})  # ← ${} 展开在 equal-style 里非法
region fixed_region block INF INF INF INF INF ${zcut}
```

根因：LAMMPS equal-style variable 不能用 `${var}` 引用另一个 equal-style variable；compute 的标量结果必须通过 `c_name` 引用，而且 `run 0` 必须在 compute 定义**之后**、variable 引用**之前**执行。LLM 每次 retry 都复现同一个错误，无法自愈。

**解决方案：** 将 `_equilibrate_task()` 改为在 Python 层直接计算 `z_fixed = 0.15 * nz * 3.165`，输出含硬编码数值的完整 LAMMPS 命令，不再让 LLM 推断 compute/variable 逻辑：

```lammps
region fixed_region  block INF INF INF INF INF 4.7475   # Python 预算好的数值
region thermo_region block INF INF INF INF 4.7475 9.4950
group fixed_atoms  region fixed_region
...
```

---

**问题二：`[RETRY 4/3]` 显示越界**

`for attempt in range(max_retries + 1)` 中，最后一次循环（`attempt == max_retries == 3`）失败后仍调用 `_log_retry(attempt + 1, ...)` 打印 `[RETRY 4/3]`。此时已无下一次重试，该日志没有意义且数字越界。

**解决方案：** 在 `_log_retry` / `_log_retry_unknown` 调用处增加 `if attempt < max_retries` 判断，最后一次失败只记录错误上下文，不再打印 retry 日志。

---

**修改文件**
- `src/lammps_agent/workflows/nanoindent.py` — `_equilibrate_task()` 改为硬编码 z 边界值
- `src/lammps_agent/agent.py` — retry 日志只在 `attempt < max_retries` 时打印

**验证**：smoke test（5×5×10，equil 500步）三阶段全部通过，equilibrate 阶段 retry=0。

---

## [v4.7] — 2026-04-22

### 新增：Nanoindent / Crack / DBTT Workflow 自动报告 + 统一 Dump 单文件策略

**新增文件**
- `src/lammps_agent/report/base_reporter.py` — 跨 Workflow 公共报告基础设施（稳定性检查、文件清单、JSON/MD 写入、阶段信息提取）
- `src/lammps_agent/report/nanoindent_reporter.py` — 纳米压痕报告生成器（Oliver-Pharr 结果 + P-h 曲线）
- `src/lammps_agent/report/crack_reporter.py` — 裂纹扩展报告生成器（脆/韧断裂模式分类 + 应力-步骤曲线）
- `src/lammps_agent/report/dbtt_reporter.py` — DBTT 温度扫描汇总报告（双面板图 + sigmoid 拟合）

**修改文件**
- `src/lammps_agent/report/__init__.py` — 导出全部四组 Reporter/Config
- `src/lammps_agent/workflows/nanoindent.py` — 覆盖 `run()`；新增 `dump_every` / `dump_split=False` 参数
- `src/lammps_agent/workflows/crack.py` — 覆盖 `run()`；新增 `dump_every` / `dump_split=False` 参数
- `src/lammps_agent/workflows/dbtt_sweep.py` — 覆盖 `run()`，全温度完成后生成聚合 DBTT 报告
- `src/lammps_agent/workflows/__init__.py` — `build_workflow()` 向 nanoindent/crack 透传 `dump_every`
- `src/lammps_agent/main.py` — 统一注入四个 workflow 的 `_report_config`；`payload["report"]` 支持所有 workflow

**各 Workflow 报告产物**

| Workflow | 报告文件 | 额外产物 |
|----------|----------|----------|
| nanoindent | `nanoindent_report.md/json` | `indent_ph_curve.png/csv` |
| crack | `crack_report.md/json` | `crack_curve.png/csv` |
| dbtt | `dbtt_report.md/json` | `dbtt_summary.png/csv`（写入首个阶段上级目录） |

**Dump 单文件策略**
- 所有 workflow 默认 `dump_split=False`：轨迹输出到单一文件（如 `dump.crack.lammpstrj`）而非 `dump.crack.*.lammpstrj`（每帧单独文件）
- 通过 `--dump-every N` CLI 参数控制频率；不指定则不输出 dump（nanoindent/crack）或固定 1000 步

**公共基础设施复用**
`base_reporter.py` 提供所有 Reporter 共享的逻辑：`run_stability_checks()`、`inventory_files()`、`write_json()`、`write_md()`、`extract_stage_infos()`

**运行命令（生产）**
```bash
# Nanoindent
conda run -n iam-lammps python -m src.lammps_agent.main \
  --workflow nanoindent --temperature 300 --supercell 20,20,30 \
  --dump-every 500 --runs-dir /ssd1/jinzongxiao/iam_runs/runs --run

# Crack
conda run -n iam-lammps python -m src.lammps_agent.main \
  --workflow crack --temperature 300 --supercell 30,15,5 \
  --dump-every 500 --runs-dir /ssd1/jinzongxiao/iam_runs/runs --run

# DBTT
conda run -n iam-lammps python -m src.lammps_agent.main \
  --workflow dbtt --temperature 300 --supercell 10,10,20 \
  --equil-steps 20000 --thermo-every 1000 --dump-every 1000 \
  --runs-dir /ssd1/jinzongxiao/iam_runs/runs --run
```

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

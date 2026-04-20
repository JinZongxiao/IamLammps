# IamLAMMPS 演进规划：ODS-W 合金力学性能模拟智能体

## Context

项目从一个最小可用的 LAMMPS Agent MVP 起步，经过 4 个 Phase 的迭代，已演进为具备完整力学性能测试能力的自动化仿真智能体。Phase 1~4 已全部完成并通过端到端验证。

用户目标：最终面向 **ODS-W（氧化物弥散强化钨）合金力学性能模拟** 的自进化智能体。

---

## 总体架构演进

```
Phase 1 (完成)          Phase 2 (完成)          Phase 3 (完成)          Phase 4 (完成+验证)       Phase 4.5 (完成)
单次生成 → 执行    →   自纠错闭环 + 奖励    →  势函数 + 多阶段流水线  →  力学性能专项         →   平台扩展
                       ↓                       ↓                      ↓                        ↓
                    error_parser           pipeline.py             workflows/                backends/
                    reward.py              potentials/             tensile, indent, crack     core_types.py
                    retry loop             model_builder.py        DBTT sweep                 structures/dft/mlip/
```

**Phase 4.5（完成）**：平台扩展 — backends/ + core_types + 新 skills/workflows + structures/dft/mlip 领域包

**Phase 5（远期）**：RL 自进化 — 经验回放 + Prompt 优化 + 可选 DPO/RLHF 微调

---

## Phase 2：自纠错闭环与奖励系统 ✅ 已完成

> 目标：Agent 能自动诊断 LAMMPS 错误、修正脚本、重试，并积累经验

### 2.1 LAMMPS 错误解析器 ✅

- `src/lammps_agent/error_parser.py`
- `LammpsError` dataclass：error_type、raw_message、source_file、suggested_fix
- 覆盖 LAMMPS 常见的 ~20 种错误模式（正则匹配）
- 每种错误附带结构化修复建议，注入 LLM 的 retry prompt

### 2.2 Agent 重试循环 ✅

- `src/lammps_agent/agent.py` 中 `run()` 方法
- 每次重试将错误信息 + 修复建议注入 LLM 上下文
- 每次重试的脚本保存为 `in.lammps.retryN` 便于回溯
- CLI `--max-retries` 参数控制（默认 3）

### 2.3 奖励评分系统 ✅

- `src/lammps_agent/reward.py`
- `RewardSignal` dataclass：script_validity(0/1)、convergence(0~1)、physical_plausibility(0~1)、efficiency(0~1)、total
- 权重：validity×0.4 + convergence×0.25 + plausibility×0.25 + efficiency×0.1
- 收敛性通过尾部方差/全局方差评估

### 2.4 增强 Memory 系统 ✅

- `src/lammps_agent/memory.py`
- `SimulationMemoryEntry` 新增字段：reward_score、error_type、retry_count、script_hash
- 压缩上下文按 reward 排序，高分成功经验优先展示
- 向后兼容旧 JSONL 格式（过滤未知 key + 填充默认值）

### 2.5 Skills 包重构 + 物理验证 ✅

- `skills/` 目录替换旧 `skills.py`
- `base.py`：BasePostprocessSkill ABC + SkillReport dataclass
- `thermo_tail.py`：从旧文件迁移
- `convergence.py`：能量/温度收敛性检查
- `thermo_validator.py`：物理量合理范围检查（T>0、能量非NaN、压力不发散）
- `__init__.py`：统一 registry + 向后兼容导出

### 2.6 共享工具 ✅

- `src/lammps_agent/thermo_parser.py`：解析 LAMMPS thermo 输出为 `{列名: [值]}` 字典
- reward.py、convergence skill、thermo_validator skill 共用此解析器

---

## Phase 3：势函数管理 + 多阶段流水线 ✅ 已完成

> 目标：支持多组分体系的势函数管理，实现多步模拟流水线

### 3.1 势函数管理 ✅

- `src/lammps_agent/potentials/` 包
- `registry.py`：PotentialEntry dataclass + PotentialRegistry（自动发现 LAMMPS 内置势）
- `manager.py`：PotentialManager — resolve、copy 势文件到 run_dir、生成 pair_style/pair_coeff 命令
- 势路径发现策略：`LAMMPS_POTENTIALS_DIR` 环境变量 → `which lmp_mpi` 相对路径 → Homebrew 默认路径
- 内置注册 `W_Zhou2004_EAM`（Zhou 2004 W EAM，随 LAMMPS 自带）
- Agent 的 `run()` 在创建 run_dir 后自动 copy 所有已注册势文件

### 3.2 多阶段流水线 ✅

- `src/lammps_agent/pipeline.py`
- `PipelineStage`：name、task_description、depends_on、input_files、timeout、potential_name
- `PipelineResult`：stages 列表、success、failed_stage
- Pipeline 先创建 run_dir、copy 上一阶段的 restart/data 文件，再调用 Agent 执行
- 同一秒内多阶段自动添加后缀避免 run_dir 冲突

### 3.3 LAMMPS Runner 增强 ✅

- `src/lammps_agent/lammps_runner.py`
- `LammpsRunResult` 新增字段：wall_time、timed_out、output_files
- `run_lammps()` 新参数：timeout、mpi_np
- MPI 支持：mpi_np > 1 时自动 prepend `mpirun -np N`
- 超时处理：subprocess.TimeoutExpired 捕获
- 执行前后对比目录，检测新产生的 output_files

### 3.4 SYSTEM_PROMPT 增强 ✅

- 添加 velocity 命令要求（修复动力学模拟 T≈0 的 bug）
- 添加 W 材料专用指南：`units metal`、`lattice bcc 3.165`、`pair_style eam/alloy`、`mass 1 183.84`、`timestep 0.001`
- 添加 read_restart 指导（restart 后仍需 pair_style/pair_coeff/mass，但不需 lattice/create_atoms）

### 3.5 Config 增强 ✅

- `AgentConfig` 新增：lammps_timeout、mpi_np、potentials_dir
- 默认 ENABLED_SKILLS 扩展为 `thermo_tail,convergence,thermo_validator`
- 新环境变量：LAMMPS_TIMEOUT、MPI_NP、LAMMPS_POTENTIALS_DIR

---

## Phase 4：纯 W 力学性能模拟专项 ✅ 已完成 + 端到端验证

> 目标：针对纯 W 实现标准力学性能测试工作流

### 4.1 模型构建工具 ✅

- `src/lammps_agent/model_builder.py`
- `TungstenModelBuilder`：生成 LAMMPS 命令字符串
  - `bcc_supercell_commands(nx, ny, nz)`：BCC W 超胞
  - `slab_for_indentation()`：3 层板模型（固定层 + 恒温层 + 自由层）
  - `precracked_specimen()`：预裂纹试样（region + delete_atoms）

### 4.2 模拟工作流 ✅

- `src/lammps_agent/workflows/` 包

#### 4.2.1 单轴拉伸（tensile.py）✅ 端到端验证通过
- 3 阶段流水线：minimize → NPT equilibrate → fix deform 拉伸
- 自动计算 LAMMPS erate（从物理应变率换算）
- 默认：300K、应变率 10^8 /s、10x10x20 超胞
- **e2e 结果**：3 阶段 1.9s，stress_strain skill 输出 E=874 GPa, UTS=1.74 GPa

#### 4.2.2 纳米压痕（nanoindent.py）✅ 端到端验证通过
- 3 阶段流水线：minimize → equilibrate → fix indent 球形压头
- 支持配置压头半径和速度
- **e2e 结果**：3 阶段 7.4s，indent_analysis skill 输出 P_max=82.93 nN

#### 4.2.3 裂纹扩展（crack.py）✅ 端到端验证通过
- 4 阶段流水线：build_crack → minimize → Mode I load → analyze
- region + delete_atoms 构建预裂纹
- **e2e 结果**：4 阶段 2.2s，crack_analysis skill 输出 σ_max(Mode I)=-0.08 GPa

#### 4.2.4 DBTT 温度扫描（dbtt_sweep.py）✅ 端到端验证通过
- 多温度点循环调用拉伸工作流（默认 200K~1200K 间隔 100K）
- 汇总各温度延伸率 + Boltzmann sigmoid 拟合 DBTT
- **e2e 结果**：2 温度点 smoke test 3.7s，300K/600K 均通过

### 4.3 力学性能分析 Skills ✅

| Skill | 文件 | 说明 | e2e 验证 |
|-------|------|------|----------|
| `stress_strain` | `skills/stress_strain.py` | 弹性模量（线性拟合 ε<0.02）、屈服强度（0.2% offset）、UTS | ✅ |
| `indent_analysis` | `skills/indent_analysis.py` | Oliver-Pharr 方法（硬度、弹性模量）| ✅ |
| `crack_analysis` | `skills/crack_analysis.py` | Mode I 最大应力、断裂应变 | ✅ |
| `dbtt_fit` | `skills/dbtt_fit.py` | Boltzmann sigmoid 拟合 DBTT（需 scipy curve_fit）| ✅ |

### 4.4 CLI 扩展 ✅

- `--workflow tensile|nanoindent|crack|dbtt`
- `--potential`、`--temperature`、`--supercell`、`--strain-rate`
- `--mpi-np`、`--timeout`

---

## Bug 修复记录

### 势文件未 copy 到 run_dir（2026-04-17 修复）

- **现象**：LAMMPS 报 `cannot open eam/alloy potential file W_zhou.eam.alloy: No such file or directory`
- **根因**：`agent.py` 的 `run()` 创建 run_dir 后不会 copy 势文件，LAMMPS 通过相对路径找不到
- **修复**：`agent.py` 新增 `PotentialManager` 集成和 `_prepare_potentials()` 方法，`run()` 创建目录后自动 copy 所有已注册势文件

### Pipeline 阶段间文件未传递（2026-04-17 修复）

- **现象**：Pipeline stage 2 报 `restart.minimize` 不存在
- **根因**：`pipeline.py` 存储了 `_pending_copy` 引用但从未实际执行文件 copy
- **修复**：重写 `pipeline.run()` — 先创建 run_dir、copy 继承文件，再传给 agent 执行

### 同秒 run_dir 冲突（2026-04-17 修复）

- **现象**：Pipeline 多阶段在同一秒内创建 run_dir，timestamp 冲突导致覆盖
- **根因**：`_create_run_dir()` 使用 `exist_ok=True`，同秒内两阶段指向同一目录
- **修复**：timestamp 冲突时自动添加 `_1`, `_2`, ... 后缀

---

## Phase 4.5：平台扩展（增量重构） ✅ 已完成

> 目标：在不修改任何现有代码行为的前提下，添加 Backend 抽象、新 Skills/Workflows 骨架、DFT/MLIP 钩子，为 ODS-W 研究平台奠基。

### 设计原则

- **增量添加**：22 个新文件，仅修改 4 个现有文件（共 ~16 行纯增量改动）
- **依赖方向**：新代码导入旧代码，旧代码不导入任何新模块
- **零回归**：所有 72 个单元测试 + 5 个 e2e 测试通过

### 4.5.1 统一任务/结果模式 ✅

- `src/lammps_agent/core_types.py`
- `SimulationTask`：code、backend、resources、metadata、work_dir
- `SimulationResult`：success、outputs、timing、errors、raw_result
- raw_result 字段保留原始 LammpsRunResult 供现有代码访问

### 4.5.2 Backend 抽象 ✅

- `src/lammps_agent/backends/` 包
- `base.py`：JobBackend ABC（submit/poll/collect/run_sync）
- `local.py`：LocalBackend 包装现有 `run_lammps()`，同步执行
- `slurm.py`：SlurmBackend 占位（NotImplementedError）
- `runner_adapter.py`：`get_backend(config)` + `run_via_backend()` 高级接口
- `config.py` 新增 `backend: str = "local"` 字段（默认值，无破坏性）

### 4.5.3 新分析 Skills ✅

| Skill | 文件 | 说明 |
|-------|------|------|
| `interface_analysis` | `skills/interface_analysis.py` | 界面粘附功、分离功、峰值应力 |
| `defect_analysis` | `skills/defect_analysis.py` | LAMMPS dump 配位数分析，空位/间隙/He 泡检测 |

注册到 `skills/__init__.py` 的 `_SKILL_REGISTRY`，默认不启用。

### 4.5.4 新工作流骨架 ✅

| 工作流 | 文件 | 阶段 |
|--------|------|------|
| InterfaceMechanicsWorkflow | `workflows/interface_mechanics.py` | build_interface → minimize → equilibrate → tensile_normal |
| HeTrappingWorkflow | `workflows/he_trapping.py` | create_defects → minimize → insert_he → anneal → analyze |

注册到 `workflows/__init__.py` 的 `build_workflow()` 工厂。CLI `--workflow interface|he_trapping` 可用。

### 4.5.5 领域扩展包 ✅

| 包 | 文件 | 说明 |
|----|------|------|
| `structures/` | `interface_builder.py` | InterfaceBuilder（Y2O3/Al2O3 占位，compute_interface_area 可用） |
| `dft/` | `vasp_workflow.py` | VASPWorkflow + DFTResult（全占位） |
| `mlip/` | `trainer.py` | MLIPTrainer（DeePMD/MACE/ACE）+ TrainingConfig/ValidationMetrics |

### 4.5.6 现有文件改动汇总

| 文件 | 改动 | 新增行数 |
|------|------|---------|
| `config.py` | 加 `backend` 字段 + 读取 | 2 |
| `skills/__init__.py` | 加 2 个 skill 的 import/注册 | 6 |
| `workflows/__init__.py` | 加 2 个 workflow 的 import/注册 | 8 |
| `main.py` | 扩展 --workflow choices | 1 |

---

## Phase 5（远期）：RL 自进化 — 未开始

> 目标：基于积累的 (task, script, reward) 数据，实现 Agent 的持续优化

### 5.1 经验回放 + Prompt 优化
- 从 memory 中提取高 reward 样本，动态构建 few-shot examples
- 按 error_type 统计常见错误，更新 SYSTEM_PROMPT 的规则

### 5.2 DPO/RLHF 微调（可选）
- 收集 (prompt, good_script, bad_script) 三元组
- 对开源模型做 preference tuning
- 条件：500+ 有效样本对 + GPU 资源

---

## 测试状态

### 单元测试 ✅（72/72 通过）

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_thermo_parser.py` | thermo 输出解析（正常/异常/文件读取） |
| `test_reward.py` | 奖励信号计算（成功/失败/重试/零温） |
| `test_memory.py` | 新字段、序列化、向后兼容、reward 排序 |
| `test_skills_package.py` | Skills 包导入、6 个 skill 分析逻辑 |
| `test_core_types.py` | SimulationTask / SimulationResult 构造与字段 |
| `test_backends.py` | LocalBackend / SlurmBackend / run_sync |
| `test_runner_adapter.py` | get_backend / run_via_backend |
| `test_new_skills.py` | interface_analysis / defect_analysis |
| `test_new_workflows.py` | InterfaceMechanics / HeTrapping + 工厂回归 |
| `test_domain_extensions.py` | structures / dft / mlip 导入+实例化 |

### 端到端测试 ✅ 全部通过（真实 LAMMPS 执行）

| 测试 | 文件 | LAMMPS 阶段 | 耗时 | 关键输出 |
|------|------|-------------|------|----------|
| W BCC 最小化 | `test_e2e_w_minimize.py` | 1 | 0.2s | rc=0, reward=0.875, 势文件 copy 确认 |
| Tensile | `test_e2e_mechanics.py` | 3 | 1.9s | E=874 GPa, UTS=1.74 GPa |
| Nanoindent | `test_e2e_mechanics.py` | 3 | 7.4s | P_max=82.93 nN |
| Crack | `test_e2e_mechanics.py` | 4 | 2.2s | σ_max(Mode I)=-0.08 GPa |
| DBTT (2 temps) | `test_e2e_mechanics.py` | 6 | 3.7s | 300K: UTS=1.74, 600K: UTS=2.82 |

**测试环境**：conda `auto-lammps`, Python 3.11, LAMMPS 22 Jul 2025 Update 3, macOS ARM64

> 端到端测试 bypass LLM（用已知正确的 LAMMPS 脚本），验证完整执行链：势文件 copy → restart 文件传递 → LAMMPS 执行 → skill 分析 → reward 计算 → memory 写入。数值仅验证 pipeline 逻辑，不代表文献精度（超胞小+平衡短+应变率高）。

### 待完成

- [ ] LLM 端到端验证：实际调用 LLM 生成 W 脚本（需配置 API key）
- [ ] 生产级数值验证：大超胞 W 弹性模量 ≈ 380-420 GPa（对比文献值）
- [ ] DBTT 全温度范围扫描（11 个温度点）
- [ ] SlurmBackend 实现（sbatch 提交 + squeue 轮询）
- [ ] InterfaceBuilder 实体实现（需晶体结构数据）
- [ ] MLIPTrainer DeePMD-kit 集成
- [ ] VASPWorkflow VASP 输入/输出集成

---

## ODS-W 势函数策略

| 体系 | 现有势 | 推荐路线 | 状态 |
|------|--------|----------|------|
| 纯 W | EAM (Zhou 2004), GAP, ACE | 直接使用 | ✅ 已集成 + 验证 |
| W-He | EAM/FS + ZBL, GAP | 辐照研究可用 | 待集成 |
| W-O | 缺乏成熟势 | 需训练 DeePMD/MACE | 未开始 |
| W-Y-O | 几乎空白 | 核心研究任务 | 未开始 |
| W-Y-O-Ti（完整 ODS-W） | 不存在 | 长期目标 | 未开始 |

**现实路线**：
- **短期**：纯 W EAM 势跑通全套力学性能流水线 ✅ **已完成**
- **中期**：DeePMD-kit 或 MACE 训练 W-O 二元势（需 DFT 数据）
- **长期**：扩展到 W-Y-O 三元势（系统 DFT 计算 + 主动学习）

---

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-04 | 先用纯 W EAM 跑通全套流水线 | 验证 pipeline 逻辑，积累经验 |
| 2026-04 | 使用 conda `auto-lammps` 环境 | 已集成 LAMMPS + Python 依赖 |
| 2026-04 | W EAM 势无需额外下载 | LAMMPS 安装自带 `W_zhou.eam.alloy` |
| 2026-04 | ML 势框架待定 | DeePMD / MACE / ACE 均可行，后续选型 |
| 待定 | 计算资源 | 有 GPU 集群可用于 ML 势训练 |

---

## 目录结构

```
src/lammps_agent/
    __init__.py          agent.py             config.py (改)
    core_types.py (新)   error_parser.py      lammps_runner.py
    llm_clients.py       main.py (改)         memory.py
    model_builder.py     pipeline.py          reward.py
    runner_adapter.py (新)  thermo_parser.py
    backends/ (新)
        __init__.py      base.py              local.py
        slurm.py
    dft/ (新)
        __init__.py      vasp_workflow.py
    mlip/ (新)
        __init__.py      trainer.py
    potentials/
        __init__.py      registry.py          manager.py
    skills/ (改)
        __init__.py      base.py              thermo_tail.py
        convergence.py   thermo_validator.py  stress_strain.py
        indent_analysis.py  crack_analysis.py dbtt_fit.py
        interface_analysis.py (新)  defect_analysis.py (新)
    structures/ (新)
        __init__.py      interface_builder.py
    workflows/ (改)
        __init__.py      base.py              tensile.py
        nanoindent.py    crack.py             dbtt_sweep.py
        interface_mechanics.py (新)  he_trapping.py (新)
tests/
    conftest.py              test_thermo_parser.py
    test_reward.py           test_memory.py
    test_skills_package.py   test_e2e_w_minimize.py
    test_e2e_mechanics.py
    test_core_types.py (新)      test_backends.py (新)
    test_runner_adapter.py (新)  test_new_skills.py (新)
    test_new_workflows.py (新)   test_domain_extensions.py (新)
```

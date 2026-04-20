# IamLAMMPS 项目上下文（Agent Handoff Document）

> 本文件供 AI Agent 在新会话中快速理解项目全貌。最后更新：2026-04-17。

---

## 一、项目定位

IamLAMMPS 是一个 **LLM 驱动的 LAMMPS 分子动力学仿真自动化 Agent**。核心流程：

```
用户自然语言任务 → LLM 生成 LAMMPS 脚本 → 执行 → 错误解析/自动重试 → 后处理分析 → 记忆积累
```

最终目标：面向 ODS-W（氧化物弥散强化钨）合金力学性能模拟的自进化智能体。
当前阶段：**纯 W 力学性能全套工作流已跑通并通过端到端验证；平台扩展层（Backend 抽象、interface/He 工作流骨架、DFT/MLIP 钩子）已就位**。

---

## 二、技术栈与环境

| 项 | 值 |
|----|-----|
| Python | 3.11（conda `auto-lammps` 环境） |
| LAMMPS | 22 Jul 2025 Update 3（Homebrew，`lmp_mpi`） |
| LLM | ERNIE（百度文心）通过 OpenAI 兼容接口，`LLM_PROVIDER=openai_compatible` |
| 势文件 | `W_zhou.eam.alloy`（Zhou 2004 EAM），LAMMPS 自带，自动发现+copy |
| 平台 | macOS ARM64 (Darwin 24.6.0)，计划部署至 Linux HPC 服务器 |
| 项目路径 | `/Users/jinzongxiao/Desktop/codespace/IamLAMMPS` |

**环境激活**：
```bash
conda activate auto-lammps
cd /Users/jinzongxiao/Desktop/codespace/IamLAMMPS
```

**注意**：项目根目录有一个 `.venv/`，解释器路径已失效（旧路径 `/Users/jinzongxiao/Desktop/IamLAMMPS/`），**不要使用**，始终用 conda `auto-lammps`。

### 服务器部署要点

项目设计上已考虑 Linux HPC 部署：
- **LAMMPS**：服务器上通过 module 或 conda 安装，设 `LAMMPS_COMMAND` 指向可执行文件（如 `lmp_mpi` 或 `lmp`）
- **MPI**：`MPI_NP=N` 或 `--mpi-np N` 控制并行数
- **势文件路径**：`LAMMPS_POTENTIALS_DIR` 环境变量指向服务器上的势文件目录
- **LLM 后端**：内网可用 Ollama（`LLM_PROVIDER=local_ollama`）或 vLLM/TGI（`openai_compatible`）
- **Backend 抽象**：`BACKEND=slurm` 切换到 Slurm 提交模式（占位，待实现 sbatch/squeue 集成）
- **无 GPU 依赖**：当前代码纯 CPU，仅 MLIP 训练需要 GPU

---

## 三、架构总览

```
src/lammps_agent/
│
├── agent.py            # 核心 Agent：LLM 调用 + 重试循环 + 势文件部署 + reward/memory 集成
├── config.py           # AgentConfig dataclass，从 .env 加载（含 backend 字段）
├── main.py             # CLI 入口：--task 单任务 / --workflow 工作流模式
├── llm_clients.py      # OpenAI 兼容 + Ollama 两种 LLM 客户端
├── lammps_runner.py    # subprocess 调用 lmp_mpi，支持 MPI/timeout/output_files 检测
├── error_parser.py     # LAMMPS ~20 种错误模式正则匹配 + 结构化修复建议
├── reward.py           # 多维奖励信号：validity×0.4 + convergence×0.25 + plausibility×0.25 + efficiency×0.1
├── memory.py           # JSONL 记忆系统，按 reward 排序注入上下文
├── thermo_parser.py    # 解析 LAMMPS thermo 输出为 {列名: [值]} 字典
├── pipeline.py         # 多阶段流水线：run_dir 创建 → 继承文件 copy → agent 执行
├── model_builder.py    # TungstenModelBuilder：BCC 超胞 / 压痕板 / 预裂纹试样
├── core_types.py       # [新] SimulationTask / SimulationResult 统一模式
├── runner_adapter.py   # [新] get_backend() / run_via_backend() 适配器
│
├── potentials/
│   ├── registry.py     # PotentialEntry + PotentialRegistry（自动发现 LAMMPS 安装路径）
│   └── manager.py      # PotentialManager：resolve / copy 到 run_dir / 生成 pair 命令
│
├── backends/           # [新] 执行后端抽象
│   ├── base.py         # JobBackend ABC：submit() / poll() / collect() / run_sync()
│   ├── local.py        # LocalBackend：包装现有 run_lammps()，同步执行
│   └── slurm.py        # SlurmBackend：占位（sbatch/squeue/sacct）
│
├── skills/
│   ├── base.py         # BasePostprocessSkill ABC + SkillReport dataclass
│   ├── thermo_tail.py  # thermo 尾部提取
│   ├── convergence.py  # 尾部方差/全局方差 → 收敛性
│   ├── thermo_validator.py  # T>0, 能量非NaN, 压力不发散
│   ├── stress_strain.py     # E(线性拟合 ε<0.02), σ_y(0.2% offset), UTS
│   ├── indent_analysis.py   # Oliver-Pharr: P_max, H, E_r
│   ├── crack_analysis.py    # Mode I: σ_max, fracture_strain
│   ├── dbtt_fit.py          # Boltzmann sigmoid → DBTT (需 scipy)
│   ├── interface_analysis.py # [新] 界面粘附功、分离功、峰值应力
│   └── defect_analysis.py    # [新] 空位/间隙/He 泡检测（配位数分析）
│
├── workflows/
│   ├── base.py         # WorkflowConfig + BaseWorkflow ABC
│   ├── tensile.py      # 3 阶段：minimize → NPT equilibrate → fix deform
│   ├── nanoindent.py   # 3 阶段：minimize → equilibrate → fix indent
│   ├── crack.py        # 4 阶段：build_crack → minimize → Mode I load → analyze
│   ├── dbtt_sweep.py   # 多温度点循环调用 TensileWorkflow
│   ├── interface_mechanics.py  # [新] oxide/W 界面力学 (4 阶段)
│   └── he_trapping.py         # [新] He 陷阱/气泡形成 (5 阶段)
│
├── structures/         # [新] 多组分结构构建
│   └── interface_builder.py  # InterfaceBuilder：oxide/W 界面模型（占位）
│
├── dft/                # [新] DFT 工作流钩子
│   └── vasp_workflow.py      # VASPWorkflow：VASP DFT 计算占位
│
└── mlip/               # [新] ML 势训练钩子
    └── trainer.py            # MLIPTrainer：DeePMD/MACE/ACE 训练占位
```

---

## 四、关键实现细节（Agent 必知）

### 4.1 势文件部署

`agent.py` 的 `run()` 在创建 run_dir 后立即调用 `_prepare_potentials()`，将所有已注册势文件 copy 到 run_dir。LAMMPS 脚本用相对路径 `pair_coeff * * W_zhou.eam.alloy W` 即可。

势文件位置自动发现链：
1. `LAMMPS_POTENTIALS_DIR` 环境变量
2. `$(which lmp_mpi)/../share/lammps/potentials/`
3. `/opt/homebrew/share/lammps/potentials/`（Homebrew macOS）

实际位置：`/opt/homebrew/Cellar/lammps/20250722-update3/share/lammps/potentials/W_zhou.eam.alloy`

### 4.2 Pipeline 阶段间文件传递

`pipeline.py` 的 `run()` 流程：
1. 调用 `agent._create_run_dir()` 创建新目录
2. 调用 `_copy_inherited_files(prev_run_dir, new_run_dir, patterns)` 复制上阶段的 restart 文件
3. 调用 `agent.run(task, execute=True, run_dir=new_run_dir)` — agent 的 run() 接受外部 run_dir

**run_dir 冲突处理**：同一秒内多阶段创建目录时，自动添加 `_1`, `_2` 后缀。

### 4.3 SYSTEM_PROMPT 中的 W 专用规则

`agent.py` 的 SYSTEM_PROMPT 包含：
- 命令排序规则（11 条，违反即 fatal error）
- velocity 命令强制要求（无 velocity → T≈0）
- W 材料指南：`units metal`, `lattice bcc 3.165`, `pair_style eam/alloy`, `mass 1 183.84`, `timestep 0.001`
- read_restart 指导（restart 后仍需 pair_style/pair_coeff/mass，不需 lattice/create_atoms）

### 4.4 工作流 restart 文件链

| 工作流 | 阶段 1 写 | 阶段 2 读/写 | 阶段 3 读/写 | 阶段 4 读 |
|--------|----------|-------------|-------------|----------|
| Tensile | `restart.minimize` | `restart.minimize` → `restart.equilibrate` | `restart.equilibrate` | — |
| Nanoindent | `restart.minimize` | `restart.minimize` → `restart.equilibrate` | `restart.equilibrate` | — |
| Crack | `restart.cracked` | `restart.cracked` → `restart.minimized` | `restart.minimized` → `restart.loaded` | `restart.loaded` |
| DBTT | 同 Tensile × N 温度 | | | |
| Interface | `restart.interface` | → `restart.minimized` | → `restart.equilibrate` | `restart.equilibrate` |
| He Trapping | `restart.defects` | → `restart.minimized` | → `restart.he_inserted` → `restart.annealed` | `restart.annealed` |

### 4.5 Backend 抽象层（新增）

`backends/` 提供 `JobBackend` ABC，支持 `submit()`/`poll()`/`collect()` 接口：

- **LocalBackend**：包装现有 `run_lammps()`，同步执行，结果封装为 `SimulationResult`
- **SlurmBackend**：占位，预留 sbatch/squeue 接口
- **runner_adapter.py**：`get_backend(config)` 根据 `config.backend` 返回对应后端

**关键设计**：现有 `agent.py` → `lammps_runner.py` 的调用链**完全不变**。Backend 层仅供新代码使用，依赖方向：`新代码 → runner_adapter → backends/local → lammps_runner`。

### 4.6 Memory 格式

`runs/simulation_memory.jsonl`，每行一条 JSON：
```json
{
  "timestamp": "2026-04-17T19:54:32",
  "task": "...",
  "run_dir": "runs/20260417_195432",
  "return_code": 0,
  "summary": "return_code=0 | thermo_tail:... | convergence:...",
  "reward_score": 0.875,
  "error_type": null,
  "retry_count": 0,
  "script_hash": "1d0a34ebd6f5"
}
```

向后兼容：旧 JSONL（无 reward_score 等字段）可正常加载。

---

## 五、测试现状

### 单元测试（72/72 ✅）

```bash
python -m pytest tests/ -v
```

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_thermo_parser.py` | thermo 输出解析 |
| `test_reward.py` | 奖励信号计算 |
| `test_memory.py` | 新字段、序列化、向后兼容、reward 排序 |
| `test_skills_package.py` | 7 个原有 skill 导入与分析 |
| `test_core_types.py` | SimulationTask / SimulationResult |
| `test_backends.py` | LocalBackend / SlurmBackend |
| `test_runner_adapter.py` | get_backend / run_via_backend |
| `test_new_skills.py` | interface_analysis / defect_analysis |
| `test_new_workflows.py` | InterfaceMechanics / HeTrapping + 工厂函数 |
| `test_domain_extensions.py` | structures / dft / mlip 导入测试 |

### 端到端测试（全部 ✅，真实 LAMMPS 执行）

```bash
python tests/test_e2e_w_minimize.py           # 基础最小化
python tests/test_e2e_mechanics.py all        # 全部力学性能工作流 (~15s)
```

| 测试 | 阶段 | 耗时 | 关键输出 |
|------|------|------|----------|
| W BCC 最小化 | 1 | 0.2s | rc=0, reward=0.875 |
| Tensile | 3 | 1.8s | E=874 GPa, UTS=1.74 GPa |
| Nanoindent | 3 | 7.2s | P_max=82.93 nN |
| Crack | 4 | 2.2s | σ_max=-0.08 GPa |
| DBTT (2 temps) | 6 | 3.5s | 300K/600K 均通过 |

> 端到端测试 bypass LLM（mock generate_input_script），验证完整执行链。数值因小超胞+短平衡而偏高，不代表文献精度。

---

## 六、已修复的 Bug（重要，避免重复踩坑）

| Bug | 文件 | 现象 | 根因 | 修复 |
|-----|------|------|------|------|
| 势文件缺失 | `agent.py` | LAMMPS 报 `cannot open W_zhou.eam.alloy` | `run()` 不 copy 势文件到 run_dir | 新增 `_prepare_potentials()`，创建目录后自动 copy |
| Pipeline 文件不传递 | `pipeline.py` | stage 2 找不到 `restart.minimize` | `_pending_copy` 存引用但不执行 | 重写 `run()`：先创建 run_dir + copy，再传给 agent |
| run_dir 同秒冲突 | `agent.py` | 多阶段覆盖同一目录 | timestamp 精确到秒，`exist_ok=True` | 冲突时添加 `_1`, `_2` 后缀 |

---

## 七、.env 配置

没有 `.env` 文件时，自动 fallback 到 `.env.example`。用户使用 ERNIE（百度文心）作为 LLM 后端，通过 OpenAI 兼容接口调用。

关键配置项：
```
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=<百度 ERNIE API 地址>
OPENAI_API_KEY=<用户的 key>
OPENAI_MODEL=<ERNIE 模型名>
LAMMPS_COMMAND=lmp_mpi
ENABLED_SKILLS=thermo_tail,convergence,thermo_validator
BACKEND=local                    # "local" 或 "slurm"（slurm 为占位）
```

新增可选 skill 名（按需加入 ENABLED_SKILLS）：
- `interface_analysis`：界面粘附功分析
- `defect_analysis`：空位/间隙/He 泡检测

---

## 八、项目演进阶段

| Phase | 状态 | 内容 |
|-------|------|------|
| 1. MVP | ✅ 完成 | 单次 LLM 生成 + 执行 |
| 2. 自纠错 | ✅ 完成 | error_parser + 重试循环 + reward + memory 增强 + skills 重构 |
| 3. 流水线 | ✅ 完成 | potentials/ + pipeline + runner 增强 + SYSTEM_PROMPT W 规则 |
| 4. 力学性能 | ✅ 完成+验证 | workflows/ + model_builder + 4 个力学 skills + CLI + e2e 测试 |
| 4.5 平台扩展 | ✅ 完成 | backends/ + core_types + 新 skills/workflows + structures/dft/mlip 领域包 |
| 5. RL 自进化 | 未开始 | 经验回放 + Prompt 优化 + 可选 DPO/RLHF |

---

## 九、待做 / 下一步方向

### 短期（可立即执行）

- [ ] **LLM 端到端验证**：配好 ERNIE API key 后，用 `--task` 模式实际调用 LLM 生成 W 脚本
- [ ] **生产级数值验证**：用大超胞（10x10x20）+ 长平衡（50000 步）跑拉伸，对比文献值 E ≈ 380-420 GPa
- [ ] **DBTT 全温度范围**：11 个温度点（200K~1200K）完整扫描
- [ ] **SlurmBackend 实现**：在服务器上实现 sbatch 提交 + squeue 轮询 + 结果收集
- [ ] **interface_mechanics 工作流实跑**：需要先准备 oxide/W 势文件

### 中期

- [ ] **Phase 5 经验回放**：从 memory 提取高 reward 样本构建 few-shot examples
- [ ] **W-O 二元 ML 势训练**：实现 `mlip/trainer.py`，用 DeePMD-kit 或 MACE
- [ ] **InterfaceBuilder 实现**：`structures/interface_builder.py`，对接 Materials Project 晶体数据
- [ ] **VASPWorkflow 实现**：`dft/vasp_workflow.py`，生成 DFT 训练数据
- [ ] **更多势函数注册**：W-He EAM/FS+ZBL 等

### 长期

- [ ] ODS-W 完整四元体系（W-Y-O-Ti）ML 势
- [ ] DPO/RLHF 微调开源模型

---

## 十、文件速查

| 要做的事 | 看哪个文件 |
|----------|-----------|
| 改 LLM 调用逻辑 | `llm_clients.py` |
| 改 Agent 核心流程（重试/势文件/memory） | `agent.py` |
| 改 LAMMPS 执行方式（MPI/timeout） | `lammps_runner.py` |
| 改 SYSTEM_PROMPT（LLM 生成规则） | `agent.py` 顶部 `SYSTEM_PROMPT` |
| 加新势函数 | `potentials/registry.py` 的 `_register_builtin()` |
| 加新后处理 skill | 继承 `skills/base.py` → 注册到 `skills/__init__.py` |
| 加新工作流 | 继承 `workflows/base.py` → 注册到 `workflows/__init__.py` |
| 改 CLI 参数 | `main.py` 的 `build_parser()` |
| 改配置项 | `config.py` 的 `AgentConfig` + `load_config()` |
| 改 Pipeline 阶段逻辑 | `pipeline.py` 的 `SimulationPipeline.run()` |
| 改 W 模型几何 | `model_builder.py` 的 `TungstenModelBuilder` |
| 用 Backend 抽象执行仿真 | `runner_adapter.py` 的 `run_via_backend()` |
| 实现 Slurm 提交 | `backends/slurm.py` 的 `SlurmBackend` |
| 构建 oxide/W 界面模型 | `structures/interface_builder.py` 的 `InterfaceBuilder` |
| 训练 ML 势 | `mlip/trainer.py` 的 `MLIPTrainer` |
| 生成 DFT 训练数据 | `dft/vasp_workflow.py` 的 `VASPWorkflow` |
| 跑全部单元测试 | `python -m pytest tests/ -v` （72 个测试） |
| 跑端到端测试 | `python tests/test_e2e_mechanics.py all` |
| 查看 run 记录 | `runs/simulation_memory.jsonl` |

---

## 十一、服务器部署指南

### 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n auto-lammps python=3.11 -y
conda activate auto-lammps

# 2. 安装 LAMMPS（方式任选）
# 方式 A：conda-forge
conda install -c conda-forge lammps -y
# 方式 B：module load（HPC 集群）
module load lammps/2025

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env：
#   LAMMPS_COMMAND=lmp_mpi          (或 lmp, 取决于安装方式)
#   LAMMPS_POTENTIALS_DIR=/path/to/lammps/share/lammps/potentials
#   LLM_PROVIDER=local_ollama       (或 openai_compatible)
#   BACKEND=local                   (或 slurm)
```

### 势文件路径

服务器上 LAMMPS 势文件通常在以下位置之一：
- `$CONDA_PREFIX/share/lammps/potentials/`（conda 安装）
- `/usr/share/lammps/potentials/`（系统安装）
- module 安装：用 `which lmp_mpi` 找到可执行文件，势文件在 `../share/lammps/potentials/`

设 `LAMMPS_POTENTIALS_DIR` 指向上述目录即可，Agent 会自动发现并 copy 势文件。

### 验证安装

```bash
# 1. 验证 LAMMPS 可执行
lmp_mpi -help

# 2. 验证单元测试
python -m pytest tests/ -v

# 3. 验证端到端（需要 LAMMPS）
python tests/test_e2e_w_minimize.py
python tests/test_e2e_mechanics.py all
```

---

## 十二、注意事项

1. **不要用 `.venv/`**，解释器路径已失效，用 `conda activate auto-lammps`
2. **势文件不需要手动管理**，agent.run() 自动 copy 到每个 run_dir
3. **restart 文件由 pipeline 自动传递**，不需要手动 copy
4. **SYSTEM_PROMPT 中有 W 专用规则**，改其他材料时记得更新
5. **端到端测试 bypass LLM**（mock），验证 pipeline 逻辑；LLM 端到端需要配 API key
6. **W EAM 势文件随 LAMMPS 自带**，无需额外下载
7. **每次 run 的结果都写入 memory**（含 reward_score），影响后续 LLM 上下文
8. **Backend 层不影响现有代码**——agent.py 仍直接调用 run_lammps()，Backend 仅供新代码路径使用
9. **新 skills 默认不启用**——需在 `ENABLED_SKILLS` 中显式添加 `interface_analysis` 或 `defect_analysis`
10. **interface/he_trapping 工作流为骨架**——pipeline 结构已定义，但需要相应势文件才能实跑

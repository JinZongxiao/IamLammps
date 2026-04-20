# IamLAMMPS — LLM-Driven LAMMPS Simulation Agent

一个基于大语言模型的 LAMMPS 自动化仿真 Agent，支持从自然语言到完整力学性能测试的全流程：
- 根据自然语言任务自动生成 LAMMPS 输入脚本
- 内置错误解析 + 自动重试循环（最多 N 次自我修正）
- 奖励信号系统驱动脚本质量持续提升
- 多阶段 Pipeline 支持复杂仿真流水线（自动传递 restart 文件和势文件）
- 预置纯 W（钨）力学性能工作流（拉伸、纳米压痕、裂纹扩展、DBTT 扫描）
- 新增 oxide/W 界面力学、He 陷阱工作流骨架
- Backend 抽象层支持本地执行和 HPC Slurm 提交
- 自动势函数管理（自动发现 LAMMPS 安装目录 → copy 到 run_dir）
- 同时支持 OpenAI 兼容接口和本地模型（Ollama）

## 项目结构

```
src/lammps_agent/
    __init__.py          agent.py             config.py
    core_types.py        error_parser.py      lammps_runner.py
    llm_clients.py       main.py              memory.py
    model_builder.py     pipeline.py          reward.py
    runner_adapter.py    thermo_parser.py
    backends/
        __init__.py      base.py              local.py             slurm.py
    dft/
        __init__.py      vasp_workflow.py
    mlip/
        __init__.py      trainer.py
    potentials/
        __init__.py      registry.py          manager.py
    skills/
        __init__.py      base.py              thermo_tail.py
        convergence.py   thermo_validator.py  stress_strain.py
        indent_analysis.py  crack_analysis.py dbtt_fit.py
        interface_analysis.py  defect_analysis.py
    structures/
        __init__.py      interface_builder.py
    workflows/
        __init__.py      base.py              tensile.py
        nanoindent.py    crack.py             dbtt_sweep.py
        interface_mechanics.py  he_trapping.py
tests/
    conftest.py              test_thermo_parser.py
    test_reward.py           test_memory.py
    test_skills_package.py   test_e2e_w_minimize.py
    test_e2e_mechanics.py    test_core_types.py
    test_backends.py         test_runner_adapter.py
    test_new_skills.py       test_new_workflows.py
    test_domain_extensions.py
```

## 1. 安装

推荐使用 conda 环境 `auto-lammps`（已集成 LAMMPS 和 Python 依赖）：

```bash
conda activate auto-lammps
pip install -r requirements.txt
```

或使用 venv：

```bash
cd /path/to/IamLAMMPS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖：`openai`, `requests`, `python-dotenv`, `numpy`, `scipy`, `pytest`

## 2. 环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API 配置
```

### 核心配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `openai_compatible` | `openai_compatible` 或 `local_ollama` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容端点 |
| `OPENAI_API_KEY` | (空) | API Key，本地服务可留空 |
| `OPENAI_MODEL` | `gpt-4.1-mini` | 模型名 |
| `LOCAL_BASE_URL` | `http://127.0.0.1:11434` | Ollama 地址 |
| `LOCAL_MODEL` | `qwen2.5:7b` | Ollama 模型名 |
| `LAMMPS_COMMAND` | `lmp_mpi` | LAMMPS 可执行文件 |

### 仿真控制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_RETRIES` | `3` | 错误自动重试次数 |
| `LAMMPS_TIMEOUT` | (无) | 单次执行超时（秒） |
| `MPI_NP` | (无) | MPI 进程数 |
| `LAMMPS_POTENTIALS_DIR` | (自动发现) | 势函数目录，留空则自动搜索 LAMMPS 安装路径 |
| `BACKEND` | `local` | 执行后端：`local`（subprocess）或 `slurm`（HPC，占位） |

### Skills 与 Memory

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLED_SKILLS` | `thermo_tail,convergence,thermo_validator` | 启用的后处理 Skills |
| `MEMORY_FILE` | `runs/simulation_memory.jsonl` | 记忆文件路径 |
| `MEMORY_MAX_ENTRIES` | `200` | 最大记忆条数 |
| `MEMORY_CONTEXT_CHARS` | `2400` | 注入模型的记忆上下文字符数 |

## 3. 使用方式

### 单任务模式

```bash
# 生成脚本并执行
python -m src.lammps_agent.main \
  --task "构建BCC钨5x5x5超胞并最小化能量，使用EAM势" \
  --run

# 仅生成脚本
python -m src.lammps_agent.main \
  --task "创建FCC铝晶体并最小化能量" \
  --no-run

# 指定重试次数和超时
python -m src.lammps_agent.main \
  --task "LJ流体NVT平衡10000步" \
  --run --max-retries 5 --timeout 300
```

### 工作流模式

```bash
# 纯 W 力学性能（已验证）
python -m src.lammps_agent.main --workflow tensile --temperature 300 --supercell 10,10,20 --strain-rate 1e8
python -m src.lammps_agent.main --workflow nanoindent --temperature 300
python -m src.lammps_agent.main --workflow crack --temperature 300
python -m src.lammps_agent.main --workflow dbtt --strain-rate 1e8

# 新增工作流（骨架，需相应势文件）
python -m src.lammps_agent.main --workflow interface --temperature 300
python -m src.lammps_agent.main --workflow he_trapping --temperature 1000

# 通用选项
  --potential W_Zhou2004_EAM   # 势函数（默认 W Zhou 2004 EAM）
  --mpi-np 4                   # MPI 并行
  --timeout 600                # 每阶段超时
```

### 输出目录

```
runs/
  <timestamp>/
    in.lammps            # 生成的输入脚本
    in.lammps.retry1     # 重试脚本（如有）
    W_zhou.eam.alloy     # 自动 copy 的势文件
    restart.*            # 阶段间传递的 restart 文件
    stdout.log           # LAMMPS 标准输出
    stderr.log           # LAMMPS 错误输出
    result.json          # 结构化执行结果（含 wall_time、output_files）
  simulation_memory.jsonl  # 跨轮持久化记忆（含 reward_score）
```

## 4. 核心模块说明

### 错误解析 + 自动重试

`error_parser.py` 解析 LAMMPS 报错并分类（语法错误、势函数缺失、数值发散等），Agent 根据错误类型自动修正脚本并重试，最多 `MAX_RETRIES` 次。每次重试的脚本保存为 `in.lammps.retryN` 便于回溯。

### 奖励系统

`reward.py` 为每次仿真计算多维奖励信号：
- **script_validity** (0/1)：脚本是否成功执行
- **convergence** (0~1)：能量/温度是否收敛（尾部方差/全局方差）
- **physical_plausibility** (0~1)：物理量是否合理（T>0、能量非NaN、压力不发散）
- **efficiency** (0~1)：重试次数越少分越高

总分 = validity×0.4 + convergence×0.25 + plausibility×0.25 + efficiency×0.1

### 模拟记忆

`memory.py` 将每次运行结果（任务、返回码、奖励分数、错误类型、脚本哈希）写入 JSONL。后续生成脚本时，高奖励的成功经验优先注入模型上下文，减少重复犯错。向后兼容旧格式 JSONL。

### 后处理 Skills

可插拔的后处理分析框架，每个 Skill 实现 `analyze(run_dir) -> SkillReport`：

| Skill | 说明 | 端到端验证 |
|-------|------|-----------|
| `thermo_tail` | 提取 thermo 表尾部数据 | ✅ |
| `convergence` | 检查能量/温度收敛性 | ✅ |
| `thermo_validator` | 检查物理量合理范围 | ✅ |
| `stress_strain` | 应力-应变曲线分析（弹性模量、屈服强度、UTS） | ✅ 输出 E, σ_y, UTS |
| `indent_analysis` | Oliver-Pharr 纳米压痕分析（硬度、弹性模量） | ✅ 输出 P_max |
| `crack_analysis` | 裂纹 Mode I 加载分析（最大应力、K_IC） | ✅ 输出 σ_max |
| `dbtt_fit` | Boltzmann sigmoid 拟合 DBTT 转变温度 | ✅ |
| `interface_analysis` | 界面粘附功、分离功、峰值应力 | ✅ 单元测试 |
| `defect_analysis` | 空位/间隙/He 泡检测（配位数分析） | ✅ 单元测试 |

扩展方式：继承 `BasePostprocessSkill`，实现 `analyze()`，在 `skills/__init__.py` 注册即可。

### 势函数管理

`potentials/` 包自动管理势文件的全生命周期：
1. **发现**：`LAMMPS_POTENTIALS_DIR` 环境变量 → `which lmp_mpi` 相对路径 → Homebrew 默认路径
2. **注册**：内置 `W_Zhou2004_EAM`（Zhou 2004 W EAM，随 LAMMPS 安装自带）
3. **部署**：Agent 每次 `run()` 时自动 copy 所有已注册势文件到 run_dir，LAMMPS 通过相对路径读取

### 多阶段 Pipeline

`pipeline.py` 支持多阶段仿真流水线：
- Pipeline 先创建 run_dir，copy 上一阶段的 restart/data 文件，再调用 Agent 生成并执行脚本
- 势文件在每个阶段的 run_dir 中自动可用
- 同一秒内多阶段自动添加后缀避免 run_dir 冲突（`_1`, `_2`, ...）
- 任一阶段失败立即中止并报告 `failed_stage`

### Backend 抽象

`backends/` 包提供执行后端抽象，支持未来 HPC 部署：

| 后端 | 文件 | 说明 |
|------|------|------|
| `LocalBackend` | `backends/local.py` | 包装 `run_lammps()` 的同步执行后端（默认） |
| `SlurmBackend` | `backends/slurm.py` | Slurm HPC 提交占位（sbatch/squeue） |

通过 `runner_adapter.py` 的 `run_via_backend()` 调用，或设 `BACKEND=slurm`。现有 Agent 核心流程不受影响。

### 工作流

`workflows/` 包封装了完整力学测试和研究工作流：

| 工作流 | 阶段数 | 阶段链 | 状态 |
|--------|--------|--------|------|
| **TensileWorkflow** | 3 | minimize → NPT equilibrate → fix deform 拉伸 | ✅ e2e 验证 |
| **NanoindentWorkflow** | 3 | minimize → equilibrate → fix indent 球形压头 | ✅ e2e 验证 |
| **CrackWorkflow** | 4 | build_crack → minimize → Mode I load → analyze | ✅ e2e 验证 |
| **DBTTSweepWorkflow** | 3×N | 对 N 个温度点循环调用 TensileWorkflow | ✅ e2e 验证 |
| **InterfaceMechanicsWorkflow** | 4 | build_interface → minimize → equilibrate → tensile_normal | 骨架就绪 |
| **HeTrappingWorkflow** | 5 | create_defects → minimize → insert_he → anneal → analyze | 骨架就绪 |

### 领域扩展（占位）

| 模块 | 说明 | 状态 |
|------|------|------|
| `structures/interface_builder.py` | oxide/W 界面模型构建器（Y2O3/Al2O3） | 接口定义，待实现 |
| `dft/vasp_workflow.py` | VASP DFT 计算（训练数据生成） | 接口定义，待实现 |
| `mlip/trainer.py` | ML 势训练（DeePMD/MACE/ACE） | 接口定义，待实现 |

## 5. 测试

### 单元测试

```bash
conda activate auto-lammps
python -m pytest tests/ -v
```

72 个单元测试，覆盖：thermo 解析、奖励计算、记忆系统、Skills（含新增 interface/defect）、Backend 抽象、工作流工厂、领域扩展包导入。

### 端到端测试（真实 LAMMPS 执行）

```bash
# W BCC 最小化（最简单，~0.2s）
python tests/test_e2e_w_minimize.py

# 单个工作流
python tests/test_e2e_mechanics.py tensile
python tests/test_e2e_mechanics.py nanoindent
python tests/test_e2e_mechanics.py crack
python tests/test_e2e_mechanics.py dbtt

# 全部力学性能工作流（~15s）
python tests/test_e2e_mechanics.py all
```

端到端测试 bypass LLM（用已知正确的脚本），验证完整执行链：势文件 copy → restart 文件传递 → LAMMPS 执行 → skill 分析 → reward 计算 → memory 写入。

### 测试结果

| 测试 | 环境 | 结果 |
|------|------|------|
| 单元测试 (72) | auto-lammps Python 3.11 | ✅ 72/72 通过 |
| W BCC 最小化 | lmp_mpi (LAMMPS 22 Jul 2025) | ✅ rc=0, 0.16s, reward=0.875 |
| Tensile 3 阶段 | lmp_mpi | ✅ E=874 GPa, UTS=1.74 GPa |
| Nanoindent 3 阶段 | lmp_mpi | ✅ P_max=82.93 nN |
| Crack 4 阶段 | lmp_mpi | ✅ σ_max(Mode I)=-0.08 GPa |
| DBTT 2 温度点 | lmp_mpi | ✅ 300K/600K 均通过 |

> **注意**：端到端测试使用小超胞 + 短时间步以加速，力学性能数值仅验证 pipeline 逻辑，不代表文献精度。生产级模拟应使用更大超胞（10x10x20+）和更长平衡时间。

## 6. 服务器部署

### 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n auto-lammps python=3.11 -y
conda activate auto-lammps

# 2. 安装 LAMMPS（方式任选）
conda install -c conda-forge lammps -y     # 方式 A：conda-forge
# module load lammps/2025                  # 方式 B：HPC module

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env：
#   LAMMPS_COMMAND=lmp_mpi
#   LAMMPS_POTENTIALS_DIR=/path/to/lammps/share/lammps/potentials
#   LLM_PROVIDER=local_ollama  (或 openai_compatible)
#   BACKEND=local              (或 slurm)
```

### LLM 后端选择

1. **Ollama**：`LLM_PROVIDER=local_ollama`，在可联网机器 `ollama pull` 后拷贝模型目录
2. **OpenAI 兼容服务**（vLLM / TGI / llama.cpp / LM Studio）：`LLM_PROVIDER=openai_compatible`，`OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
3. **ERNIE（百度文心）**：`LLM_PROVIDER=openai_compatible`，配置百度 API 的 base_url 和 key
4. **多模型切换**建议用 OpenAI 兼容接口统一接入

### 势文件路径

服务器上 LAMMPS 势文件位置取决于安装方式：
- conda 安装：`$CONDA_PREFIX/share/lammps/potentials/`
- 系统安装：`/usr/share/lammps/potentials/`
- module 安装：`$(which lmp_mpi)/../share/lammps/potentials/`

设 `LAMMPS_POTENTIALS_DIR` 指向上述目录，或留空让 Agent 自动发现。

### 验证安装

```bash
lmp_mpi -help                              # 验证 LAMMPS
python -m pytest tests/ -v                 # 72 个单元测试
python tests/test_e2e_w_minimize.py        # 基础 e2e
python tests/test_e2e_mechanics.py all     # 全力学性能 e2e
```

## 7. 注意事项

- 力学性能工作流目前针对纯 W (BCC, a=3.165 A) 优化，其他材料需调整 `model_builder.py` 和势函数
- 物理参数请按课题自行校验，Agent 生成的脚本仅作为起点
- 势函数自动从 LAMMPS 安装目录发现并 copy 到每个 run_dir，无需手动管理
- Pipeline 阶段间 restart 文件自动传递，支持 `read_restart` 继续模拟
- Backend 抽象层已就绪，SlurmBackend 为占位实现，需在服务器上补充 sbatch/squeue 逻辑
- `interface` 和 `he_trapping` 工作流为骨架，pipeline 结构已定义，需相应势文件实跑
- 新 skills（interface_analysis、defect_analysis）默认不启用，需在 ENABLED_SKILLS 中显式添加
- 后续目标：ODS-W 合金（W + Y₂O₃ 弥散强化）多尺度模拟

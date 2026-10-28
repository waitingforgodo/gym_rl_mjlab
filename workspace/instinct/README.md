# InstinctMJ：训练、评测与仿真工具链

这是新增的 `instinct` 后端，使用官方 [InstinctMJ](https://github.com/project-instinct/InstinctMJ) 和固定版本的 `instinct_rl`。你提供的 InstinctLab 是 Isaac Lab 版本；InstinctMJ 才是同一项目的 mjlab 移植。继续使用本仓库的原 Python / PyTorch / CUDA / mjlab 环境，源码和附加依赖只在本方法进程中加载。

新任务可以从这里开始。原 `amp`、`beyondmimic`、`velocity`、`supermimic` 入口保留，原模型仍走原入口。两套 G1 根刚体、关节约定、观测与模型格式不同，不能直接共用 checkpoint。

## 1. 检查与准备

在项目根目录执行：

```bash
bash g1.sh methods
bash g1.sh instinct tasks
bash g1.sh instinct doctor

# 首次拉取本仓库，或 doctor 显示缺少附加包/网格时执行。
# 安装到 .cache/instinct-deps，--no-deps 不升级原环境的核心包。
bash g1.sh instinct setup
```

本机已完成 setup。源码、文本配置和许可证纳入仓库；网格按固定上游提交下载并校验 SHA-256，训练结果和参考动作仍只保存在本地。新机器首先要有本项目原运行栈和 `.runtime-path`；`instinct setup` 只补辅助包，不是完整 CUDA 环境安装器。

| 本地 profile | 上游任务 | 默认环境数 | 默认新增更新数 | 参考数据 |
| --- | --- | --- | --- | --- |
| `locomotion` | G1 平地速度跟踪 | 4096 | 5000 | 不需要 |
| `beyondmimic` | G1 BeyondMimic 逐帧跟踪 | 4096 | 30000 | 下方转换后的 NPZ |
| `shadowing` | G1 Whole-body Shadowing | 4096 | 30000 | 下方转换后的 NPZ |
| `beyondmimic_lafan` | G1 BeyondMimic 多动作跟踪 | 4096 | 30000 | LAFAN1 40 段动作 |
| `shadowing_lafan` | G1 Whole-body Shadowing 多动作 | 4096 | 30000 | LAFAN1 40 段动作 |
| `parkour` | G1 Parkour Target AMP | 256 | 30000 | LAFAN walk/run/sprint 子集 |
| `perceptive` | G1 Perceptive Shadowing | 512 | 50000 | 动作与地形配对场景 |
| `perceptive_one_motion` | G1 Perceptive OneMotion | 512 | 50000 | 动作与地形配对场景 |
| `perceptive_vae` | G1 Perceptive VAE | 256 | 50000 | 配对场景 + 已训练感知 teacher |
| `hoi` | G1 Perceptive HOI Shadowing | 256 | 50000 | OMOMO 风格对象轨迹、网格与映射 |

若要训练本地舞蹈，先转换已有标准动作：

```bash
# 如果本地还没有 amp_dance.npz，先运行现有 AMP 数据准备入口。
bash g1.sh amp prepare

# pelvis 根 → torso 根，按关节名映射并修正三处 waist 角度符号。
bash g1.sh instinct prepare --source motions/processed/amp_dance.npz
```

默认输出 `motions/processed/instinct/amp_dance_retargeted.npz` 和转换检查 JSON。相同源与输出可复用；源文件不覆盖。其他标准 50 Hz G1 NPZ 可传 `--source`，并通过 `--output /路径/新动作_retargeted.npz` 指定输出；训练时用 `--motion` 选择它。不是相同骨架的数据必须先重定向，不能只改键名。

多动作、感知和 HOI profile 使用显式数据准备与校验：

```bash
# 下载固定版本的 40 段 LAFAN1 G1 CSV，生成标准/Instinct NPZ；
# 其中 18 段 walk/run/sprint 另存为 Parkour 工程基线。
bash g1.sh instinct prepare-lafan

# 生成平地 + 深度相机的感知集成 demo。
bash g1.sh instinct prepare-scene --demo

# 校验用户提供的真实配对地形或 HOI 数据，不复制或猜测缺失文件。
bash g1.sh instinct prepare-scene --dataset /数据目录 --kind perceptive
bash g1.sh instinct prepare-scene --dataset /数据目录 --kind hoi
```

感知数据的 `metadata.yaml` 必须同时列出 `terrains` 和 `motion_files`；`terrain_id` 为从 0 开始、连续、有序的非负整数，所有引用只能指向数据目录内部。`--demo` 只验证平地、深度相机和训练链路，不是扫描地形 benchmark。HOI 数据还需要 `objects.json` 的对象名到网格/尺度映射，以及 `subXX_OBJECTNAME_*_retargeted.npz`；每段轨迹必须包含 `object_pos_w` 和 `object_quat_w`。

当前 Parkour 默认使用上述 18 段 LAFAN locomotion 子集，是可运行的工程基线，不是论文专用跑酷数据或论文结果复现。HOI profile 已完成工厂与数据契约接线，但本仓库不分发真实 OMOMO 数据。

## 2. 训练与续训

```bash
# 检查配置、来源与动作文件，不创建物理仿真。
bash g1.sh instinct train --profile locomotion --dry-run
bash g1.sh instinct train --profile beyondmimic --dry-run

# 按需选择一条；名称应唯一，已有运行不会被覆盖。
bash g1.sh instinct train --profile locomotion --iterations 5000 --name g1_walk
bash g1.sh instinct train --profile beyondmimic --iterations 30000 --name g1_dance
bash g1.sh instinct train --profile shadowing --iterations 30000 --name g1_shadow
bash g1.sh instinct train --profile beyondmimic_lafan --name g1_lafan
bash g1.sh instinct train --profile perceptive --name perceptive_teacher

# VAE 训练必须显式绑定已训练的 Perceptive/OneMotion teacher。
bash g1.sh instinct train --profile perceptive_vae \
  --teacher-checkpoint /绝对路径/perceptive_teacher_perceptive/model_N.pt \
  --name perceptive_vae

# 追加 1000 次 PPO 更新，恢复原任务、动作、网络、归一化器及优化器。
bash g1.sh instinct resume --profile beyondmimic --checkpoint latest \
  --iterations 1000 --name g1_dance_more
```

`train` / `resume` 的 `--iterations` 均为**本次新增 PPO 更新数**，不是 SuperMimic 的总控制步数。可用 `--envs 512` 降低并行量，`--seed 123` 更改种子。默认设备 `cuda:0`；全仓库训练共用一把锁，防止误启动多个训练抢占显卡。Ctrl+C 会尝试保留当前模型并记录 `interrupted`，资源报告区分请求更新数与实际完成更新数。

运行目录为 `outputs/instinct/runs/<name>_<profile>/`，保存 `model_N.pt`、`run_manifest.json`、`params/env.yaml`、`params/agent.yaml`、TensorBoard 日志和 `resources.json`。`RUN_DIR=` 会在终端打印完整路径。

`latest` 只搜索 Instinct 运行。传 `--profile` 可以继续限定任务；建议正式评估和部署写明确模型路径，例如 `outputs/instinct/runs/g1_dance_beyondmimic/model_30000.pt`。原 AMP/RSL-RL/SuperMimic 模型会在反序列化前被拒绝。

自定义配置用 `--config /路径/配置.json`：根对象只接受 `profile`、`env_overrides` 和 `agent_overrides`；若命令行也传 `--profile`，两者必须一致。环境和算法字段通过点号路径修改已有字段，未知字段拒绝；task、数据路径和 teacher 不能藏在配置文件中改写，分别使用对应 CLI 参数。不要修改已保存运行的配置快照。一个新训练配置的例子：

```json
{
  "profile": "beyondmimic",
  "env_overrides": {"episode_length_s": 10.0},
  "agent_overrides": {"algorithm.learning_rate": 0.0003}
}
```

```bash
bash g1.sh instinct train --profile beyondmimic --config /路径/配置.json --dry-run
```

`--motion` 与 `--motion-dir` 互斥；场景任务从 `--scene-dataset` 取动作，不能再传通用 motion。VAE teacher 只能来自 `perceptive` 或 `perceptive_one_motion`，且必须是该 run 最新的数字 checkpoint；训练会同时固定模型和 `params/agent.yaml` 的 SHA-256。续训继承旧快照，不允许通过 `--config`、motion、motion directory、scene dataset 或 teacher 偷换任务与数据；需要修改时创建新训练。

## 3. 曲线、网页回放和评测

```bash
bash scripts/07_tensorboard.sh
bash g1.sh instinct summary --profile beyondmimic --checkpoint latest

# 输出可点击网页地址 http://localhost:8080。
bash g1.sh instinct play --profile beyondmimic --checkpoint latest --port 8080

# 固定 500 个控制步、16 个环境；默认控制周期 0.02 s，即 10 秒。
bash g1.sh instinct evaluate --profile beyondmimic --checkpoint latest --envs 16 --steps 500
```

如果浏览器位于另一台电脑，需要 VS Code/SSH 转发相应端口。`--viewer native` 可使用本地图形窗口；`play --headless --steps 500` 可以只跑无界面检查。

评测输出到 `outputs/instinct/evaluations/`：平均每步奖励、真实失败事件、各环境**首回合存活时间**及尚未结束的首回合数。超时/动作结束与跌倒分开处理；失败后的自动重置不会被误算成首回合成功。不同 profile 的奖励量纲不同，不能只用奖励排名；这里是固定时间基础评测，还不是整套论文 benchmark。

## 4. 导出与独立仿真部署

```bash
bash g1.sh instinct export --profile beyondmimic --checkpoint latest

# 将上条命令输出的 DEPLOY_BUNDLE= 后的完整目录替换到下面。
bash g1.sh instinct deploy --bundle /完整路径/部署包 --headless --steps 500
bash g1.sh instinct deploy --bundle /完整路径/部署包 --engine onnx --headless --steps 500
bash g1.sh instinct deploy --bundle /完整路径/部署包 --port 8080
```

导出包位于 `outputs/instinct/deploy/`，包含：

- `policy_jit.pt` / `policy.onnx`：含冻结观测归一化器的 actor，输入 `obs`，输出 `actions`；当前仅支持前馈策略。
- `manifest.json`：源提交、运行版本、任务配置、关节/动作顺序、动作比例与偏置、初始关节角、控制周期、观测格式和一致性误差。
- `motion_retargeted.npz`：单动作跟踪任务所需的参考动作；行走任务不带此文件。
- `motion_dataset/`：LAFAN 等多动作数据集的递归副本。
- `scene_dataset/`：感知/HOI 的 metadata、动作和网格递归副本。
- `hashes.json`：包内所有文件的递归 SHA-256；部署同时拒绝绝对路径和路径穿越，运行后另写 `simulation_result.json`。

导出会采样真实环境观测，与训练策略进行 JIT 和 ONNX Runtime 数值对照。VAE 部署采用 posterior mean 作为确定性 latent，并在 manifest 写入 `vae_latent_policy`；部署包不保存 teacher 路径，只保留 teacher profile、checkpoint SHA-256 和参数 SHA-256 的 `teacher_provenance`。`deploy` 直接加载导出 actor，不构造 PPO runner，也不再读取训练 checkpoint。包可以换目录；依然需要本机相同后端源码、运行环境和机器人网格。JIT 在所选设备推理，ONNX 当前采用 CPUExecutionProvider，物理仿真仍可运行在 GPU。

部署范围为**同后端仿真**。该 torso-root 模型与现有 pelvis-root / 真机 SDK 的 waist 方向约定不同，真机映射未验收，清单中 `hardware_enabled=false`。

## 5. 源码入口与已验证范围

```text
instinct/
├── cli.py                     所有功能参数与分发
├── bootstrap.py, runtime.py    项目内依赖、资产恢复、源哈希及运行环境
├── configuration.py           上游工厂 + profile / 覆盖 / 动作选择
├── data.py                    torso 根变换、waist 符号与 FK 校验
├── training.py                原生 runner 训练/续训、快照、计时和资源
├── checkpoints.py             本后端模型定位与契约检查
├── inference.py               原生模型回放、基础评测、网页服务
├── deployment.py              JIT/ONNX 数值校验和独立仿真
├── profiles/                  十份基础/数据集/场景配置（256–4096 环境）
├── requirements-addons.txt     仅本项目使用的附加包版本
└── upstream/
    ├── InstinctMJ/             官方 mjlab 任务与机器人源码
    └── instinct_rl/            匹配版本的 PPO/AMP 等算法源码
```

本机已通过三种基础 profile 各 4096 环境、1 次真实更新，以及 4 环境、1 次更新的 Perceptive、OneMotion、VAE、LAFAN BeyondMimic、LAFAN Shadowing 和 Parkour。VAE 另通过续训、无界面评测、JIT/ONNX 导出与两种引擎独立部署；实测导出最大绝对误差分别为 0.0 和约 2.68e-7。40 段 LAFAN motion directory 也已完成递归打包、JIT/ONNX 导出和两种引擎各 10 步部署。HOI 因本机没有真实 OMOMO 轨迹/网格，只验收到创建仿真前的缺数据明确拒绝。76 项完整回归测试全部通过。这些是链路和短时容量检查，不证明长训收敛；Parkour 的 LAFAN 基线尤其不能冒充论文复现。

固定源码、环境差异、与原实现的替换关系、测试数值及许可证见 [接入说明](../../docs/InstinctMJ接入与完整工具链.md)。上游共有 16 个注册任务 ID（含 train/play 对）；本地 10 个 profile 覆盖基础、数据集和场景任务，但真实感知 benchmark 与 HOI 仍需用户提供匹配数据。固定的两个上游提交采用 CC BY-NC 4.0，并保留文件原有许可声明。

# InstinctMJ 接入与现有工具链的关系

## 采用哪个仓库

用户提供的 [InstinctLab](https://github.com/project-instinct/InstinctLab) 主分支仍是 Isaac Lab / Isaac Sim 工程。同一团队的 [InstinctMJ](https://github.com/project-instinct/InstinctMJ) 才是基于 mjlab 的官方移植，目标 mjlab 1.6.0，与本机版本相同。本仓库接入 InstinctMJ；训练和部署命令集中在 [workspace/instinct/README.md](../workspace/instinct/README.md)。

这里“保留环境”指保留原 Python / CUDA / PyTorch / mjlab 安装。InstinctMJ 自己的机器人模型、MDP、奖励和任务配置与旧后端有差异，新训练使用其原生配置，不能声称物理场景与旧任务完全相同。

| 来源 | 固定提交 | 用途 |
| --- | --- | --- |
| InstinctMJ | `6edbc19d4e386ba42ec8a0f8c705ec4a2f983ffb` | 机器人、motion reference、环境/奖励/任务、viewer 连接 |
| instinct_rl | `3a2844890387eda6d93a4465cdef9e767aba8546` | 与上述移植匹配的 1.0.2 算法实现 |

完整上游文本源码、配置、文档与许可证在 `workspace/instinct/upstream/`，共 279 个文件；每份 `SOURCE_INFO.json` 记录来源和文件 SHA-256。另有 84 个本地二进制资源，Git 忽略它们，`instinct setup` 按相同提交下载并校验。接入层不修改上游算法文件，避免混淆移植代码与本地改动。

两份固定提交的根许可证都是 **CC BY-NC 4.0**，不是允许任意商业使用的宽松许可；保留所有文件内原有 BSD 等声明。应以对应提交的 [InstinctMJ LICENSE](https://github.com/project-instinct/InstinctMJ/blob/6edbc19d4e386ba42ec8a0f8c705ec4a2f983ffb/LICENSE) 和 [instinct_rl LICENSE](https://github.com/project-instinct/instinct_rl/blob/3a2844890387eda6d93a4465cdef9e767aba8546/LICENSE) 为准，不能用另一个分支的新许可替代。

## 与原代码如何分工

| 功能 | 原入口与实现 | 新入口与用途 |
| --- | --- | --- |
| 平地速度跟踪 | `velocity`，mjlab + RSL-RL | `instinct --profile locomotion`，新的原生任务选择 |
| BeyondMimic 跟踪 | `beyondmimic`，mjlab 原跟踪任务 | `instinct --profile beyondmimic`，使用 InstinctMJ 原生 G1 跟踪 |
| Whole-body Shadowing | 原工具链无对应专用任务 | `instinct --profile shadowing`，本轮新增 |
| AMP 舞蹈 | `amp`，本地 AmpPPO + 专家舞蹈特征 | 继续保留；InstinctMJ 的 Parkour AMP 不是舞蹈任务的直接替代 |
| SuperMimic | `supermimic`，独立复现的世界模型 + BPTT | 继续保留；InstinctMJ 未实现这个世界模型训练器 |
| 感知/OneMotion/VAE | 原工具链没有完整专用配置 | 场景数据契约、平地集成 demo、teacher 固定及训练/部署链路已接入；真实 benchmark 仍需匹配地形 |
| Parkour | 原工具链没有对应任务 | 已接入 LAFAN locomotion 工程基线；论文专用跑酷数据与结果未复现 |
| HOI | 原工具链没有对应任务 | 工厂和对象数据契约已接入；真实 OMOMO 轨迹/网格不随仓库提供 |

因此新增 `instinct` 后端作为新行走与模仿实验入口，同时保留已有训练器和模型。没有删除 AMP 等有效功能，也没有伪装成替换了所有算法。`bash g1.sh methods` 列出五个入口、后端和 profiles；各自 `latest` 分开解析，训练共用全仓库锁。

旧模型和部署包继续从原入口读取，`outputs/runs/`、`outputs/deploy/` 与基准模型均未迁移或修改。Instinct 的新产物位于 `outputs/instinct/`。模型契约记录两个源提交、核心运行版本、profile、数据哈希；错误后端和 profile 不匹配在加载权重前拒绝。

## 运行环境与安装边界

本次使用原 `.runtime-path` 的解释器，实际核心版本为：

| 包 | 本机版本 |
| --- | --- |
| PyTorch | 2.9.1+cu128 |
| mjlab | 1.6.0 |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0 |
| Warp | 1.14.0 |

`instinct_rl` 采用 InstinctMJ 锁定的提交，其 PyTorch 要求允许本机版本；没有直接拉最新算法分支并升级原环境。附加包安装到 `.cache/instinct-deps/`，仅在 Instinct 命令的 `sys.path` 中加入。setup 使用精确版本及 `--no-deps`，禁止把 torch、numpy、mjlab、MuJoCo、Warp、RSL-RL 等核心包加入该安装清单。

这是经过本机短测的兼容接入，不是原样复刻上游 lock：本机 NumPy 为 2.5.4，超过上游声明的 `<2.5`；为保留原环境没有降级。`doctor` 会明确报告这一偏差。`pyvista` 尚未安装，部分上游高级可视化/地形工具可能需要它；本文列出的训练链路不依赖该可选包。不能由短测 profile 的通过推断所有上游工具都已兼容。

## 动作转换为什么需要专门实现

旧 mjlab G1 的自由根为 pelvis，链条是 `pelvis → yaw → roll → pitch → torso`。InstinctMJ 使用 torso-root，waist 链反向为 `torso → pitch → roll → yaw → pelvis`。关节名相同，但沿反向链使用原正轴时，三个 waist 角都必须取负；只重排 29 列会给出错误姿态。

`workspace/instinct/data.py` 执行以下步骤：

1. 检查原标准 NPZ 的关节/刚体数组、有限值及 50 Hz 频率。
2. 按原动作中的 `torso_link` 世界位置和 wxyz 四元数建立新根，按名字映射关节。
3. 反转 waist yaw / roll / pitch 三个角度符号。
4. 对最多 32 帧运行目标 G1 MuJoCo 正运动学，对比所有同名刚体；最大位置误差超过 3 cm 时拒绝快捷转换，要求重新 retarget。
5. 写出 `framerate / joint_names / joint_pos / base_pos_w / base_quat_w` 和检查元数据，保留原文件。

本机 `amp_dance.npz` 转换后的采样 FK 最大位置差约 **0.01094 m**。校正前仅复制 waist 角时，最大差可达约 0.35 m。剩余厘米级差异来自两份模型的骨架/几何定义，不能把快捷转换理解为通用机器人重定向器。转换元数据记录输入/输出哈希、反号关节与误差；同一源可重复执行，变更源不会覆盖旧输出。

训练配置把参考动作选择文件写入本仓库 `.cache/instinct-motion-selections/`，替换上游工厂里的开发者数据路径。单动作、motion directory 和 scene dataset 都会在训练前校验并固定哈希；感知地形 ID 还必须是从 0 开始连续、有序的整数。导出时把实际动作或整个数据集递归复制进包并改为相对路径，部署包换位置后仍可找到它；不会依赖训练机原动作目录。

LAFAN1 入口固定 Hugging Face 数据 revision，逐文件校验 git object ID，再复用固定 SuperMimic 转换器生成 40 段 50 Hz 标准 NPZ 和 Instinct torso-root NPZ；其中 18 段 walk/run/sprint 形成 Parkour 的可运行基线。平地感知 demo 则明确标记 `real_scanned_terrain=false`，只用于验证深度相机与场景链路。HOI 校验要求 `objects.json`、内部相对网格路径，以及文件名与对象名一致、包含 `object_pos_w/object_quat_w` 的真实轨迹；缺项会在创建 GPU 仿真前报错。

## 训练和部署实现

`training.py` 通过上游 registry 创建原生配置、环境、vec wrapper 和 runner，真实 PPO 更新仍由 `instinct_rl` 执行。每次运行保存实际 env/agent YAML、配置快照、源契约和资源记录；runner 计时采用单调时钟，避免 WSL 校时导致负耗时。续训恢复优化器和归一化器，沿用原数据哈希。VAE 训练必须显式指定 Perceptive/OneMotion teacher，并同时固定最新数字 checkpoint 与 `params/agent.yaml` 的 SHA-256，续训不能更换。两种训练预算含义需要区分：Instinct 的 `--iterations` 是追加 PPO 更新数，SuperMimic 是目标总控制步。

`inference.py` 提供网页/native 回放，以及固定步数的基础评测，记录奖励、失败事件、首回合时长与未结束环境数。失败后自动 reset 不会把首回合时长续上。该协议用于观察策略基本表现；完整数据集成功率、整段动作误差、多种子统计和真实扰动 benchmark 仍需要后续专项评估。

`deployment.py` 将冻结观测归一化器和 actor 一起导出，分别对 JIT / ONNX Runtime 做真实观测数值检查。独立 deploy 只加载导出策略，重建相同仿真任务，不需要训练模型或 PPO runner。单动作、多动作目录、场景数据都随包 vendoring，`hashes.json` 递归覆盖所有文件并拒绝路径穿越。动作顺序、比例、偏置、初始关节角和控制周期写入 manifest；机器人 PD 等由固定源码与保存的配置覆盖重建。VAE 用 posterior mean 取代训练时采样 latent，保证导出与部署确定性；teacher 路径不进入包，只保留 profile 和两个哈希的 provenance。当前仅支持前馈导出；循环策略需显式隐藏状态接口，不会静默按普通 MLP 导出。

当前真机接口未验收，尤其 torso-root waist 与硬件 SDK 的符号关系不能假定相同；部署清单明确禁用硬件。这里完成的是同后端仿真工具链。

## 本次验收

运行 GPU 为本机 RTX 5070 Laptop 8 GiB；以下均为短测：

| 检查 | 结果 |
| --- | --- |
| 全仓库 CPU/契约/接口回归 | 76 项通过，包含新后端隔离、未知配置/数据拒绝、teacher 固定、递归包完整性/路径边界、轻量帮助及真实 CPU FK 转换 |
| 行走 | 8 环境 2 次更新，续训 1 次；另完成 4096 环境 1 次更新 |
| BeyondMimic | 修正根/waist 映射后，8 环境与 4096 环境各 1 次更新 |
| Whole-body Shadowing | 8 环境与 4096 环境各 1 次更新 |
| 4096 Shadowing 资源 | 最低剩余 GPU 显存约 2132 MiB，短测无 OOM；不代表长时温度或资源峰值验收 |
| Perceptive / OneMotion | 各 4 环境、1 次更新；使用明确标记的平地感知 demo |
| Perceptive VAE | 4 环境、1 次更新并续训 1 次；teacher checkpoint 与 agent 参数双哈希固定 |
| LAFAN 多动作 | 固定 revision 的 40 段动作完成下载/标准转换/Instinct 转换；BeyondMimic 与 Shadowing 各 4 环境、1 次更新 |
| Parkour 工程基线 | 18 段 LAFAN walk/run/sprint，4 环境、1 次更新；不是论文跑酷数据或效果复现 |
| HOI 数据边界 | 本机无真实 OMOMO 数据；缺少 `objects.json`、网格或对象轨迹时在创建仿真前明确拒绝 |
| 行走导出 | 真实观测 JIT 最大绝对误差约 8.20e-8，ONNX Runtime 约 8.94e-8 |
| BeyondMimic 导出 | JIT 最大绝对误差约 1.79e-7，ONNX Runtime 约 1.38e-7 |
| LAFAN 多动作导出 | 40 段 motion directory 递归随包复制；JIT 误差 0.0，ONNX Runtime 约 1.79e-7 |
| Perceptive VAE 导出 | posterior-mean 确定性策略；JIT 最大绝对误差 0.0，ONNX Runtime 约 2.68e-7 |
| 独立部署 | 行走两种引擎各 60 步；BeyondMimic 包复制到不同目录后两种引擎各 40 步 |
| LAFAN 多动作部署 | JIT/ONNX 均从包内 40 段 `motion_dataset/` 各完成 10 步无界面仿真 |
| VAE 独立部署 | scene dataset 随包复制后，JIT/ONNX 两种引擎各完成 10 步无界面仿真 |
| 原模型与数据 | 原训练/部署文件、动作、配置与 AMP 原算法保护项核对保留 |

这些数字证明配置、采样、更新、保存、恢复、导出和仿真入口能连通，不证明学会舞蹈或优于旧 AMP。仅更新 1–3 次的测试策略仍会失败，不能作为正式效果验收。临时短测模型、部署包及评测文件在验收后清理；保留动作转换结果和本地依赖，用户从新训练开始。

基础三 profile 的首轮验收汇总保存在 `outputs/checks/20261028_instinct_toolchain_validation.json`，对应旧回归日志为同目录 `20261028_instinct_unit_tests.log`；高级 profile 的运行记录保存在 `outputs/instinct/runs/` 与部署包中。这些都属于本地产物，不上传源码仓库。

完整复制命令、参数、文件布局、曲线与脚本见 [Instinct 使用说明](../workspace/instinct/README.md)；全仓库入口见 [workspace 总览](../workspace/README.md)。

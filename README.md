# G1 · mjlab 仿真训练与部署工作空间

AMP 与 BeyondMimic 快捷训练现采用独立的 2048 环境配置，原 256 配置保留以兼容旧模型。容量扫描、配置变化及训练记录见 [2048 并行验证](docs/并行规模验证与2048训练.md)。

AMP 舞蹈已新增独立实现，入口为 `scripts/17_prepare_amp_motion.sh` 和 `scripts/18_train_amp_dance.sh`。源码、判别器原理、配置和完整流程见 [AMP 舞蹈训练与仿真部署](docs/AMP舞蹈训练与仿真部署.md)。

新增的行走、动作模仿候选配置及公开来源对照见 [训练配置优化依据与对照实验](docs/训练配置优化依据与对照实验.md)。新行走入口为 `scripts/15_train_robust.sh`，跟踪入口为 `scripts/16_train_tracking_robust.sh`；旧基线保留，收益需相同协议评估确认。

完整代码位置、独立 Python 程序和从训练到仿真部署的逐步说明，见 [完整训练与仿真部署教程](docs/training_to_sim_deployment.md)。实际实现已拆分为 `workspace/training.py`、`inference.py`、`exporting.py`、`deployment.py`，每个阶段有中文注释；独立入口位于 `scripts/reinforcement_learning/rsl_rl/` 和 `scripts/deploy/`。环境、奖励和 PPO 的实际源码可运行 `bash g1.sh show-sources` 定位。

当前依赖的完整 Python 包源码已放入 [third_party/](third_party/README.md)：[mjlab 1.6.0](third_party/mjlab-v1.6.0/src/mjlab/) 和 [RSL-RL 5.4.2](third_party/rsl_rl-v5.4.2/rsl_rl/)，共 222 个 Python 文件，附许可证、版本来源与 SHA-256 清单。源码与官方标签及当前安装文件核对一致；训练继续使用原虚拟环境中的依赖。

`gym_rl/` 参考源码、文本配置、文档和许可证纳入本仓库；权重、网格、动作数据和运行中间文件保留在本地，不上传。目录说明见 [gym_rl 参考资料](gym_rl/README.md)。

面向 Unitree G1 29 自由度，先做仿真，保留真机配置。复用你已验证的 `~/mjlab-isolated-v160/.venv`，目标版本固定为 mjlab 1.6.0；安装脚本只复制本工作空间，不安装依赖、不修改旧项目。

## 常用脚本与完整中文讲解

常用操作已经整理在 `scripts/`，安装后可直接执行：

```bash
bash ~/g1-mjlab-workspace/scripts/03_resume.sh
bash ~/g1-mjlab-workspace/scripts/04_play.sh
bash ~/g1-mjlab-workspace/scripts/06_evaluate_forward.sh
bash ~/g1-mjlab-workspace/scripts/07_tensorboard.sh
```

这些是分别执行的操作：训练结束后再回放/评估，TensorBoard 可另开终端。旧安装先同步源码与脚本：

```bash
bash /mnt/c/Users/17599/Documents/Codex/g1-mjlab-workspace/apply_update.sh
```

阅读入口：[快捷脚本说明](scripts/README.md)、[训练流程与配置详解](docs/训练流程与配置详解.md)、[源码导读与注释索引](docs/源码导读与注释索引.md)。当前工作空间生产源码与脚本已补中文注释，第三方安装目录保持原样，profile 字节内容保留以兼容已有续训。

## 1. 在 Ubuntu 安装

```bash
bash /mnt/c/Users/17599/Documents/Codex/g1-mjlab-workspace/setup.sh
bash ~/g1-mjlab-workspace/g1.sh test
bash ~/g1-mjlab-workspace/g1.sh self-check
```

目标：`~/g1-mjlab-workspace`。若目标已存在，安装脚本会停止，保留原文件。默认解释器位置可以在第一次安装时通过 `MJLAB_RUNTIME=/你的/旧环境路径 bash .../setup.sh` 指定。

`doctor` 只读报告版本和资源；`self-check` 用 4 个环境走 32 步，不执行 PPO。通过只说明软件能运行。**稳定行走要看回放、失败终止次数和速度误差。**

文件、缓存、训练输出独立；Python 包复用旧环境，不形成第二份安装。CPU、显存和 WSL 内存仍然共享：不修改驱动、WSL 内存上限、全局环境或现有训练进程。默认 CPU 2 线程、进程 nice=10，无训练视频、无 W&B 上传。GPU 没有桌面响应性的硬隔离保证。

## 2. 测试适合的并行数

在已有训练自然结束后执行。不要与旧的训练同时测吞吐。

```bash
bash ~/g1-mjlab-workspace/g1.sh benchmark --envs 128 256 512 --iterations 30
```

依次短训，各自独立进程。丢弃前 5 个更新的 FPS，比较中位数；记录整块 GPU 的最小剩余显存及 WSL 最小可用内存。推荐条件：至少 1 GiB 显存和 2 GiB WSL 内存余量；吞吐相差不超过 5% 时选较小并行数。没有资源数据就不推荐。每轮开始前剩余显存不足 2 GiB 或 WSL 内存不足 3 GiB 时跳过；失败或测得余量过低时不再测更大规模。

这是短测和事后余量判断，不是显存配额；若卡顿可以 Ctrl+C 停止。结果为 `outputs/benchmarks/<时间>/results.json`，逐轮日志在同目录。首次 Warp 编译时间不用于 PPO 吞吐比较。跟踪动作任务更耗资源，需使用其自己的 profile 再测。测试模型只保存在 `outputs/benchmarks/runs`，不会混入正式训练的 `latest`。

### 计时修正与下一档测试

2026-10-23 的用户实测中，128/256/512 环境都完成了 30 次更新，吞吐中位数分别为 4630/8851/17486 步/秒，512 档最小剩余显存约 6.60 GiB、WSL 可用内存约 11.64 GiB。短测显示可探索 1024、2048，但尚未证明更高并行数或长时间训练稳定。

同一份日志的 128 档出现负的 `Perf/learning_time` 和 FPS。固定依赖 RSL-RL 5.4.2 使用 `time.time()` 测耗时，系统时间回拨可以造成这种现象；仅凭本日志不能确认回拨来源。现在训练入口仅在当前进程的 RSL-RL runner 模块内，把耗时计时替换为 `perf_counter()`，结束时恢复绑定，不修改标准库、依赖安装文件或系统时钟。TensorBoard 的时间戳保留真实日期时间。

旧安装可在没有本工作空间训练运行时执行下面的增量更新；它会先备份被替换的源码，不覆盖 profiles、动作数据、日志和旧模型，也不安装依赖：

```bash
bash /mnt/c/Users/17599/Documents/Codex/g1-mjlab-workspace/apply_update.sh
bash ~/g1-mjlab-workspace/g1.sh benchmark --envs 512 1024 2048 --iterations 100
```

新结果必须有 `timing_valid: true` 才参与推荐。即使中位数为正，存在负耗时、非有限值或不完整计时的测量也会被标记并排除。100 次比原来的 30 次更适合做初步比较，仍不代表长期温度和桌面响应性验证。

## 3. 平地速度跟踪：训练行走

```bash
bash ~/g1-mjlab-workspace/g1.sh train --profile velocity_flat --envs 256 --iterations 3000 --name walk_256
bash ~/g1-mjlab-workspace/g1.sh play --checkpoint latest
bash ~/g1-mjlab-workspace/g1.sh evaluate --checkpoint latest --envs 16 --seconds 60
```

`256` 是起始候选，按 benchmark 的结果替换。3000 次是一个观察节点，不是收敛承诺。需要继续就执行：

```bash
bash ~/g1-mjlab-workspace/g1.sh resume --checkpoint latest --envs 256 --iterations 3000 --name walk_continue
```

`--iterations` 始终表示本次新增的 PPO 更新次数。续训写入新运行目录，保留旧模型。改变并行数会改变每轮采样总数及优化行为，支持加载不表示学习曲线完全一致。每轮默认 24 步/环境，因此 128/256/512 环境对应 3072/6144/12288 条状态转移。

回放默认使用 Viser，在 Windows 浏览器打开终端显示的 `http://localhost:8080`；端口被占用时加 `--port 8081`。支持 `--viewer native`，需要 Ubuntu 图形环境。`latest` 表示最近运行目录中的最大数字模型编号，不代表表现最好；多个任务并用时请指定终端的 `RUN_DIR` 或具体模型。

评估会报告失败终止、超时、各终止原因、片段长度、平面速度 RMSE（m/s）及偏航角速度 RMSE（rad/s）。跟踪任务报告关节角 RMSE（rad）和对齐后身体位置误差。它使用官方 play 设置和固定种子，保留失败终止，失败后自动复位；记录输出到 `outputs/evaluations`。短片段很多时，即使回放画面一直在动，也不能认为已稳定行走。

## 4. 继续使用旧 g1_continue

下面的快捷名会在旧环境的 `outputs/rsl_rl/g1_velocity/*g1_continue*` 中选最近运行，再按数字选最新模型。

```bash
bash ~/g1-mjlab-workspace/g1.sh play --checkpoint g1_continue --legacy-velocity
bash ~/g1-mjlab-workspace/g1.sh resume --checkpoint g1_continue --legacy-velocity --envs 256 --iterations 3000 --name imported_continue
```

旧模型没有新工作空间的清单。`--legacy-velocity` 明确采用“同一 mjlab 1.6.0 官方平地 G1、默认观测和 PPO”的假设，加载时严格检查模型形状，但不能追溯验证过去改过的奖励、观测语义或物理参数。若旧训练确实改过这些配置，先对应迁移配置再续训。也可直接用 `--checkpoint /home/godot/.../某次运行/model_2199.pt` 指定你自己训练的文件。

## 5. 从 gym_rl 导入动作，训练 BeyondMimic 风格跟踪

```bash
bash ~/g1-mjlab-workspace/g1.sh import-reference
bash ~/g1-mjlab-workspace/g1.sh prepare-motion dance_102
bash ~/g1-mjlab-workspace/g1.sh prepare-motion gangnam_style
bash ~/g1-mjlab-workspace/g1.sh self-check --profile tracking --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz
bash ~/g1-mjlab-workspace/g1.sh benchmark --profile tracking --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz --envs 128 256 512 --iterations 30
bash ~/g1-mjlab-workspace/g1.sh train --profile tracking --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz --envs 256 --iterations 10000 --name dance_102
```

默认优先读取本工作空间的 `gym_rl/`，没有该目录时回退到 `/mnt/d/reference/reinforce/gym_rl`；也可加 `--source /参考目录`。只复制两份 CSV，来源位置和哈希保存在 `motions/reference_sources.json`。不执行 Isaac Lab 代码，也不直接使用资料中的旧策略。

转换在 CPU 上根据当前 mjlab 的 G1 做正运动学：CSV 根四元数 `xyzw` 转为 `wxyz`，60 Hz 重采样为当前控制频率，按名称映射 29 关节和所有刚体，计算世界系线/角速度。生成 `motions/processed/*.npz` 和说明清单。已有转换结果不覆盖。转换只验证运动学，无法保证动作没有穿地、滑脚、关节极限问题或具备动力学可行性；查看 JSON 中的 `joint_limit_violation_fraction`，再用跟踪评估判断可学程度。

`tracking_no_state_estimation` 提供官方去掉 base_lin_vel 和 motion_anchor_pos_b 的 actor 观测版本，应另起训练；不能拿完整观测策略直接切换。训练后 `play/evaluate/resume/export` 会从运行清单恢复动作路径和 profile，不需要每次手写动作参数。

## 6. 导出与仿真部署

```bash
bash ~/g1-mjlab-workspace/g1.sh export --checkpoint latest
# 把上一条打印的 DEPLOY_BUNDLE 路径放入 BUNDLE
BUNDLE="$HOME/g1-mjlab-workspace/outputs/deploy/替换为实际目录"
bash ~/g1-mjlab-workspace/g1.sh deploy-sim --bundle "$BUNDLE"
bash ~/g1-mjlab-workspace/g1.sh deploy-sim --bundle "$BUNDLE" --headless --seconds 60
```

导出包包含 TorchScript、ONNX、真实观测项/维度/顺序、动作关节名/scale/offset、PD 参数、SDK 映射、控制周期及校验哈希。跟踪策略附带 NPZ，ONNX 保留官方 `time_step` 和参考轨迹输出。

导出时用 32 组仿真观测比较训练策略与 TorchScript，数值不一致则报错；ONNX 目前只做结构检查，未承诺 ONNX Runtime 数值验证。`deploy-sim` 只加载导出的 TorchScript，通过同一 mjlab 后端做回归；**这不是独立引擎的 sim2sim，也不是上真机验证。**

`hardware_template.json` 默认禁用，保留状态机、控制周期和待实现的通信/状态估计/保护接口，关节映射和训练 PD 数据在 `manifest.json`。没有 Unitree SDK 网络控制代码，不会连接机器人。真实机器人上的观测构造、延迟、PD/力矩限幅和看门狗仍需后续集成验证。

## 配置与结果

- `profiles/velocity_flat.json`：官方平地基线，建议先用它。
- `profiles/velocity_gym_ppo.json`：单独的对比实验，仅把 entropy_coef 调为资料 robot_lab 的 0.008，保留 mjlab 的观测归一化及其余默认值，不保证更优。
- `profiles/tracking*.json`：官方 G1 动作跟踪和无状态估计观测版本。
- `outputs/runs/<时间_profile_名字>`：正式训练、参数快照、清单、资源采样及模型。
- `outputs/benchmarks`、`outputs/evaluations`、`outputs/deploy`：并行对比、评估、导出包。

调 profile 文件后，新实验可以使用，原实验续训会因配置哈希变化而拒绝，防止无记录地更换训练契约。训练参数完整快照在每次运行的 `params/`。不要在未对齐物理模型、单位和频率时直接复制 Isaac Lab 的奖励权重。

更多细节见 [适配说明](docs/reference_mapping.md)、[部署接口](docs/deployment_contract.md)、[验证范围](docs/validation.md)。

本工作空间依据固定版本接口实现：[mjlab v1.6.0](https://github.com/mujocolab/mjlab/tree/v1.6.0)、[G1 跟踪配置](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/tracking/config/g1/env_cfgs.py)、[RSL-RL v5.4.2](https://github.com/leggedrobotics/rsl_rl/tree/v5.4.2)。

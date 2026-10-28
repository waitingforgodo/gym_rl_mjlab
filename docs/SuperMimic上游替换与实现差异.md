# SuperMimic 上游替换与实现差异

2026-10-28：默认 SuperMimic 已改用 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)。按本次要求，只替换源码和入口，继续使用 `.runtime-path` 指定的原虚拟环境；未安装、卸载或升级训练依赖，没有新建训练环境。AMP、BeyondMimic 的运行环境、模型和参考动作保留。后续按要求移除了旧 SuperMimic 实现与全部相关遗留产物，当前需要重新训练。

这是第三方独立复现，**不是论文作者官方代码**，也不能把上游公布的分数当作本机已达到的效果。

当前本机默认在用户确认的视频 8GB 方案上，将环境数逐档提高到 **4096 环境、H=16、world/policy batch=512、eager**，配置为 `profiles/lafan1_4096_h16_eager.json`。完整 40 段数据、replay=1000、每次 world 更新 10 个 batch、学习率 2e-4、320000 总控制步保留；19 号脚本采用此配置并记录五项字段偏差。1024/2048/4096 环境均通过 50 步/4 次更新的限时真实短测，当前 4096 的 PyTorch 峰值约 4.41 GiB，整卡最低剩余显存约 2.07 GiB。短测不代表长期动作效果已验收。

按用户后续的长期要求，`profiles/` 现在只保留上述最高已验证配置，低规格、历史不稳定及重复路径配置已删除。所有本地 train/resume 默认自动评测选优，每个 run 只留一个可续训的 `best.pt`；保存点和结束时的其他权重自动清理，日志与评测记录保留，无需每次重复要求。

同日后续更新：训练默认已改为严格核对上游参数和 LAFAN1 40 段数据。新 `dance` 小批量训练在约 1400 控制步发生数值崩溃，已补本地数值检查和有限值原子保存；原算法文件未改。本机原始尺寸显存预检仅 replay/策略图就分配约 11.04 GiB，超过 7.96 GiB 物理显存，因此尚未启动完整原始训练。

此前按用户要求新增 `profiles/lafan1_1024_h16.json`：只改为 1024 环境、H=16，其余沿用上游。19 号脚本当时默认选此配置并记录偏差；通用 `g1.sh supermimic train` 仍保留严格原始规格。该设置的显存下界预检通过，replay/策略图峰值为 5263641600 字节（约 4.90 GiB），尚不含完整训练其他组件。旧 `dance` 的 5 份 NaN 权重已永久删除，约释放 1.00 GiB；后续校验又发现原计划保留的 500/1000 步权重、latest 链接及历史检查/评测 JSON 也已消失，未找到可恢复副本，额外变动来源未确认。当前该 run 只剩配置、清单和 TensorBoard 日志，需从新 run 开始训练。记录保存在本地 `outputs/supermimic/checks/20261028_invalid_dance_cleanup.json`。

## 1. 代码放在哪里、实际执行哪份

上游源码固定为提交 `ba6105349069bf040f1fdad60873d13138d505a2`，位于 [workspace/supermimic/upstream](../workspace/supermimic/upstream/)。保留 MIT 许可证、上游 README、设计文档、测试、数据处理程序和 `uv.lock`；[SOURCE_INFO.json](../workspace/supermimic/upstream/SOURCE_INFO.json) 逐文件记录 SHA-256。上游算法文件没有本地改写。

本地连接层按功能拆到 [workspace/supermimic/](../workspace/supermimic/README.md)：`configuration.py` 处理参数与数据，`checkpoints.py` 校验模型，`training.py` 连接训练器，`inference.py` 回放/评测，`deployment.py` 直接 checkpoint 仿真，`runtime.py` 和 `paths.py` 处理原环境与源码路径。实际执行的是上游 `Trainer.run()`、世界模型、策略和优化器，没有另外实现 learner。

```text
scripts/19_train_supermimic.sh
  → g1.sh → 原 .venv/bin/python
  → workspace/supermimic/cli.py
  → workspace/supermimic/training.py
  → workspace/supermimic/upstream/src/supermimic/training/loop.py
      → envs/                 真实 G1 仿真、目标、奖励、动作噪声
      → data/replay_buffer.py GPU 连续转移与终止状态
      → world_model/          多步动力学拟合
      → imagination/          BPTT actor、TD(λ) critic
```

这里只在运行进程中把上游 `src/` 加入导入路径，不改全局 `PYTHONPATH`，不执行 `uv sync` 或 `pip install`。原 `g1.sh` 和 `.runtime-path` 没有修改。上游 Git 元数据留在被忽略的 `.cache/upstream-git/`，目录作为普通源码纳入主仓库；上游演示视频不纳入，符合“只上传源码”的约定。

## 2. 与被替换的 structured v2 有什么区别

以下保留替换时的结构对照，供理解实现差异。对照中的旧稳定版及后续独立实验源码现已删除；后续 ±16 动作范围、去掉 actor 末端项等实验不属于该稳定基线。

| 项目 | 原 structured v2 | 现在的 SuperMimic-Reproduced |
| --- | --- | --- |
| 物理状态 | 100 维：71 维物理状态 + 上一动作 29 维；根角速度为世界系 | 71 维 MuJoCo 顺序，根角速度为根部局部系；上一动作在 replay/历史中单独保存 |
| 动力学输入 | 本体系特征、上一动作、当前动作、33 维 context，共 159 维 | heading 规范化状态 71 + 动作 29 + context 33，共 133 维 |
| 动力学输出 | 35 维有界速度增量 | 70 维：35 维速度增量 + 35 维位置/旋转/关节角修正 |
| 积分与先验 | 梯形积分，显式重力先验，`tanh` 限制增量、硬裁剪关节角 | 更新后速度做显式积分，再叠加学习的位置修正；SO(3) 姿态积分；零输出为匀速外推，没有旧增量/关节角裁剪 |
| 未来目标 | 10 × 71 = 710 维原始参考状态 | 10 × 68 = 680 维目标：相对默认关节角、关节速度、重力、根角速度、heading 系线速度、根高度；可选追加 9 维 anchor 观测 |
| 历史观测 | `[batch, time, 93]`，按时间展开 | 仍为 10 × 93 = 930，但按观测项分别堆叠历史后拼接，排列不同 |
| actor 输出 | 稳定基线 `3 × tanh`；后续实验单独放宽至 ±16 | decoder 直接输出 29 维关节设定点残差，不套旧版 `tanh` 上限；随后乘 G1 action scale、加默认姿态 |
| 动作噪声 | 旧适配器采集并区分 clean/executed 动作 | 独立 `NoisyJointPositionAction`：只在执行端加噪声，观测保留 clean action；世界模型拟合 executed action |
| 奖励/观测 head | 可微 FK 与解析任务 | 同样为可微 FK 与解析 head；身体对齐采用 torso anchor，不能把两版奖励实现视为逐项相同 |
| 终止 head | 参考高度/倾斜误差的固定 sigmoid，并处理动作末尾 | 小 MLP 学习终止概率，用真实 terminal 标签做 BCE；物理环境依然按高度/倾斜硬阈值终止，片段结束属于 timeout |
| critic | 有限范围 sigmoid value、真实单步 TD/Huber 校准 + 想象 GAE | 无旧 value 上下界；规范化状态 + 目标 + context + 4 维根误差；按想象 TD(λ) 目标拟合 |
| 展开安排 | 世界模型额外预热，H=8→16→32，预测误差门控 | 默认填 replay 40 控制步，随后直接 H=32；没有旧版的 50 次模型独训/阶段门控 |
| replay 与 reset 历史 | 原窗口要求完整连续历史，会排除 reset 后前 9 步 | 环境×时间 GPU 环形缓冲，单独写 terminal 状态；历史不足时补首个有效状态与零动作，可从启动状态采样 |
| 动作库与采样 | 现有流程主要针对一个 NPZ 的相位采样 | 多 clip 库、全局帧索引、逐片段 1 秒分桶；50% 均匀 + 40% 难度 + 10% 记忆；可选 fp16 紧凑库 |
| 优化与并行 | 本地 learner，actor 使用 microbatch 累积 | 上游 world/actor/critic 训练器，支持 `torch.compile`、torchrun 梯度 all-reduce；本地包装入口只验证单 GPU |
| 模型选择与评测 | 定期真实短回合评测并保存 `best.pt`，多种子存活指标 | 上游原生 latest/编号保存由本地连接层接管；全动作 Appendix F 评测选优，每个 run 仅留 `best.pt` |
| 模型/部署格式 | 自定义清单与 v1/v2 checkpoint，已有 JIT 导出包 | 上游 `wm/policy/critic/config` checkpoint；上游没有 JIT/ONNX 部署包导出和真机部署流程 |

网络主干规模相近：world hidden 为 `(2048, 2048, 2048)`，motion encoder 与 decoder 为 `(2048, 1024, 512, 256)`，latent=32，critic `(256,128)`。两者都采用模型学习与 BPTT，替换并不意味着改成 PPO 或引入 RSL-RL 训练。

当前源码： [state_spec.py](../workspace/supermimic/upstream/src/supermimic/world_model/state_spec.py)、[world_model.py](../workspace/supermimic/upstream/src/supermimic/world_model/world_model.py)、[policy_trainer.py](../workspace/supermimic/upstream/src/supermimic/imagination/policy_trainer.py)、[nets.py](../workspace/supermimic/upstream/src/supermimic/policy/nets.py)。

### 对以前训练问题的意义

旧模型把下一位置严格绑定到两端速度的梯形积分，还限制了速度变化与关节角。新模型的位置修正项提供额外自由度，能表示控制间隔内接触、加速等造成的位姿变化，不必完全依赖端点速度积分。新 replay 的启动历史补齐也覆盖了旧诊断发现的 reset 初期采样缺口。

这说明新版直接改变了两个值得关注的结构约束，**尚不能证明它们就是此前舞蹈失败的唯一原因**。新的终止分类器、无界 critic 和固定 H=32 也引入不同的学习行为，仍需看真实整段失败率、回报和模型多步误差，不能只看模型内回报上升。

## 3. 环境保留后，有哪些兼容处理

| 依赖 | 本机继续使用 | 上游 README/锁定环境 |
| --- | --- | --- |
| PyTorch | 2.9.1+cu128 | 2.14.1+cu130 |
| mjlab | 1.6.0 | Git 提交 `8bb2bc9e187797689ec82c78c35019afd2fd0376` |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0 | 3.16 系列 |
| Warp | 1.14.0 | 1.18 系列 |

当前实测能导入、创建训练环境、采集、优化、保存、续训和回放。评测遇到一处 API 差异：mjlab 1.6 没有 `RewardManager.term_rewards`，其 `_step_reward` 存的是奖励率。连接层仅为评测实例提供代理，乘控制步长 0.02 恢复上游“逐步奖励”的含义；没有修改安装文件或全局 RewardManager 类。实测各项奖励之和与回报反推的平均逐步奖励相差约 `3.6e-9`。

因此这是“上游源码 + 现有运行环境”的移植，不是上游锁定环境的逐位复现。早期真实训练短测使用 `--no-compile`，当时没有验证本机默认编译链路。

随后实际启动 `lafan1_1024_h16`，在第 42 控制步首次更新时触发 `BackendCompilerFailed`，完成更新数为 0。错误是 TF32 新旧 API 冲突：本机 mjlab 在 PyTorch 2.9 中写入 `fp32_precision`，而 2.9.1 的 TorchInductor 内部仍读取 `allow_tf32`。独立进程可直接复现同一错误；[PyTorch 2.9 文档](https://docs.pytorch.org/docs/2.9/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)也说明不能混用两种设置。

本地 `runtime.py` 现仅在 PyTorch 2.9 的 SuperMimic 进程中替换 mjlab 后端配置函数，统一使用 `allow_tf32`，保留 TF32 开关、cuDNN benchmark 与 deterministic 参数的原含义。上游源码和已安装文件均未修改，`torch.compile` 保持启用，manifest 记录 `backend_compatibility`。容量预检只覆盖 replay 和未编译的策略前向图，因此通过预检并不证明编译链路和完整更新成功。

修正后按 1024 环境、H=16、world batch=2048×10、policy batch=4096、replay=1000、完整 40 段数据和默认 compile 运行独立 43 步短测，完成 1 次 world/actor/critic 更新，checkpoint 经加载验证全部有限。首轮含编译耗时约 320.65 秒，最低显存余量 19 MiB。测试只将总预算缩短为 43 步，正式 profile 仍是 320000；它验证编译、反向传播、优化与保存链路，不证明长期速度、容量或动作效果。原失败 run 保留，重新正式训练须用新名称，例如 `lafan1_1024_h16_tf32`。

连接层另增加首次 world / policy 更新的开始与完成日志，避免把首次编译的长时间等待误判成退出；资源记录增加最后阶段及 PyTorch 显存分配/预留峰值。默认 50 步才打印的上游指标日志保持不变。

### 视频建议与 512 环境基线

已读取用户提供的[《SuperMimic 透彻讲解 But BY AI》](https://www.xiaohongshu.com/discovery/item/6ae11bd700000000230320e0)完整字幕，并检查 19:55、20:05、20:44、22:15、23:17 等画面。19:52–20:47 的核心建议是将并行环境数、batch 和想象长度缩小，并退回 eager。它是视频作者的消费卡实验配置，不是上游原始超参数，也不能用视频中的论文成绩作为本机结果。

| 设置 | 固定上游 | 视频方案的本机基线 |
| --- | --- | --- |
| 并行环境数 | 4096 | 512 |
| world batch | 2048 | 512 |
| policy batch | 4096 | 512 |
| 想象长度 H | 32 | 16 |
| world / imagination 编译 | default / default | null / null（eager） |

视频画面写的是“批大小同步 512”，没有列出 world/policy 的独立字段；本机把两者分别都设为 512，并在 manifest 明确记录。视频还展示了 320000 步、36 小时上限和每 2000 步保存。此次用户选择的是 batch/eager 的显存方案，正式 profile 继续保留原 320000 步、无墙钟上限、每 4000 步保存，不把测试的限时设置带入正式训练。

实测对照：只减到 512 环境而保留大 batch/compile，43 步完成 1 次更新，但 PyTorch 峰值分配约 7.66 GiB、预留 8.21 GiB，整卡最低余量仅 24 MiB。改用当前 batch=512/eager 后，沿用正式总预算与学习率，仅以 `max_hours=0` 在第一个 50 步日志点停止，完成 4 次更新；耗时 23.55 秒，PyTorch 峰值分配 2203998720 字节（约 2.05 GiB）、预留 2273312768 字节（约 2.12 GiB），整卡最低剩余 5033 MiB。保存的 world/policy 学习率约为 `1.9999998e-4`，网络与优化器有限。`status=time_limited` 是该检查主动设置的停止条件，正式 profile 没有该限制。

相同设置的独立复测耗时 26.19 秒，同样完成 50 步/4 次更新并以 CLI 退出码 0 结束，最低剩余显存 4992 MiB。配置、日志、资源与清理清单已归档到本地 `outputs/supermimic/checks/20261028_runtime_and_video_8gb/report.json`；5 份检查用权重经有限值验证后移除，释放约 1.00 GiB，原失败 run 的记录保留。正式长训尚未启动。

该 512 环境基线只保留在历史检查报告的 config 中；按“只保留最高可用配置”的要求，对应 profile 已删除，不再作为当前训练入口。

### 环境数与 batch 独立设置，以及提高环境数的实测

上游 `ReplayBuffer` 按 `[env, time]` 存储；`_sample_from_mask` 将可采样位置展平，再有放回抽取 n 个位置。world trainer 用自己的 `batch_size` 抽取 8 步窗口，policy trainer 用自己的 `batch_size` 抽取起始状态，再展开 H 步。因此两类 batch 不要求等于、整除或固定比例于环境数。环境数主要影响真实采集量、仿真和 replay 占用；batch 与 H 决定学习器的计算图规模。

保持 world/policy batch=512、H=16、eager、replay=1000 和全部数据不变，沿用正式 320000 步预算及约 2e-4 学习率，各用检查专用 `max_hours=0` 在第 50 步日志点停止：

| 环境数 | PyTorch 峰值分配 | 整卡最低剩余显存 | 完成真实更新 | CLI 退出码 |
| --- | --- | --- | --- | --- |
| 512（此前独立复测） | 2.05 GiB | 4992 MiB | 4 | 0 |
| 1024 | 2.39 GiB | 4598 MiB | 4 | 0 |
| 2048 | 3.06 GiB | 3794 MiB | 4 | 0 |
| 4096 | 4.41 GiB | 2116 MiB | 4 | 0 |

当前最高验证到 4096，仍有约 2.07 GiB 显存余量，19 号脚本因此采用 `lafan1_4096_h16_eager.json`；现只保留此 profile，512/1024 等旧配置已删除，数值记录仍在历史报告。此时环境数已与上游相同，剩余偏差只有 H、两类 batch、两处 compile 共五项。当前环境数与旧大 batch/compile 的显存紧张是不同实验，不能混为同一规格。

UTD=0.4 在上游按每个向量化控制步触发更新，而非按单条新增 transition。环境翻倍、batch 与 UTD 不变时，每步真实数据翻倍，优化更新次数不变，单位新增数据对应的更新量减少。它可以增加并行采集量，不能据此保证总训练耗时减半或最终效果更好。正式长训的模型误差、整段动作成功率仍需单独验证。原始资源、配置、日志及检查权重清理记录在本地 `outputs/supermimic/checks/20261028_env_scaling_batch512/report.json`。

当前正式训练入口：

```bash
bash scripts/19_train_supermimic.sh --name lafan1_4096_h16_eager
```

## 4. 训练与仿真命令

先准备固定版本的 LAFAN1 40 段数据，再校验实际参数与版本：

```bash
bash g1.sh supermimic prepare
bash scripts/19_train_supermimic.sh --dry-run
bash g1.sh supermimic preflight
```

默认从上游 `TrainConfig` 直接构造：4096 环境、replay 1000、world batch 2048、每次 world 更新 10 个 batch、actor batch 4096、H=32、UTD=0.4、学习率 2e-4、320000 总控制步与原 compile 模式。只把数据目录映射到共享 `motions/processed/lafan1/`，训练前验证恰好 40 段动作及转换清单的逐文件哈希；输出目录为 `outputs/supermimic/runs/<name>/`。

严格入口逐字段比较合并后的配置和上游默认值，拒绝缩小 batch/replay、筛选数据、关闭 compile 或缩短预算。直接构造上游 dataclass，不再维护重复的 `upstream_lafan1.json`。原始规格命令：

```bash
bash scripts/19_train_supermimic.sh --strict-upstream --name lafan1_upstream
```

2026-10-28 在 RTX 5070 Laptop 实际测试原始 replay（4096×1000）及原始策略 batch=4096、H=32 的前向 autograd 图：峰值分配 11854662656 字节，约 11.04 GiB；物理显存为 8546484224 字节，约 7.96 GiB。该下界还不包含仿真、世界模型、critic 和优化器。驱动允许超额分配，但不能据此宣布完整训练可在物理显存内运行；入口明确停止并保存容量报告，不自动降低参数。报告在本地 `outputs/supermimic/checks/20261028_131543_808506_capacity.json`。

历史 `local_8gb.json` 已删除，其 `dance` 长训曾在 1400 步首次记录 NaN，1500 步及以后的网络和优化器参数非有限，不能视作稳定配置。500/1000 步模型虽然参数有限，整段动作评测仍在约 1.10/1.28 秒失败。当前保留诊断记录；旧权重清理及额外缺失情况见本文开头。

**迭代单位已经变化。** 新 `--iterations 3000` 为总共 3000 次向量化环境控制步，每个环境约采集 60 秒仿真数据；40 步 warmup 后按 UTD=0.4，约进行 1184 次 world/actor 交替更新。旧版 3000 指优化更新编号；若希望约 3000 次新更新，目标控制步约为 7540。续训时也填目标总控制步数。

```bash
# 严格续训要求原 checkpoint 本身也是上游规格，且目标总步数尚未达到。
bash g1.sh supermimic resume \
  --checkpoint outputs/supermimic/runs/lafan1_upstream/best.pt \
  --name lafan1_upstream_continue

# 训练结束后网页回放，终端打印可点击地址。
bash scripts/20_play_supermimic.sh --checkpoint latest --port 8080

# 每个完整片段从首帧开始，失败即记录；不会靠反复 reset 计成成功。
bash g1.sh --python scripts/supermimic/evaluate.py --checkpoint latest

# 看曲线。
bash scripts/07_tensorboard.sh
```

`latest` 只查新实现的目录；正式比较应填写明确 checkpoint。`play --headless --steps 500` 是允许自动 reset 的链路检查，质量结论应使用 `evaluate.py` 的首回合报告。训练目录保存实际 `config.json`、源码提交/动作哈希/依赖版本的 `run_manifest.json`、`tb/` 及权重。相同模型不能无提示更换参考动作或顺序后续训。

需要自定义实验时，`--config 路径.json` 支持部分 `TrainConfig` 的递归覆盖，未知字段报错，并要求显式 `--allow-config-changes`。默认严格模式只容许运行名、输出目录、设备和匹配数据的目录映射不同。上游原生 CLI 和工具也已保留，可在同一解释器的 Python 程序中先调用连接层 `initialize()` 再运行，详见其原 README。

本地 `safety.py` 在动作执行、优化器步进前、每轮更新后与保存前检查有限值；非有限梯度/损失/网络或优化器状态会立即终止，写出 `resources.json`。有效更新的参数结果与未加防护的优化器逐值一致。保存先验证有限性，写临时文件后原子替换目标，避免坏状态或半写文件覆盖最后正常 checkpoint；加载及推理动作也拒绝 NaN/Inf。它不改变上游奖励、损失、学习率或采样公式，也没有把旧版有界 critic 等算法修改混入上游源码。

### 自动选出最佳并清理中间权重

`retention.py` 接管本地所有 train/resume 的保存调用，不改固定上游文件。每 4000 控制步保存点、正常结束、限时结束及正常 Ctrl+C 保存时，先保存一份有限值候选，再在独立进程中对该 run 全部动作按 Appendix F 协议评测：每段从首帧开始一次，关闭噪声与域随机化，固定种子。依次比较整段成功率、平均动作完成比例、平均归一化回报；相同分数保留较早最佳。同一步的 latest/编号/最终保存合并，不重复评测。

若候选更好，将完整网络、优化器、配置、步数和评测证据写入同一个临时 checkpoint，原子替换为 `best.pt`。确认有有效最佳后，只清理当前 run 的 `.candidate.pt`、`ckpt_latest.pt` 和编号权重；不删除其他 run、其他算法模型或任意命名文件。最终每个 run 只有一个可续训的 `best.pt`，`best.json` 是可重建索引，`selection.jsonl` 与 `selection/` 保留决策及全片段评测记录。

续训先继承源最佳，后续候选变差时不覆盖它。独立评测进程不改变训练进程的随机数、环境或采样器；候选评测失败时停止训练，保留旧最佳及一份待查候选，错误写入 `selection_error.json`，不会先删除旧最佳。`latest` 和 run 目录优先解析保留的 `best.pt`，兼容读取历史 latest；不再将“最后一步”当成最佳效果。

参数严格对齐与依赖锁定分开记录：继续复用原环境，manifest 的 `runtime_lock_differences` 列出 PyTorch/MuJoCo/Warp 与 `uv.lock` 的差异，`upstream_lock_environment=false`。mjlab 的版本号相同也不能证明安装来源等于上游 Git 提交，因此仍记录预期源提交而不宣称完整环境复现。

## 5. 清理后的仓库状态

2026-10-28 已按要求移除原 v1/v2 实现、专用 profile、兼容入口、旧诊断与消融工具、对应测试和三份旧诊断/实现文档。旧训练记录、checkpoint、JIT 包、评测图表、源码快照及接入时的临时短测模型已清空，共清理约 10.51 GiB 运行产物。

当前统一入口为 `bash g1.sh supermimic train|resume|play|evaluate|deploy`，19/20 号与原独立 Python 程序保留转发。旧导出/部署入口 21/22 号已删除。上游尚未提供 JIT/ONNX 部署包导出和真机部署，`deploy --checkpoint` 直接运行训练模型；与 AMP/BeyondMimic 的 `deploy --bundle` 区别见 [统一接口](../workspace/README.md)。

`latest` 现在优先选择最近产出保留最佳文件的 run，并读取其 `best.pt`；只在历史 run 无 best 时兼容读取旧 latest。旧 `dance` 当前没有可用 checkpoint。严格原始规格因本机容量限制尚未生成训练模型；短测只能验证软件链路。

AMP/BeyondMimic 的正式模型、基准模型、部署包和参考动作保留。共享环境配方移除废弃分支后恢复为这些模型记录的原源码哈希，未降低续训/部署检查要求。上游来源、许可证、逐文件源码校验与论文 PDF 继续保留。

## 6. 验证范围

接入阶段曾以小 batch/replay 验证真实采集、world/actor/critic 更新、保存、续训、回放与整段 benchmark；后续长训暴露 NaN，不能以当时的容量短测证明稳定性。本次数值防护测试包含有效更新逐参数一致、坏梯度禁止更新、坏优化器/权重拒绝加载和有限值原子保存；完整上游源码哈希仍保持不变。

严格配置 dry-run 校验通过，原始尺寸容量检查明确拒绝本机超额显存请求。完整 batch 长训、原始 compile 的端到端训练和论文数值仍未验证。下一步需要可容纳原始规格的 GPU；独立 benchmark 继续作为策略效果验收依据。

本次全仓库回归 86 项通过，含 10 项新严格配置/数值防护测试；固定上游 72 个文件哈希均保持不变。实际对 NaN latest 的加载在仿真前被拒绝；500 步有限模型加防护后的整段评测仍于第 55 步失败、归一化回报 0.519438，与此前结果一致。

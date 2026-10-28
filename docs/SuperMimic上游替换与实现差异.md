# SuperMimic 上游替换与实现差异

2026-10-28：默认 SuperMimic 已改用 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)。按本次要求，只替换源码和入口，继续使用 `.runtime-path` 指定的原虚拟环境；未安装、卸载或升级训练依赖，没有新建训练环境。AMP、BeyondMimic 的运行环境、模型和参考动作保留。后续按要求移除了旧 SuperMimic 实现与全部相关遗留产物，当前需要重新训练。

这是第三方独立复现，**不是论文作者官方代码**，也不能把上游公布的分数当作本机已达到的效果。

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
| 模型选择与评测 | 定期真实短回合评测并保存 `best.pt`，多种子存活指标 | 默认定期 `ckpt_latest.pt` / `ckpt_N.pt`，没有旧版自动 best 选择；独立 Appendix F 整段首回合 benchmark |
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

因此这是“上游源码 + 现有运行环境”的移植，不是上游锁定环境的逐位复现。默认编译模式在配置中保留；本次真实训练短测使用 `--no-compile`，不声称已经验证本机完整编译性能。

## 4. 训练与仿真命令

先检查数据、实际参数与版本，不创建仿真：

```bash
bash scripts/19_train_supermimic.sh --iterations 3000 --dry-run
```

未覆盖的训练参数直接取上游 `TrainConfig`：4096 环境、replay 1000、world batch 2048、每次 world 更新 10 个 batch、actor batch 4096、H=32、UTD=0.4、学习率 2e-4。默认动作改接本地 `motions/processed/amp_dance.npz`，输出目录独立为 `outputs/supermimic/runs/<name>/`。

本机 8 GB 显存已通过的 **4096 环境工程短测规格** 如下；保留完整网络宽度和 H=32，减小 batch/replay。延长至 3000 控制步可用：

```bash
bash scripts/19_train_supermimic.sh \
  --iterations 3000 --name reproduced_dance_3000 \
  --replay-capacity 64 --world-batch-size 256 \
  --world-batches-per-update 1 --policy-batch-size 128 \
  --horizon 32 --no-compile --save-interval 500
```

这套工程规格改变了上游训练批量和 replay 覆盖范围；它不是论文全量训练参数，46 步容量检查也不能保证长训收敛。尚未验证本机 8 GB 上原始全 batch 的训练容量，上游没有旧版 actor microbatch 选项。

**迭代单位已经变化。** 新 `--iterations 3000` 为总共 3000 次向量化环境控制步，每个环境约采集 60 秒仿真数据；40 步 warmup 后按 UTD=0.4，约进行 1184 次 world/actor 交替更新。旧版 3000 指优化更新编号；若希望约 3000 次新更新，目标控制步约为 7540。续训时也填目标总控制步数。

```bash
# 续训先恢复 checkpoint 中配置，再应用显式命令行覆盖；新实验另开目录。
bash scripts/19_train_supermimic.sh \
  --resume outputs/supermimic/runs/reproduced_dance_3000/ckpt_latest.pt \
  --iterations 7540 --name reproduced_dance_continue

# 训练结束后网页回放，终端打印可点击地址。
bash scripts/20_play_supermimic.sh --checkpoint latest --port 8080

# 每个完整片段从首帧开始，失败即记录；不会靠反复 reset 计成成功。
bash g1.sh --python scripts/supermimic/evaluate.py --checkpoint latest

# 看曲线。
bash scripts/07_tensorboard.sh --logdir "$PWD/outputs/supermimic/runs"
```

`latest` 只查新实现的目录；正式比较应填写明确 checkpoint。`play --headless --steps 500` 是允许自动 reset 的链路检查，质量结论应使用 `evaluate.py` 的首回合报告。训练目录保存实际 `config.json`、源码提交/动作哈希/依赖版本的 `run_manifest.json`、`tb/` 及权重。相同模型不能无提示更换参考动作或顺序后续训。

需要上游其余参数时，`--config 路径.json` 支持部分 `TrainConfig` 的递归覆盖，未知字段报错；配置直接对应上游 TrainConfig。上游原生 CLI 和工具也已保留，可在同一解释器的 Python 程序中先调用连接层 `initialize()` 再运行，详见其原 README。

## 5. 清理后的仓库状态

2026-10-28 已按要求移除原 v1/v2 实现、专用 profile、兼容入口、旧诊断与消融工具、对应测试和三份旧诊断/实现文档。旧训练记录、checkpoint、JIT 包、评测图表、源码快照及接入时的临时短测模型已清空，共清理约 10.51 GiB 运行产物。

当前统一入口为 `bash g1.sh supermimic train|resume|play|evaluate|deploy`，19/20 号与原独立 Python 程序保留转发。旧导出/部署入口 21/22 号已删除。上游尚未提供 JIT/ONNX 部署包导出和真机部署，`deploy --checkpoint` 直接运行训练模型；与 AMP/BeyondMimic 的 `deploy --bundle` 区别见 [统一接口](../workspace/README.md)。

`latest` 在尚未重新训练时会明确提示没有模型。上面的续训、回放和评测示例都须先有新训练产物；目前没有已训练好的新舞蹈策略。此前短测只验证软件链路，不能视作训练效果验收。

AMP/BeyondMimic 的正式模型、基准模型、部署包和参考动作保留。共享环境配方移除废弃分支后恢复为这些模型记录的原源码哈希，未降低续训/部署检查要求。上游来源、许可证、逐文件源码校验与论文 PDF 继续保留。

## 6. 验证范围

接入阶段已验证真实采集、world/actor/critic 优化、保存、续训、无界面回放及整段 benchmark；4096 环境以第 4 节的小 batch/replay 规格通过短测。保留现有环境的移植仍不等于上游依赖锁定下的逐位复现，也未验证完整 batch 的长训或论文数值。

清理后 `bash g1.sh test` 的 55 项回归测试通过，`bash scripts/19_train_supermimic.sh --dry-run` 的数据和实际配置检查通过。相关产物已清空，当前入口检查没有重新生成 checkpoint。后续训练成功后再用独立 benchmark 检查整段动作的真实失败率与回报。

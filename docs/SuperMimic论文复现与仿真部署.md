# SuperMimic 论文复现与仿真部署

本目录实现的是论文 [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) 的**独立复现**，不是作者官方源码。截至 2026-10-27，[官方项目页](https://supermimic.github.io/)和论文页没有提供训练仓库链接，因此不能宣称代码逐行一致或已复现论文数值。

论文 PDF 已固定到 [supermimic_2610.28480v1.pdf](papers/supermimic_2610.28480v1.pdf)，SHA-256：`fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`。

## 1. 实现范围

实现了论文的主要训练闭环：真实 MuJoCo/Warp 环境采样、确定性可微世界模型、每轮重新拟合模型、通过冻结世界模型做 32 步 BPTT 策略优化、带 soft survival 的终止预测、末端 critic、GAE critic 更新、序列回放、checkpoint、TensorBoard、回放、TorchScript 导出和仅加载 JIT 的 mjlab 仿真部署。

没有实现或没有条件验证的部分：BONES-SEED 129514 段/264.56 小时全量数据、多 GPU data parallel、H100 三小时训练对比、真机通信与安全状态机。仓库默认的单个 `dance_102.npz` 只适合验证链路和单动作跟踪，不能代表论文的大规模通用动作结果。

## 2. 论文参数与本实现

| 项目 | 当前值 | 位置 |
| --- | ---: | --- |
| 并行环境 | 4096 | `profiles/supermimic_4096.json` |
| 控制频率 | 50 Hz；4 × 5 ms 物理步 | mjlab tracking 配置 |
| 本体观测 | 93：29 关节位置、29 速度、3 角速度、3 投影重力、29 上一步动作 | `adapter.py` |
| 历史/未来目标 | 10 / 10 帧；每帧 71 维完整广义状态 | `config.py`、`adapter.py` |
| motion latent | 32，tanh | `networks.py` |
| 世界模型 | 2048-2048-2048 确定性残差 MLP | `networks.py` |
| 世界模型窗口 | 8；每次 10 × 2048 windows | `learner.py` |
| 想象 horizon / batch | 32 / 4096 | `learner.py` |
| critic | 256-128；5 epochs × 4 minibatches | `learner.py` |
| 优化 | Adam 2e-4；actor β=(0.9,0.95)；全局 clip 1、逐元素 clip 0.1 | `config.py` |
| 折扣 / GAE | 0.99 / 0.95 | `config.py` |
| 探索噪声 | action Gaussian σ=0.1 | `training.py` |
| replay / UTD | 每环境 1000 步 / 0.4 | `replay.py`、`training.py` |
| 学习率 | 25 万次余弦降至 3%，之后保持 | `learner.py` |
| 总预算 | 32 万次采集循环 | profile |

环境沿用 BeyondMimic tracking，但按论文移除外力 push、自碰撞奖励和末端高度终止，将 joint-limit 权重改为 -2；根高度阈值 0.25 m、姿态阈值 0.8 保持。摩擦 `[0.3,1.2]`、编码器偏置 `[-0.01,0.01]` 和躯干 COM 随机化均保留。

reset 将动作时间线划为 1 秒 bin，按论文混合 50% 均匀、40% `softmax(-25 G/Gmax)` 和 10% 历史 softmax 权重；`Gmax=5×0.02/(1-0.99)=10`。论文没有公布 running-return 更新率，本实现明确采用 EMA=0.01。UTD=0.4 按每次世界模型/策略更新平均采集 2.5 个真实控制步实现，2/3 步交替且每次更新总计使用 10×2048 个 8-step windows。

本地 mjlab 的 tracking command 到动作末尾时会直接抽取新帧并传送机器人。为避免世界模型把这个传送学成物理转移，SuperMimic recipe 增加 `motion_done`：在最后一帧前正常终止，由手动 reset 形成明确的 replay boundary。

论文未公开的细节被明确隔离：world state 和“完整参考帧”的数值布局都采用 71 维广义状态（局部 root 位置、wxyz 姿态、世界线/角速度、29 关节位置和速度）；MLP 激活采用 ELU；learned reward/observation/termination heads 使用 256-128；return running estimate 的 EMA 为 0.01。它们是工程选择，不是论文原值。context 直接读取 29 个编码器偏置、当前 torso COM 参数和脚部切向摩擦，共 33 维。世界模型拟合无噪声观测，32 步 imagined rollout 会在模型外重新采样论文规定的传感噪声。

## 3. 文件结构

```text
profiles/supermimic_4096.json       论文超参数与 4096 环境配置
workspace/supermimic/
  adapter.py                        mjlab → 论文张量，观测噪声、history、future goal
  replay.py                         不跨 episode reset 的 CPU 序列回放
  networks.py                       世界模型、目标编码器、actor、critic
  learner.py                        8 步 WM、32 步 BPTT、soft survival、GAE
  training.py                       真实采集/手动 reset/交替更新/checkpoint
  inference.py                      checkpoint 回放
  exporting.py                      JIT 导出与数值一致性检查
  deployment.py                     不加载训练 checkpoint 的仿真部署
scripts/supermimic/                 四个独立 Python 程序
scripts/19_*.sh ... 22_*.sh         对应快捷入口
tests/test_supermimic.py            CPU 算法回归
```

## 4. 训练

先确认参考动作存在：

```bash
ls motions/processed/dance_102.npz
bash scripts/19_train_supermimic.sh
```

更换动作或做短链路检查：

```bash
bash scripts/19_train_supermimic.sh \
  --motion "$PWD/motions/processed/gangnam_style.npz" \
  --iterations 100 --name supermimic_smoke
```

训练输出位于 `outputs/supermimic/runs/<时间_名称_4096>/`，包括 `run_manifest.json`、环境/算法配置、TensorBoard event、资源记录及 `supermimic_<迭代>.pt`。checkpoint 保存世界模型、actor、critic 和三个优化器；为避免数 GB checkpoint，默认不保存完整 replay，所以续训会重新积累 warmup 数据，不是逐位连续重放。

论文使用 H100。当前机器此前的 4096 测试针对 PPO/AMP，并不能证明这组 2048 宽世界模型和 32 步 BPTT 能放入同一显存；若发生 OOM，应先降低 `imagination_microbatch`（不改变总 batch），世界模型 batch 仍需进一步实现分块梯度累积才能保持论文总 batch。

## 5. checkpoint 回放

```bash
bash scripts/20_play_supermimic.sh --checkpoint latest --port 8080
```

终端会打印 `http://localhost:8080`。也可把 `--checkpoint` 指向明确的 `supermimic_N.pt`，避免 `latest` 选错实验。

## 6. 导出与仿真部署

导出只保留 actor；世界模型和 critic 不参与部署：

```bash
bash scripts/21_export_supermimic.sh --checkpoint latest
```

复制终端打印的 `DEPLOY_BUNDLE`，再执行：

```bash
bash scripts/22_deploy_supermimic_sim.sh --bundle /完整/DEPLOY_BUNDLE/路径 --port 8080

# 无窗口闭环 60 秒，只验证 JIT、观测和 mjlab 接口
bash scripts/22_deploy_supermimic_sim.sh --bundle /完整/DEPLOY_BUNDLE/路径 \
  --headless --seconds 60
```

部署程序校验 `hashes.json`，只读取 `policy_jit.pt`、`motion.npz` 和 manifest，不读取训练 checkpoint，也不创建 optimizer。这里的“部署”仅表示同一 mjlab 后端中的导出策略闭环回放，不是 sim2sim 或真机部署。

## 7. 验证边界

`tests/test_supermimic.py` 会实际执行小网络更新，检查序列不跨 reset、世界模型参数变化、actor 通过固定世界模型得到梯度、critic 更新和 checkpoint 往返一致。它不证明 32 万次训练收敛。质量验收至少还需记录动作跟踪误差、失败率、回报、世界模型多步误差和不同动作留出集表现。

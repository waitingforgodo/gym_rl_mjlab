# G1 AMP 舞蹈训练与仿真部署

当前工作空间新增独立的 **AMP（Adversarial Motion Priors，对抗动作先验）** 实现，复用 mjlab 1.6.0 和 RSL-RL 5.4.2，无需安装 Isaac Lab/skrl。原有 BeyondMimic `tracking*` 继续负责参考动作跟踪。

当前快捷脚本使用 **4096 环境**的 `amp_dance_4096`。重复的 2048 源 profile 已删除；已完成模型的运行清单嵌入了原始配置，仍可回放和导出。4096 的实际容量结果见 [4096 并行配置与容量结果](4096并行配置与容量结果.md)。

AMP 学习专家动作的连续状态分布。actor 不接收参考目标、时间进度或动作帧号，因此这里学习的是舞蹈风格，不保证按原顺序和节拍复现完整编舞。需要严格复现指定片段时，使用 BeyondMimic 跟踪任务。

## 1. 源码和配置在哪里

| 文件 | 作用 |
| --- | --- |
| [amp/tasks.py](../workspace/amp/tasks.py) | 独立任务、观测、奖励/终止、PPO 和判别器默认参数 |
| [amp/features.py](../workspace/amp/features.py) | 专家/仿真共享的状态特征及连续两帧排列 |
| [amp/reference.py](../workspace/amp/reference.py) | 专家加载、连续帧采样、固定专家统计归一化 |
| [amp/environment.py](../workspace/amp/environment.py) | 参考状态初始化和自动复位前的终止帧捕获 |
| [amp/algorithm.py](../workspace/amp/algorithm.py) | PPO 扩展、判别器、梯度惩罚、风格奖励、历史回放和完整续训 |
| [amp/conversion.py](../workspace/amp/conversion.py) | robot_lab AMP 数据到当前 G1 模型的转换 |
| [amp_dance_4096.json](../workspace/amp/profiles/amp_dance_4096.json) | 当前默认 4096 配置及实测参数说明 |
| [amp_dance.json](../workspace/amp/profiles/amp_dance.json) | 保留的 256 配置，兼容已有 AMP 模型 |
| [prepare_amp_motion.py](../scripts/tools/prepare_amp_motion.py) | 独立数据准备程序 |
| [17_prepare_amp_motion.sh](../scripts/17_prepare_amp_motion.sh) | CPU 数据转换入口 |
| [18_train_amp_dance.sh](../scripts/18_train_amp_dance.sh) | AMP 舞蹈训练入口 |
| [test_amp.py](../tests/test_amp.py) | 真正的 PPO/判别器更新、复位边界、特征、回放及续训测试 |

[runtime.py](../workspace/common/runtime.py) 在当前进程注册 `Workspace-AMP-Dance-Flat-Unitree-G1`，沿用已有训练、回放、导出和部署程序。第三方源码副本及虚拟环境安装文件保持原样。

## 2. 默认动作与数据转换

默认动作是本地 `gym_rl/robot_lab-main/.../g1_amp/motions/g1_dance1_subject2_30.npz`：30 Hz、301 帧、约 10 秒的舞蹈。资料存在重复目录时按路径排序选择，实际来源和 SHA-256 写入转换报告。

原数据有 29 关节、14 刚体，字段为 `dof_positions/body_positions/...`。转换只按名称取 29 关节和 `pelvis` 根位置、wxyz 姿态，再用当前 mjlab G1 做 CPU 正运动学，重新计算全部刚体和世界系速度。Isaac 的刚体下标与手部名称不直接沿用。

```text
motions/raw/amp_dance.csv           # 本地转换中间文件
motions/processed/amp_dance.npz     # 50 Hz、501 帧、10 秒的标准专家轨迹
motions/processed/amp_dance.json    # 来源、哈希、名称、关节越限比例
```

相同来源、相同哈希的结果可重复校验后复用，不覆盖不同来源数据。运动学转换通过不等于接触和动力学质量已验收。动作数据、模型、缓存和训练输出继续被 Git 忽略。

## 3. AMP 如何学习

每帧 83 维：29 关节角、29 关节速度、根高度 1、去航向姿态 6、去航向根线/角速度各 3、四个末端相对根的位置 12。末端为左右脚踝和左右腕部。移除世界 xy 与全局航向，避免判别器根据并行环境的位置判断真假。

判别器接收 `[当前帧, 下一帧]`，共 166 维。专家也取相隔 0.02 秒的真实相邻帧，不把首尾强行连成循环。策略下一帧在物理积分后、自动复位前捕获，额外 `sim.forward()` 对齐关节和刚体状态，避免把跌倒前状态与复位后姿态错误拼接。

训练每轮执行：

1. actor 根据本体状态输出动作，由 mjlab 推进物理。
2. 用本轮固定判别器计算风格奖励，与辅助奖励合并后，交给官方 PPO 计算 GAE/return。
3. 执行 PPO 更新，再训练判别器；专家为正样本，本轮和历史策略为负样本。
4. 新策略转移进入有界回放池，供后续判别更新采样。

判别器使用 BCE、专家输入的 R1 梯度惩罚、输出层 logit 正则与 Adam 权重衰减。风格奖励为稳定计算的 `softplus(logit)`，对应 `-log(1-sigmoid(logit))`：

```text
combined_reward = task_reward + 2.0 * 0.02 * clamp(softplus(logit), max=10)
```

`0.02` 为策略周期，与 mjlab 按 dt 积分的辅助奖励统一尺度。失败终止当步风格奖励置零，超时保留正常风格奖励与 PPO time-out bootstrap。辅助项只有存活、失败、动作变化和关节极限，没有逐帧模仿误差。

实现形式参考 [skrl AMP 官方算法说明](https://skrl.readthedocs.io/en/develop/api/agents/amp.html)，数据和参数起点参考本地 [robot_lab AMP 配置](../gym_rl/robot_lab-main/source/robot_lab/robot_lab/tasks/direct/g1_amp/agents/skrl_dance_amp_cfg.yaml)。当前特征、控制周期和 PPO 配置属于本机 mjlab 适配，不声称逐项复现 skrl 或原论文实验。

## 4. 默认参数

| 参数 | 默认值与说明 |
| --- | --- |
| 并行环境、训练预算 | 4096 环境、30000 次更新；这是候选预算，不保证收敛 |
| 物理步、decimation、控制步 | 0.005 s、4、0.02 s，复用原 G1 模型及 PD |
| actor/critic 观测 | 各 96 维：根线/角速度、重力投影、相对关节角、关节速度、上一动作 |
| 策略网络 | 512/256/128、ELU、观测归一化，29 维动作；log 标准差初值 0.5 |
| PPO 采样与更新 | 24 步、5 epochs、4 minibatches；4096 环境时每轮 98304 样本，每 minibatch 24576 |
| PPO 初始学习率、熵系数 | 0.0003、0.01；学习率按 KL 自适应调整 |
| 判别器 | 166→256→128→1、ReLU，输出 logit |
| 判别器优化 | 独立 Adam，学习率 0.0001、batch 4096、每 PPO 轮更新 8 次 |
| 正则系数 | 梯度惩罚 5、logit 正则 0.05、weight decay 0.0001 |
| 回放池 | 400000 条转移，原始 float32 数据约 253.3 MiB |
| 保存间隔 | 500 次更新；保存 PPO、判别器、独立优化器、专家归一化统计、有效回放数据 |

首版支持单 GPU、单 NPZ 专家库。训练随机选择参考初态，回放从参考第 0 帧初始化；此后不再强制写入参考姿态。保留 startup 随机化，首版不加入随机推搡。PPO 的其他参数及判别器默认值分别见 profile 和 `amp/tasks.py`，最终生效配置保存在每次运行的 `params/agent.yaml`。

原 256 配置仍为判别器 batch=1024、每轮更新 4 次、回放 50000 条；2048 配置为 batch=4096、每轮 8 次、回放 400000。4096 profile 保持实际容量短测使用的 2048 判别器设置，只把环境数改为 4096，避免把未经测试的更大判别器批量混入本次切换。因此每轮 PPO rollout 翻倍，但判别采样占比和回放覆盖轮数减半；三种配置的优化行为不能视为完全等价。

## 5. 数据准备与训练

在仓库根目录执行：

```bash
# 1. CPU 转换，可重复执行
bash scripts/17_prepare_amp_motion.sh

# 2. 4 环境、32 步零动作的软件检查，不更新网络
bash g1.sh self-check --profile amp_dance_4096

# 3. 可选：独立 benchmark 目录中的短训检查
bash g1.sh benchmark --profile amp_dance_4096 --envs 4096 --iterations 30

# 4. 正式训练：默认 4096 环境、30000 更新
bash scripts/18_train_amp_dance.sh --seed 42
```

独立 Python 训练入口也可使用：

```bash
bash g1.sh --python scripts/reinforcement_learning/rsl_rl/train.py \
  --profile amp_dance_4096 --envs 4096 --iterations 30000 --name amp_dance --seed 42
```

先观察 100 次更新可加 `--iterations 100`，但不能据此验收舞蹈质量。也可以将已有标准 NPZ 作为风格专家，分别新建实验：

```bash
bash scripts/18_train_amp_dance.sh \
  --motion "$PWD/motions/processed/dance_102.npz" --name amp_dance_102
```

`gangnam_style.npz` 转换后同样可通过 `--motion` 指定。采用相同动作数据做 AMP 风格学习和 BeyondMimic 跟踪，属于不同实验，模型不能直接混用。

旧 checkpoint 应直接交给回放或导出程序；程序读取同目录 `run_manifest.json` 中的嵌入 profile，不能用新 4096 profile 冒充旧训练配置。

## 6. 续训、回放、导出和仿真部署

用正式训练打印的 `RUN_DIR`，避免误选其他任务的模型：

```bash
AMP_RUN="$HOME/g1-mjlab-workspace/outputs/runs/替换为实际AMP运行目录"

# 续训自动继承 profile、动作、环境数、种子，写入新目录
bash scripts/03_resume.sh --checkpoint "$AMP_RUN" --iterations 5000 --name amp_continue

# 回放后打开终端打印的 http://localhost:8080
bash scripts/04_play.sh --checkpoint "$AMP_RUN"

# 评估失败、终止原因和存活时间
bash g1.sh evaluate --checkpoint "$AMP_RUN" --envs 16 --seconds 60

# JIT 数值对比、ONNX 结构检查
bash scripts/08_export.sh --checkpoint "$AMP_RUN"

# 填入上一步打印的 DEPLOY_BUNDLE
AMP_BUNDLE="$HOME/g1-mjlab-workspace/outputs/deploy/替换为实际AMP导出目录"
bash scripts/09_deploy_sim.sh --bundle "$AMP_BUNDLE"
```

AMP 全量续训恢复判别器、独立优化器、固定专家统计和回放池；清单还检查 profile、专家数据哈希和 AMP 源码哈希。该过程不恢复完整物理状态与全部 RNG 轨迹，因此不承诺逐位一致重放。

推理只运行 actor；JIT/ONNX 不包含判别器，也不需要 `time_step`。包内参考 NPZ 仅供当前仿真后端初始化使用。`deploy-sim` 仍是同一 mjlab 后端的导出回归。

AMP 没有同步参考进度，通用评估器的逐帧关节 RMSE 不适用于本任务，会返回空值。该评估器从底层环境统计的 `reward_per_step_mean` 仅包含辅助奖励；训练时的风格奖励通过 `AMP/*` 曲线查看。舞蹈质量需结合回放、失败/存活统计和风格表现，不能只看判别器奖励。

## 7. TensorBoard 曲线

使用 `bash scripts/07_tensorboard.sh`，选择 AMP 运行目录：

| 曲线 | 含义 |
| --- | --- |
| `AMP/style_reward` | 已乘风格系数和 dt 的每步风格奖励 |
| `AMP/task_reward` | 存活、失败、平滑、关节极限的辅助奖励 |
| `AMP/combined_reward` | PPO 实际学习的每步合成奖励 |
| `Loss/amp_discriminator` | BCE 分类损失，不含额外正则项 |
| `Loss/amp_gradient_penalty` | 专家输入梯度平方均值，总损失中再乘 5 |
| `Loss/amp_expert_accuracy` | 专家样本判为真的比例 |
| `Loss/amp_policy_accuracy` | 策略样本判为假的比例 |
| `Loss/amp_replay_size` | 有效历史转移数量，上限 400000；原 256 配置为 50000 |
| `Train/mean_reward` | 已完成片段的累计合成回报，受片段长度影响 |
| `Episode_Reward/*` | 环境辅助奖励分项，不含算法内加入的风格奖励 |
| `Episode_Termination/*` | 低高度、倾倒和超时等终止原因 |

准确率和回放计数也位于 `Loss/` 下，是沿用官方 logger 对 update 字典的命名方式。判别器准确率长期接近 100% 可能说明专家和策略仍容易区分；接近 50% 也不能单独证明舞蹈已学会。

实际软件验证范围见 [validation.md](validation.md)。短训策略用于接口检查，不代表舞蹈已经训练完成。

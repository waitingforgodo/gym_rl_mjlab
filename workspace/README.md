# 按方法组织的训练与仿真接口

统一用 `bash g1.sh <方法> <功能> [参数]`，继续使用 `.runtime-path` 中原有解释器、CUDA 和 mjlab。也可以在该解释器中执行 `python -m workspace.<方法> <功能>`；模块导入和 `--help` 不启动训练或仿真。

新接入的 **[instinct 工具链](instinct/README.md)** 提供官方 InstinctMJ 行走、BeyondMimic 和 Whole-body Shadowing，支持完整训练/导出/仿真流程；原四种方法继续用于原模型与各自专有算法。用 `bash g1.sh methods` 查看五种入口。它们共用原运行环境；Instinct 的机器人、PPO 后端和模型契约单独管理。

## 目录与职责

```text
workspace/
├── amp/                       AMP：PPO + 动作风格判别器
│   ├── cli.py                 本方法命令行
│   ├── training.py            train / resume
│   ├── inference.py           play / evaluate
│   ├── deployment.py          export / deploy
│   ├── data.py                prepare
│   ├── profiles/              AMP JSON 配置
│   └── algorithm.py, ...      判别器、replay、特征和任务原实现
├── beyondmimic/               BeyondMimic 风格的参考动作跟踪
│   ├── cli.py
│   ├── training.py
│   ├── inference.py
│   ├── deployment.py
│   ├── data.py
│   └── profiles/              tracking / 无状态估计 / robust
├── supermimic/                世界模型 + BPTT
│   ├── cli.py
│   ├── configuration.py       参数合并、数据与来源校验
│   ├── checkpoints.py         模型格式、动作契约、latest
│   ├── training.py            上游 Trainer 的训练/续训连接层
│   ├── inference.py           回放、整段 benchmark
│   ├── deployment.py          直接 checkpoint 仿真
│   ├── runtime.py, paths.py   原运行环境与源码路径
│   ├── strict.py, safety.py    上游参数/数据契约与数值防护
│   ├── capacity.py            原始尺寸的显存容量下界检查
│   ├── retention.py           独立整段评测选优、每个 run 仅留 best.pt
│   ├── profiles/              仅 lafan1_4096_h16_eager.json
│   └── upstream/              固定提交的完整 SuperMimic-Reproduced
├── velocity/                  速度指令 PPO，接口布局同上
│   └── profiles/
├── instinct/                  官方 InstinctMJ，独立机器人/模型契约
│   ├── cli.py                 doctor / setup / tasks / 全流程命令
│   ├── training.py            instinct_rl 原生 train / resume
│   ├── inference.py           play / evaluate
│   ├── deployment.py          JIT / ONNX 导出和独立部署
│   ├── data.py                pelvis → torso 根转换与 waist 校正
│   ├── profiles/              locomotion / beyondmimic / shadowing
│   └── upstream/              固定 InstinctMJ 与 instinct_rl 源码
├── common/                    共享 PPO 流程、配置、协议和工具
│   ├── method_cli.py          AMP/BeyondMimic/velocity 参数与分发
│   ├── methods.py             按方法定位模型与产物
│   ├── profiles.py            配置定位及方法判断
│   ├── workflows.py           方法接口复用的功能流程
│   ├── training.py            PPO 采样/训练/恢复的连接层
│   ├── inference.py           actor 加载与仿真评测
│   ├── exporting.py           JIT/ONNX 导出与校验
│   ├── deployment.py          导出包仿真闭环
│   └── runtime.py, ...        环境构造、动作转换、资源与计时
└── cli.py                     g1.sh 总入口
```

算法专有代码在对应方法目录，共享 PPO 流程只实现一份。BeyondMimic 环境和 PPO 本体继续使用已安装的 mjlab/RSL-RL，可用 `bash g1.sh show-sources --kind tracking` 定位；本地参考副本在 `third_party/`。SuperMimic 的实际算法源码在本方法的 `upstream/src/supermimic/`，不是另装一个 Python 环境。

根目录的 `contract.py`、`motion.py`、`runtime.py` 是小型导入兼容文件，实际实现均在 `common/`；保留它们是为了不改 AMP 原文件和既有调用路径。

## 功能对应关系

| 功能 | AMP | BeyondMimic | SuperMimic | velocity |
| --- | --- | --- | --- | --- |
| 准备数据 | `prepare` | `prepare dance_102` | 使用标准 NPZ；上游转换工具在 upstream/scripts/data | 不需要动作数据 |
| 从零训练 | `train` | `train` | `train` | `train` |
| 续训 | `resume --checkpoint ...` | 同左 | 同左，目标总控制步数 | 同左 |
| 网页回放 | `play --checkpoint ...` | 同左 | 同左 | 同左 |
| 定量评测 | `evaluate`，固定时间统计 | `evaluate`，固定时间统计 | `evaluate`，完整 clip 首回合 benchmark | `evaluate --velocity-cmd ...` |
| 导出 | `export`，JIT/ONNX 包 | 同左 | 上游暂未提供 | 同左 |
| 仿真部署 | `deploy --bundle ...` | 同左 | `deploy --checkpoint ...` | 同左 |

AMP/BeyondMimic/velocity 的 `--iterations` 是**本次新增 PPO 更新数**。SuperMimic 的 `--iterations` 是**目标总向量化控制步数**，含 warmup；默认 UTD=0.4，3000 步约有 1184 次模型/策略更新。这两个数字不能直接作为同等训练预算。

Instinct 同样使用新增 PPO 更新数，支持 `prepare / train / resume / play / evaluate / export / deploy`，还提供 `setup / doctor / tasks / summary`。动作数据、checkpoint 与部署包按自己的后端约定检查。其训练/部署命令见下节，源版本、任务差异和许可见 [接入说明](../docs/InstinctMJ接入与完整工具链.md)。

## InstinctMJ

```bash
bash g1.sh instinct doctor
# 首次拉取时缺附加依赖/资产才需要 setup，原核心环境不升级。
bash g1.sh instinct setup
bash g1.sh instinct prepare --source motions/processed/amp_dance.npz
bash g1.sh instinct train --profile beyondmimic --dry-run
bash g1.sh instinct train --profile beyondmimic --iterations 30000 --name g1_dance
bash g1.sh instinct resume --profile beyondmimic --checkpoint latest --iterations 1000
bash g1.sh instinct play --profile beyondmimic --checkpoint latest --port 8080
bash g1.sh instinct evaluate --profile beyondmimic --checkpoint latest --envs 16 --steps 500
bash g1.sh instinct export --profile beyondmimic --checkpoint latest
bash g1.sh instinct deploy --bundle /完整路径/部署包 --headless --steps 500
bash g1.sh instinct deploy --bundle /完整路径/部署包 --engine onnx --headless --steps 500
```

三份默认 profile 都为 4096 环境；行走选择 `locomotion`，不需要参考动作，Whole-body Shadowing 选择 `shadowing`。固定的 Instinct 上游许可为 CC BY-NC 4.0，感知/HOI/Parkour 等任务还需额外数据。详细操作、参数覆盖和已验证范围见 [Instinct 使用说明](instinct/README.md)。

## AMP

```bash
bash g1.sh amp prepare
bash g1.sh amp train --dry-run
bash g1.sh amp train --iterations 30000 --name dance
bash g1.sh amp resume --checkpoint latest --iterations 1000
bash g1.sh amp play --checkpoint latest --port 8080
bash g1.sh amp evaluate --checkpoint latest --envs 16 --seconds 10
bash g1.sh amp export --checkpoint latest
# 把 export 打印的 DEPLOY_BUNDLE 填到这里。
bash g1.sh amp deploy --bundle /完整路径/部署包 --headless --seconds 10
```

默认 `amp_dance_4096`、4096 环境、动作 `motions/processed/amp_dance.npz`。判别器只参与训练，导出包只包含 actor 及归一化器；仿真仍用参考数据初始化机器人。详情见 [AMP 教程](../docs/AMP舞蹈训练与仿真部署.md)。

## BeyondMimic

```bash
bash g1.sh beyondmimic prepare dance_102
bash g1.sh beyondmimic train --dry-run
bash g1.sh beyondmimic train --iterations 10000 --name dance_102
bash g1.sh beyondmimic train --profile tracking_no_state_estimation_4096 \
  --motion motions/processed/dance_102.npz
bash g1.sh beyondmimic resume --checkpoint latest --iterations 1000
bash g1.sh beyondmimic play --checkpoint latest --port 8080
bash g1.sh beyondmimic evaluate --checkpoint latest --seconds 10
bash g1.sh beyondmimic export --checkpoint latest
bash g1.sh beyondmimic deploy --bundle /完整路径/部署包 --headless --seconds 10
```

默认 `tracking_4096`、4096 环境、动作 `motions/processed/dance_102.npz`；配置名继续保留 tracking 前缀，已有模型的配置字节和哈希不变。策略以逐帧参考目标驱动，不能把 AMP 模型给这个入口使用。

## SuperMimic

当前本机训练入口保持 4096 环境、H=16、world/policy batch=512、eager；其余训练设置沿用上游，LAFAN1 40 段数据保持完整。使用 `bash scripts/19_train_supermimic.sh --dry-run` 检查、`bash scripts/19_train_supermimic.sh --name lafan1_4096_h16_eager` 开始新 run。只保留 `workspace/supermimic/profiles/lafan1_4096_h16_eager.json` 这一份最高已验证配置，低规格与重复配置已删除，五项字段偏差写入 manifest。4096 更新短测的 PyTorch 峰值约 4.41 GiB，完整长训仍未验收。

每次 SuperMimic train/resume 默认在保存点和结束时独立评测所有动作，依次比较整段成功率、完成比例、回报，每个 run 只保留可续训的 `best.pt`，中间权重自动删除。`latest`/run 目录优先解析该最佳文件，旧 latest 格式仍可读。日志、清理与评测记录保留；其他 run 和其他算法模型不受本 run 清理影响。评测失败时保护已有最佳并保留一份候选供排查。

下面保留通用上游严格入口；19 号脚本传 `--strict-upstream` 时也使用该规格：

```bash
bash g1.sh supermimic prepare
bash g1.sh supermimic train --dry-run
bash g1.sh supermimic preflight
bash g1.sh supermimic train --name lafan1_upstream
bash g1.sh supermimic play --checkpoint latest --port 8080
bash g1.sh supermimic evaluate --checkpoint latest
bash g1.sh supermimic deploy --checkpoint latest --headless --steps 500
```

通用严格入口对齐上游参数：4096 环境、replay 1000、world batch 2048×10、policy batch 4096、H=32、320000 总控制步及全部 40 段 LAFAN1。本机原始尺寸预检仅 replay/策略图就分配约 11.04 GiB，超过 7.96 GiB 物理显存，正式训练会被阻止。历史不稳定配置文件已删除；当前唯一配置显式记录所有偏差。训练/保存/加载遇 NaN/Inf 会拒绝继续，原上游算法源码不改。运行环境差异单独记录，尚未达到依赖锁定环境的完整复现。详情见 [上游实现说明](../docs/SuperMimic上游替换与实现差异.md)。

## 产物、旧模型和快捷脚本

新产物按方法保存：`outputs/<方法>/runs/`、`evaluations/`、`deploy/`；SuperMimic 当前没有导出包目录。原 `outputs/runs/`、`outputs/deploy/` 内模型保持原位置，方法入口兼容读取它们。`latest` 根据清单匹配本方法，不会因为 AMP 模型更新更晚就被 BeyondMimic 选中；正式对照仍应填写明确 checkpoint。

编号脚本继续可用：17/18 是 AMP，11/16 是 BeyondMimic，19/20 是 SuperMimic，02/15 是速度 PPO，21–25 是 Instinct；原独立 Python 入口保留转发。公共 doctor、test、benchmark、TensorBoard 仍在 `g1.sh` / `scripts/`。TensorBoard 默认只显示最近一次正式训练，打印选中的日志目录，历史日志保留。同目录服务会直接复用；端口被其他服务占用时自动换可用端口并打印实际地址。下一轮训练开始后重新运行脚本即可。需要查看某方法的历史对比时，可以显式指定根目录：

```bash
bash scripts/07_tensorboard.sh --logdir "$PWD/outputs/amp/runs"
bash g1.sh amp --help
bash g1.sh beyondmimic deploy --help
bash g1.sh supermimic train --help
bash g1.sh test
```

目录整理不改变机器人、奖励、学习参数或模型输入输出，不自动开始长训，也不移动既有模型。仿真短测用于验证接口；动作质量仍要用正式评测和回放确认。

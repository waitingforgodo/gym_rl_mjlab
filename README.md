# G1 · mjlab 训练与仿真工作空间

训练代码按方法集中到 [workspace/](workspace/README.md)：AMP、BeyondMimic、SuperMimic、速度 PPO 与新增 InstinctMJ 后端各自提供功能接口，公共流程放在 `workspace/common/`。继续使用 `.runtime-path` 指向的原运行环境；Instinct 的辅助依赖单独放在项目缓存，不升级核心运行栈。

新行走/动作模仿实验可使用 **[InstinctMJ 完整工具链](workspace/instinct/README.md)**，支持配置检查、动作转换、训练/续训、曲线、网页回放、评测、JIT/ONNX 导出与独立仿真。它是 InstinctLab 的官方 mjlab 移植；原模型仍使用原入口，两套机器人/模型约定不混用。接入范围和差异见 [接入说明](docs/InstinctMJ接入与完整工具链.md)。

| 方法 | 代码与配置 | 默认任务 | 仿真部署 |
| --- | --- | --- | --- |
| AMP | [workspace/amp](workspace/amp/README.md) | 舞蹈风格，4096 环境 | 导出 JIT/ONNX 后 `deploy --bundle` |
| BeyondMimic | [workspace/beyondmimic](workspace/beyondmimic/README.md) | 参考动作跟踪，4096 环境 | 导出包 `deploy --bundle` |
| SuperMimic | [workspace/supermimic](workspace/supermimic/README.md) | 世界模型 + BPTT，本机 1024 环境 / H=16 | 直接 `deploy --checkpoint` |
| 速度 PPO | [workspace/velocity](workspace/velocity/README.md) | 平地速度指令，4096 环境 | 导出包 `deploy --bundle` |
| InstinctMJ | [workspace/instinct](workspace/instinct/README.md) | 行走、单/多动作模仿、感知/VAE、Parkour 与 HOI，默认 256–4096 环境 | JIT/ONNX 包 `deploy --bundle` |

SuperMimic 使用固定提交的第三方独立复现 [SuperMimic-Reproduced](workspace/supermimic/upstream/)，完整上游源码已放入本方法目录；它不是论文作者官方源码。其他方法继续复用已安装的 mjlab 1.6.0 / RSL-RL 5.4.2，参考源码保存在 [third_party/](third_party/README.md)。

## 常用命令

在项目目录执行以下命令；`g1.sh` 自动选择原解释器、缓存与线程设置。

```bash
# 按方法查看帮助。
bash g1.sh methods
bash g1.sh instinct --help
bash g1.sh amp --help
bash g1.sh beyondmimic --help
bash g1.sh supermimic --help

# 先检查配置，不创建仿真或开始训练。
bash g1.sh amp train --dry-run
bash g1.sh beyondmimic train --dry-run
bash scripts/19_train_supermimic.sh --dry-run

# 分别执行需要的训练。
bash g1.sh amp train --iterations 30000 --name dance
bash g1.sh beyondmimic train --iterations 10000 --name dance_102
bash g1.sh supermimic preflight --allow-config-changes \
  --config workspace/supermimic/profiles/lafan1_4096_h16_eager.json
bash scripts/19_train_supermimic.sh --name lafan1_4096_h16_eager

# 各自 latest 只选择本方法模型。
bash g1.sh amp play --checkpoint latest --port 8080
bash g1.sh beyondmimic play --checkpoint latest --port 8080
bash g1.sh supermimic play --checkpoint latest --port 8080

# 公共检查和训练曲线。
bash g1.sh test
bash scripts/07_tensorboard.sh
```

AMP/BeyondMimic/速度 PPO 的 `--iterations` 是本次新增优化更新数，SuperMimic 是目标总仿真控制步数。SuperMimic 的 19 号脚本在此前视频 8GB 方案上，将环境数提升到 4096，保持 H=16、world/policy batch=512、eager；其余训练设置沿用上游，偏差写入 manifest。环境数与 replay 采样 batch 无固定比例，1024/2048/4096 环境各已通过 50 步/4 次更新容量短测，4096 最低仍剩约 2.07 GiB 显存；通用 `g1.sh supermimic train` 仍为上游严格规格。完整长训和动作效果尚未验证。历史 `local_8gb.json` 已发生长训 NaN。

SuperMimic 只保留上述 4096 环境配置。以后每个 run 在保存点和训练结束时，自动对全部动作做独立整段评测，按成功率、完成比例、回报选出最佳，最终只保留可续训的 `best.pt`；中间权重自动删除，日志与评测记录保留。`latest` 优先读取保留的最佳模型，无需每次手工清理。

数据准备、续训、定量评测、导出和仿真部署的完整命令矩阵见 **[workspace/README.md](workspace/README.md)**。默认 BeyondMimic 动作为 `motions/processed/dance_102.npz`，AMP 为 `amp_dance.npz`；SuperMimic 使用 `supermimic prepare` 准备固定版本的 LAFAN1 40 段动作。

Instinct 新训练的例子（`--iterations` 为新增 PPO 更新数）：

```bash
bash g1.sh instinct doctor
bash g1.sh instinct prepare --source motions/processed/amp_dance.npz
bash g1.sh instinct prepare-lafan
bash g1.sh instinct prepare-scene --demo
bash g1.sh instinct train --profile beyondmimic --iterations 30000 --name g1_dance
bash g1.sh instinct train --profile perceptive --name perceptive_teacher
# VAE 必须传已训练 Perceptive/OneMotion run 的最新数字 checkpoint。
bash g1.sh instinct train --profile perceptive_vae \
  --teacher-checkpoint /绝对路径/perceptive_teacher_perceptive/model_N.pt
bash g1.sh instinct play --profile beyondmimic --checkpoint latest --port 8080
```

首次拉取源码若缺少辅助包或网格，先执行 `bash g1.sh instinct setup`。行走无需准备动作，使用 `--profile locomotion`。平地感知 demo 可验证链路，但真实 benchmark 仍需匹配地形；Parkour 的 LAFAN locomotion 子集只是工程基线；HOI 仍需真实对象轨迹和网格。固定 InstinctMJ / instinct_rl 提交为 CC BY-NC 4.0，详见该方法说明。

## 目录与兼容性

- `workspace/<方法>/`：方法代码、功能接口和 `profiles/`。
- `workspace/common/`：PPO 共享流程、模型契约、动作转换、计时与资源工具。
- `outputs/<方法>/runs/`、`evaluations/`、`deploy/`：新训练与评测产物。
- `motions/`：共享参考动作；不在每种方法目录中重复复制。
- `scripts/`：编号快捷脚本及旧独立 Python 转发入口。
- `docs/`：原理、配置、评测与训练部署教程。
- `gym_rl/`、`third_party/`：参考资料与依赖源码，遵守各自许可证。

现有 `outputs/runs/`、`outputs/deploy/`、基准模型和动作数据保持原位置；新接口兼容它们。原配置字节、AMP 原算法文件与类路径保留，目录整理不会让已有模型因源码指纹变化而失效。旧 SuperMimic 实现和历史 checkpoint 已在上一轮清理，不恢复；SuperMimic 需要重新训练后才能回放。

## 阅读顺序

1. [目录、接口与完整命令](workspace/README.md)。
2. [InstinctMJ 训练到仿真工具链](workspace/instinct/README.md)、[接入差异与验收](docs/InstinctMJ接入与完整工具链.md)。
3. [AMP 舞蹈训练与仿真部署](docs/AMP舞蹈训练与仿真部署.md)。
4. [SuperMimic 实现差异与运行说明](docs/SuperMimic上游替换与实现差异.md)。
5. [训练到仿真部署教程](docs/training_to_sim_deployment.md)、[训练流程与配置详解](docs/训练流程与配置详解.md)。
6. [源码导航](docs/源码导读与注释索引.md)、[验证范围](docs/validation.md)、[快捷脚本说明](scripts/README.md)。

模型、训练曲线、缓存、网格和动作数据不上传；本仓库主要保存源码、文本配置、文档与许可证。当前部署范围为仿真，真机仅保留接口模板。

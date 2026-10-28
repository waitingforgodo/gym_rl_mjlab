# 常用脚本

当前推荐按方法使用 `bash g1.sh amp|beyondmimic|supermimic|velocity|instinct <功能>`。训练、回放、评测和部署代码已集中到对应的 `workspace/<方法>/`；完整目录、参数单位和部署方式见 **[统一方法接口](../workspace/README.md)**。编号脚本和独立 Python 程序继续作为兼容入口。

新增 21–25 号为官方 **[InstinctMJ 工具链](../workspace/instinct/README.md)** 的通用转发入口，共有 10 份 profile：基础行走/模仿默认 4096 环境，高级感知、VAE、Parkour、HOI 默认 256/512 环境。编号脚本不为每个高级 profile 重复建文件。原 21/22 号 SuperMimic 导出脚本已随旧实现移除，当前同编号使用不同文件名，不是旧脚本恢复。Instinct 的模型与原 AMP/RSL-RL 模型不兼容，回放与导出均只选本后端模型。

当前从零训练快捷脚本已经统一切换到 **4096 环境**。AMP、BeyondMimic 完整状态和无状态估计的实测结果见 [4096 并行配置与容量结果](../docs/4096并行配置与容量结果.md)。重复的 2048 profile 已删除；旧运行通过嵌入配置快照兼容。

AMP 舞蹈入口：`17_prepare_amp_motion.sh` 转换本地专家数据，`18_train_amp_dance.sh` 使用独立 AMP 任务训练；默认 `amp_dance_4096`、4096 环境、30000 更新。可在末尾加 `--iterations 100` 做短训，或用 `--motion` 选择其他已转换的标准 NPZ。完整中文说明见 [AMP 舞蹈训练与仿真部署](../docs/AMP舞蹈训练与仿真部署.md)。

方法入口在 `workspace/<方法>/cli.py`，训练、推理和部署接口分别在 `training.py`、`inference.py`、`deployment.py`。旧 `reinforcement_learning/rsl_rl/` 和 `deploy/deploy_sim.py` 仍可用 `bash g1.sh --python 程序路径 参数...` 启动，实际复用 `workspace/common/`。完整流程见 [训练到仿真部署教程](../docs/training_to_sim_deployment.md)。

这些脚本安装后位于 `~/g1-mjlab-workspace/scripts/`，即当前独立 mjlab 工作空间内。可以从任何目录执行；统一用 `bash 脚本路径`，不需要手动激活环境或修改执行权限。

## 最常用的四项

```bash
# 继续本工作空间最新正式训练，继承原环境数/种子，新增 1000 次更新
bash ~/g1-mjlab-workspace/scripts/03_resume.sh

# 浏览器回放，地址 http://localhost:8080
bash ~/g1-mjlab-workspace/scripts/04_play.sh

# 固定 0.5 m/s 前进、零转向，16 环境各评估 60 秒
bash ~/g1-mjlab-workspace/scripts/06_evaluate_forward.sh

# 另开终端启动曲线网页，地址 http://localhost:6006
bash ~/g1-mjlab-workspace/scripts/07_tensorboard.sh
```

SuperMimic 的 19 号训练、20 号回放、`supermimic/evaluate.py` 使用固定 `workspace/supermimic/upstream`，继续复用原运行环境。只保留 **4096 环境、H=16、world/policy batch=512、eager**（`lafan1_4096_h16_eager.json`）这一份最高已验证配置；使用完整 LAFAN1 40 段数据，其余训练设置保留上游默认，五项字段差异记录到 manifest。低规格与重复配置已删除。直接调用 `g1.sh supermimic train` 或给 19 号传 `--strict-upstream` 仍使用原始 dataclass，无需另一份配置文件。新 `--iterations` 为目标总控制步，默认 320000。

旧 SuperMimic 代码、兼容入口、诊断工具及产物已清理，临时短测模型也已清空。当前须先训练再回放。上游没有部署包导出功能，直接用 20 号仿真回放；原 21/22 号随旧部署代码移除。

```bash
# 校验当前 4096/H16、batch=512、eager 设置和固定数据，不创建仿真。
bash scripts/19_train_supermimic.sh --dry-run
# 当前设置的实际显存容量下界检查。
bash g1.sh supermimic preflight --allow-config-changes \
  --config workspace/supermimic/profiles/lafan1_4096_h16_eager.json
# 只有容量通过后才执行训练。
bash scripts/19_train_supermimic.sh --name lafan1_4096_h16_eager
bash scripts/07_tensorboard.sh
bash scripts/20_play_supermimic.sh --checkpoint latest --port 8080
bash g1.sh --python scripts/supermimic/evaluate.py --checkpoint latest
```

所有 SuperMimic train/resume 入口默认自动选优与清理：每个保存点及结束时独立评测全部动作，按整段成功率、完成比例、回报比较，每个 run 只保留可续训的 `best.pt`。同一步的重复保存合并，编号/latest 副本自动删除；`latest` 优先解析保留的最佳。日志与评测报告保留；评测失败会保护已有最佳并保留一份待查候选。数值防护继续拒绝 NaN/Inf。详情见 [上游替换与实现差异](../docs/SuperMimic上游替换与实现差异.md)。

## 全部入口

- `00_check.sh`：依赖/资源清单、CPU 测试、4 环境 32 步检查。
- `01_benchmark.sh`：1024/2048/4096 环境，各短训 100 次，比较吞吐和余量。
- `02_train.sh`：从零启动官方平地基线，`velocity_flat_4096`、4096 环境、3000 次更新。
- `03_resume.sh`：续训本空间最新正式模型，默认继承原环境数与种子，新增 1000 次更新。
- `04_play.sh`：回放策略，默认 Viser 网页。
- `05_evaluate.sh`：官方 play 随机命令，16 环境 × 60 秒。
- `06_evaluate_forward.sh`：固定机体系 `(0.5, 0, 0)`，16 环境 × 60 秒。
- `07_tensorboard.sh`：默认只载入最近一次正式训练的日志目录，排除历史 run、checks 和 benchmarks，不使用 GPU。同目录已有服务时直接返回网页链接；其他服务占用端口时自动选择后续可用端口，打印实际地址。开始新一轮训练后重新运行脚本即可选择新目录。`--logdir` 可手动指定其他目录，额外服务选项会传递给新服务。在 VS Code 终端按住 Ctrl 点击链接访问。PPO 曲线说明见 [训练流程与配置详解第 12 节](../docs/训练流程与配置详解.md#12-怎么用-tensorboard-看进度)。
- `08_export.sh`：导出 JIT/ONNX/部署清单并进行内置检查。
- `09_deploy_sim.sh`：指定导出包，运行同后端仿真回归。
- `10_import_motions.sh`：复制并转换 dance_102 和 gangnam_style，已有 NPZ 只检查。
- `11_train_tracking.sh`：另起 dance_102 动作跟踪，默认 `tracking_4096`、4096 环境。
- `12_summary.sh`：读取最新运行的曲线末值、吞吐及资源记录。
- `13_resume_legacy.sh`：从旧环境的 g1_continue 迁入。迁入后用 03。
- `14_show_config.sh`：打印实际保存的完整训练配置和运行清单。
- `15_train_robust.sh`：新的行走研究候选，默认 `velocity_robust_4096`、4096 环境、10000 更新。
- `16_train_tracking_robust.sh`：新的跟踪研究候选，默认 `tracking_robust_4096`、4096 环境、30000 更新；无状态估计版本传 `--profile tracking_no_state_estimation_robust_4096`。

- `17_prepare_amp_motion.sh`：将本地 AMP 舞蹈转换为当前 G1 标准 NPZ，相同来源的已有数据校验后复用。
- `18_train_amp_dance.sh`：默认 `amp_dance_4096`、4096 环境、30000 更新，判别器参数保持实际 4096 容量短测时的设置。
- `19_train_supermimic.sh`：上游世界模型/BPTT，当前默认 4096 环境、H=16、world/policy batch=512、eager；其余设置沿用上游，默认 320000 总控制步。`--strict-upstream` 回到完整原始规格。
- `20_play_supermimic.sh`：上游 checkpoint 网页/原生回放或无界面链路检查。
- `21_train_instinct.sh`：默认 Instinct 行走、4096 环境；`--profile` 可选全部 10 个 Instinct profile，`--iterations` 为本次新增 PPO 更新数。高级数据用 `g1.sh instinct prepare-lafan` 或 `prepare-scene` 准备。
- `22_play_instinct.sh`：仅回放 Instinct 模型，可用 `--profile` 限定 latest 的任务，输出网页链接。
- `23_export_instinct.sh`：导出冻结归一化器与 actor 的 JIT/ONNX，执行真实观测数值对照。
- `24_deploy_instinct.sh`：指定 `--bundle`，使用 `--engine jit|onnx` 运行独立仿真。
- `25_prepare_instinct_motion.sh`：将已有标准 NPZ 转为 torso-root 参考动作，校正 waist 并校验 FK。

```bash
bash g1.sh instinct doctor
# 缺少项目内辅助包或网格时先运行：bash g1.sh instinct setup
bash scripts/25_prepare_instinct_motion.sh --source motions/processed/amp_dance.npz
bash scripts/21_train_instinct.sh --profile beyondmimic --iterations 30000 --name g1_dance
bash scripts/22_play_instinct.sh --profile beyondmimic --checkpoint latest --port 8080
bash scripts/23_export_instinct.sh --profile beyondmimic --checkpoint latest
bash scripts/24_deploy_instinct.sh --bundle /完整路径/部署包 --headless --steps 500
```

新配方的来源、中文参数说明和公平对照步骤见 [训练配置优化依据与对照实验](../docs/训练配置优化依据与对照实验.md)。使用 `bash g1.sh --python scripts/tools/compare_policies.py --help` 查看多策略比较工具。

`common.sh` 是上述脚本的公共代码，不单独运行。进入脚本源码可以看到每个默认参数及中文说明。

## 参数覆盖

参数加在命令末尾即可覆盖脚本默认值，不必改脚本。例如：

```bash
bash ~/g1-mjlab-workspace/scripts/03_resume.sh --iterations 2000 --name walk_more
bash ~/g1-mjlab-workspace/scripts/04_play.sh --port 8081
bash ~/g1-mjlab-workspace/scripts/07_tensorboard.sh --port 6007
bash ~/g1-mjlab-workspace/scripts/06_evaluate_forward.sh --velocity-cmd 1.0 0 0 --seconds 120
bash ~/g1-mjlab-workspace/scripts/01_benchmark.sh --envs 2048 4096 --iterations 200
```

指定你刚评估过的模型，避免 `latest` 随后来新实验改变：

```bash
MODEL="$HOME/g1-mjlab-workspace/outputs/runs/20261023_173251_619996_velocity_flat_g1_2048_continue/model_3198.pt"
bash ~/g1-mjlab-workspace/scripts/06_evaluate_forward.sh --checkpoint "$MODEL"
bash ~/g1-mjlab-workspace/scripts/03_resume.sh --checkpoint "$MODEL" --iterations 1000
```

也可用环境变量设置默认值：

```bash
G1_ENV_COUNT=1024 bash ~/g1-mjlab-workspace/scripts/03_resume.sh
G1_CPU_THREADS=4 bash ~/g1-mjlab-workspace/scripts/03_resume.sh
G1_CHECKPOINT="$MODEL" bash ~/g1-mjlab-workspace/scripts/04_play.sh
```

默认 2 CPU 线程和 nice=10。GPU、系统内存仍共享；训练、回放、评估、导出都可能使用 GPU，建议依次进行。TensorBoard 可以另开终端与训练同时运行。

## 查看和导出

```bash
bash ~/g1-mjlab-workspace/scripts/12_summary.sh
bash ~/g1-mjlab-workspace/scripts/14_show_config.sh
bash ~/g1-mjlab-workspace/scripts/08_export.sh
# 使用上一条打印的 DEPLOY_BUNDLE 实际目录
bash ~/g1-mjlab-workspace/scripts/09_deploy_sim.sh --bundle /完整的导出目录
```

`latest` 只按正式运行目录时间与数字模型编号选择，不按评估成绩选择，也不会选择 benchmark 的模型。平地任务与 tracking 混合时，请明确指定 checkpoint；03 续训默认继承该运行的环境数和种子，只有显式传入 `--envs`、`--seed` 或 `G1_ENV_COUNT` 时才覆盖。

## 动作模仿

```bash
bash ~/g1-mjlab-workspace/scripts/10_import_motions.sh
bash ~/g1-mjlab-workspace/scripts/00_check.sh --profile tracking_4096 --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz
bash ~/g1-mjlab-workspace/scripts/01_benchmark.sh --profile tracking_4096 --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz --envs 2048 4096
bash ~/g1-mjlab-workspace/scripts/11_train_tracking.sh
```

跟踪任务用 05 的随机/动作协议评估，不使用专门针对速度任务的 06 固定前进测试。

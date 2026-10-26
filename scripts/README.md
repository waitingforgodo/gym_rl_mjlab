# 常用脚本

需要逐个查看 Python 程序时，训练/续训/回放/评估/导出入口在 `reinforcement_learning/rsl_rl/`，仿真部署入口在 `deploy/deploy_sim.py`。用 `bash g1.sh --python 程序路径 参数...` 启动；核心实现和完整流程见 [训练到仿真部署教程](../docs/training_to_sim_deployment.md)。原有编号 shell 脚本继续可用。

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

## 全部入口

- `00_check.sh`：依赖/资源清单、CPU 测试、4 环境 32 步检查。
- `01_benchmark.sh`：512/1024/2048 环境，各短训 100 次，比较吞吐和余量。
- `02_train.sh`：从零启动官方平地基线，2048 环境、3000 次更新。
- `03_resume.sh`：续训本空间最新正式模型，默认继承原环境数与种子，新增 1000 次更新。
- `04_play.sh`：回放策略，默认 Viser 网页。
- `05_evaluate.sh`：官方 play 随机命令，16 环境 × 60 秒。
- `06_evaluate_forward.sh`：固定机体系 `(0.5, 0, 0)`，16 环境 × 60 秒。
- `07_tensorboard.sh`：打开训练曲线服务，不使用 GPU；启动时输出可点击的网页链接，使用 `--port` 时链接同步更新。在 VS Code 终端按住 Ctrl 点击访问。41 条曲线的含义及阅读顺序见 [训练流程与配置详解第 12 节](../docs/训练流程与配置详解.md#12-怎么用-tensorboard-看进度)。
- `08_export.sh`：导出 JIT/ONNX/部署清单并进行内置检查。
- `09_deploy_sim.sh`：指定导出包，运行同后端仿真回归。
- `10_import_motions.sh`：复制并转换 dance_102 和 gangnam_style，已有 NPZ 只检查。
- `11_train_tracking.sh`：另起 dance_102 动作跟踪，暂用 256 环境。
- `12_summary.sh`：读取最新运行的曲线末值、吞吐及资源记录。
- `13_resume_legacy.sh`：从旧环境的 g1_continue 迁入。迁入后用 03。
- `14_show_config.sh`：打印实际保存的完整训练配置和运行清单。
- `15_train_robust.sh`：新的行走研究候选，默认 2048 环境、10000 更新。
- `16_train_tracking_robust.sh`：新的跟踪研究候选，默认 256 环境、30000 更新；可传 `--profile tracking_no_state_estimation_robust`。

新配方的来源、中文参数说明和公平对照步骤见 [训练配置优化依据与对照实验](../docs/训练配置优化依据与对照实验.md)。使用 `bash g1.sh --python scripts/tools/compare_policies.py --help` 查看多策略比较工具。

`common.sh` 是上述脚本的公共代码，不单独运行。进入脚本源码可以看到每个默认参数及中文说明。

## 参数覆盖

参数加在命令末尾即可覆盖脚本默认值，不必改脚本。例如：

```bash
bash ~/g1-mjlab-workspace/scripts/03_resume.sh --iterations 2000 --name walk_more
bash ~/g1-mjlab-workspace/scripts/04_play.sh --port 8081
bash ~/g1-mjlab-workspace/scripts/07_tensorboard.sh --port 6007
bash ~/g1-mjlab-workspace/scripts/06_evaluate_forward.sh --velocity-cmd 1.0 0 0 --seconds 120
bash ~/g1-mjlab-workspace/scripts/01_benchmark.sh --envs 1024 2048 --iterations 200
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
bash ~/g1-mjlab-workspace/scripts/00_check.sh --profile tracking --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz
bash ~/g1-mjlab-workspace/scripts/01_benchmark.sh --profile tracking --motion ~/g1-mjlab-workspace/motions/processed/dance_102.npz --envs 128 256 512
bash ~/g1-mjlab-workspace/scripts/11_train_tracking.sh
```

跟踪任务用 05 的随机/动作协议评估，不使用专门针对速度任务的 06 固定前进测试。

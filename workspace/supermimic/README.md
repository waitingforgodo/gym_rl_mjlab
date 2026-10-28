# SuperMimic

完整上游源码在 [upstream/](upstream/)，固定提交和文件哈希见 [SOURCE_INFO.json](upstream/SOURCE_INFO.json)。训练器、世界模型和 BPTT 来自 CuboiLeo/SuperMimic-Reproduced；它是第三方独立复现，不是作者官方源码。

本机只保留最高已验证配置 [lafan1_4096_h16_eager.json](profiles/lafan1_4096_h16_eager.json)：**4096 环境、H=16、world/policy batch=512、eager**，使用完整 LAFAN1 40 段动作。replay=1000、每次 world 更新 10 个 batch、学习率 2e-4、320000 总控制步及其余设置沿用上游。manifest 记录 H、两类 batch 和两处 compile 共五项偏差；旧 512/1024、低显存及重复路径配置已删除。

此前保留大 batch/compile 的 1024/H16 配置在首次更新处遇到 PyTorch 2.9.1 / mjlab 的 TF32 接口冲突，本地进程兼容修正后已完成真实编译更新。其完整短测最低显存余量只有 19 MiB；只降低环境数也未解决大 batch 的占用。当前配置同时降低两类 batch 并使用 eager；旧规格的报告保留在 outputs，旧配置文件已删除。

1024/2048/4096 环境均按正式预算与约 2e-4 学习率运行到第 50 步日志点，各完成 4 次真实更新及有限值保存，CLI 退出码为 0。PyTorch 峰值分配分别约 2.39/3.06/4.41 GiB，整卡最低剩余显存分别为 4598/3794/2116 MiB；当前 4096 仍有约 2.07 GiB 余量。短测的 `max_hours=0` 仅用于检查，正式 profile 无墙钟上限，仍每 4000 步保存；完整长训与动作效果尚未验证。

环境数与 batch 无固定比例。环境数决定每步采集量及 replay 的环境维度；world batch 从 replay 抽取窗口，policy batch 抽取起始状态并展开 H 步，两者可独立于环境数设置。保持 UTD=0.4 时，翻倍环境数增加每步真实数据量，优化更新次数不变，单位新增数据对应的更新量相应减少；不能把环境翻倍等同于训练耗时减半。

```bash
# 检查本机当前设置，不开始训练。
bash scripts/19_train_supermimic.sh --dry-run
bash g1.sh supermimic preflight --allow-config-changes \
  --config workspace/supermimic/profiles/lafan1_4096_h16_eager.json
# 需要开始新的完整训练时执行；不续用损坏的 dance 模型。
bash scripts/19_train_supermimic.sh --name lafan1_4096_h16_eager
# 另开终端，只显示当前这一轮训练；旧网页服务需先 Ctrl+C 后重启。
bash scripts/07_tensorboard.sh
```

通用 `g1.sh supermimic train` 继续逐字段对齐固定上游 `TrainConfig`，使用 4096 环境、H=32 及其余原始参数。19 号脚本传 `--strict-upstream` 时也回到该入口；显式 `--config` / `--resume` 按调用者参数执行。

```bash
# 只在数据缺失时准备；已有固定版本数据会校验后复用。
bash g1.sh supermimic prepare
bash g1.sh supermimic train --dry-run
# 实际分配原始尺寸的 replay 和策略展开图，不创建训练模型。
bash g1.sh supermimic preflight
# 通过容量预检后才创建新 run 并执行完整上游训练。
bash g1.sh supermimic train --name lafan1_upstream
bash g1.sh supermimic play --checkpoint latest
bash g1.sh supermimic evaluate --checkpoint latest
bash g1.sh supermimic deploy --checkpoint latest --headless --steps 500
```

`--iterations` 是目标总控制步数，默认 UTD=0.4。严格模式会拒绝缩小 batch/replay、改变 H/预算、筛选动作或关闭 compile，并列出字段差异；它直接使用上游 dataclass，不需要重复配置文件。自定义实验必须显式加 `--allow-config-changes`，差异会写入 manifest，不会称为严格复现。

2026-10-28 本机 RTX 5070 Laptop（7.96 GiB 物理显存）实测：仅原始 replay 和策略 32 步展开图，峰值分配就达到约 11.04 GiB，尚不含仿真、世界模型、critic 和优化器。预检因此阻止正式训练，不会自动缩参；驱动允许超量分配也不视为物理显存内可运行。更大 GPU 上仍需运行完整预检/训练才能确认容量。

`safety.py` 在本地连接层检查动作、梯度、更新指标及模型/优化器状态；非有限值会立即停止，状态写入 `resources.json`。Checkpoint 先通过有限值检查，再临时写入并原子替换，防止坏权重覆盖最后正常文件。加载和推理也拒绝 NaN/Inf。上游算法文件及其哈希保持不变；有效更新与未加防护时逐参数一致。

首次 world 更新和 policy/critic 更新各打印开始/完成日志；选择 compile 时，首次调用的计算图编译可能持续数分钟。`Not enough SMs to use max_autotune_gemm mode` 是优化选择提示，编译短测出现该提示后仍成功结束。`resources.json` 记录最终状态、完成步数/更新数、最后训练阶段及 PyTorch 分配/预留显存峰值；PyTorch 统计不包含 Warp 等其他分配。TF32 兼容只替换当前进程中的后端设置函数，不改安装文件或训练参数；当前 eager/batch 设置来自用户选择的 profile。

## 每次训练只留最佳

所有本地 train/resume 入口默认启用 `retention.py`，无需每次追加清理选项。每 4000 控制步的保存点、正常/限时结束和正常 Ctrl+C 保存时，独立进程按上游 Appendix F 协议，对该 run 全部动作从首帧各评测一次，关闭观测/动作/重置噪声及域随机化。依次比较整段成功率、平均动作完成比例、平均归一化回报；相同分数保留原最佳，不用模型内想象回报或最后步数代替实际效果。

每个 run 最终只保留一个 `best.pt`，包含网络、优化器、训练配置、步数和完整评测证据，可用于续训。同一步的 latest/编号/最终保存只评测一次。候选评测成功、状态有限后，才原子发布最佳并删除该 run 的中间权重。保留 `best.json`、`selection.jsonl`、`selection/` 报告和训练日志；其他 run、AMP/BeyondMimic 的模型不会被此策略删除。

若评测失败则停止训练，保留旧最佳及一份待查候选 `.candidate.pt`，记录 `selection_error.json`。续训先继承源最佳，再比较新候选；评测在独立进程运行，不消耗训练进程的随机数或改变其采样器。`latest`/run 目录优先解析 `best.pt`，旧 `ckpt_latest.pt` 仍可读取。

```bash
bash g1.sh supermimic resume --allow-config-changes \
  --checkpoint outputs/supermimic/runs/lafan1_4096_h16_eager/best.pt \
  --name lafan1_continue
```

旧 `dance` 当前没有可续训 checkpoint，情况记录保存在本地 `outputs/supermimic/checks/20261028_invalid_dance_cleanup.json`。新的正式模型需从新 run 开始。

继续使用原运行环境，没有升级核心依赖；manifest 的 `runtime_lock_differences` 会列出与上游 `uv.lock` 的差异。因此“训练参数对齐”仍不等于“上游锁定环境完全复现”。`deploy` 直接加载 checkpoint；上游尚无独立 JIT/ONNX 导出和真机部署。

本地入口：`configuration.py` 合并参数，`strict.py` 对照原始配置和 40 段数据哈希，`capacity.py` 检查显存下界，`safety.py` 提供数值与原子保存防护，`training.py` 执行上游 Trainer，`retention.py` 自动评测选优和清理，`inference.py` 回放/评测。固定上游算法与文件哈希未修改。

本次完整回归 86 项通过，包含 10 项新配置/数值防护测试；72 份固定上游文件哈希均未变。真实坏 latest 加载已明确拒绝；正常的 500 步模型复测回报仍为约 0.5194，与加防护前一致。原始训练尚未通过本机容量预检，不宣称已修复或验证长训收敛。

完整命令见 [统一说明](../README.md)，实现差异与依赖兼容说明见 [SuperMimic 文档](../../docs/SuperMimic上游替换与实现差异.md)。

# SuperMimic

完整上游源码在 [upstream/](upstream/)，固定提交和文件哈希见 [SOURCE_INFO.json](upstream/SOURCE_INFO.json)。训练器、世界模型和 BPTT 来自 CuboiLeo/SuperMimic-Reproduced；它是第三方独立复现，不是作者官方源码。

本地代码按功能拆分：`configuration.py` 解析参数/数据，`checkpoints.py` 管理模型契约，`training.py` 连接上游训练器，`inference.py` 回放/评测，`deployment.py` 提供 checkpoint 仿真，`runtime.py` 和 `paths.py` 维护源码和原环境路径。

```bash
bash g1.sh supermimic train --config workspace/supermimic/profiles/local_8gb.json --dry-run
bash g1.sh supermimic train --config workspace/supermimic/profiles/local_8gb.json \
  --iterations 3000 --name dance
bash g1.sh supermimic resume --checkpoint outputs/supermimic/runs/dance/ckpt_latest.pt \
  --iterations 7540 --name dance_continue
bash g1.sh supermimic play --checkpoint latest
bash g1.sh supermimic evaluate --checkpoint latest
bash g1.sh supermimic deploy --checkpoint latest --headless --steps 500
```

`--iterations` 是目标总控制步数，默认 UTD=0.4；3000 步约 1184 次更新。`local_8gb.json` 为 4096 环境的缩小 batch/replay 工程规格，不等于论文参数。`deploy` 直接加载 checkpoint；上游尚无独立 JIT/ONNX 导出和真机部署。

完整命令见 [统一说明](../README.md)，实现差异与依赖兼容说明见 [SuperMimic 文档](../../docs/SuperMimic上游替换与实现差异.md)。

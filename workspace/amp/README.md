# AMP

本目录集中 AMP 训练、回放、评测与部署。算法由 PPO 和动作风格判别器构成；默认 `amp_dance_4096`、4096 环境，参考 `motions/processed/amp_dance.npz`。

| 文件 | 功能 |
| --- | --- |
| cli.py | `bash g1.sh amp <功能>` |
| training.py | train / resume；复用 common 的 PPO 循环 |
| inference.py | play / evaluate |
| deployment.py | export 为 JIT/ONNX 包、deploy 同后端仿真 |
| data.py / conversion.py | prepare 专家动作 |
| algorithm.py | AmpPPO、判别器优化与历史策略 replay |
| features.py / reference.py | AMP 特征与连续专家转移 |
| tasks.py / environment.py | G1 任务、参考状态初始化、终止状态采集 |
| profiles/ | JSON 配置，保留原字节与配置名 |

```bash
bash g1.sh amp prepare
bash g1.sh amp train --dry-run
bash g1.sh amp train --iterations 30000
bash g1.sh amp resume --checkpoint latest --iterations 1000
bash g1.sh amp play --checkpoint latest
bash g1.sh amp evaluate --checkpoint latest --seconds 10
bash g1.sh amp export --checkpoint latest
bash g1.sh amp deploy --bundle /完整路径/导出包 --headless --seconds 10
```

部署只使用 actor，不使用判别器。`--iterations` 是新增 PPO 更新数。完整接口与旧模型路径兼容说明见 [统一说明](../README.md)，原理见 [AMP 教程](../../docs/AMP舞蹈训练与仿真部署.md)。

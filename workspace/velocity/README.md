# 速度指令 PPO

默认 `velocity_flat_4096`、4096 环境，训练机器人跟随速度指令。`profiles/` 包含官方平地、gym PPO 参数和 robust 候选。

与 AMP/BeyondMimic 使用同一组功能：`training.py`、`inference.py`、`deployment.py`，共享实现位于 `workspace/common/`。

```bash
bash g1.sh velocity train --dry-run
bash g1.sh velocity train --iterations 3000
bash g1.sh velocity resume --checkpoint latest --iterations 1000
bash g1.sh velocity play --checkpoint latest
bash g1.sh velocity evaluate --checkpoint latest --velocity-cmd 0.5 0 0 --seconds 60
bash g1.sh velocity export --checkpoint latest
bash g1.sh velocity deploy --bundle /完整路径/导出包 --headless --seconds 10
```

`--iterations` 是新增 PPO 更新数。完整接口、模型选择和输出位置见 [统一说明](../README.md)。

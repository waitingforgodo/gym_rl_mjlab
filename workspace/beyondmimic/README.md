# BeyondMimic 风格动作跟踪

默认 `tracking_4096`、4096 环境、`motions/processed/dance_102.npz`。本目录是方法接口与配置；环境、奖励和 PPO 仍使用当前安装的 mjlab/RSL-RL，不另复制一套训练算法。

`training.py` 提供 train/resume，`inference.py` 提供 play/evaluate，`deployment.py` 提供 export/deploy，`data.py` 准备参考动作。配置保留 tracking 名称，包括完整状态、无状态估计和 robust 版本；名字与 JSON 内容稳定，便于读取已有模型。

```bash
bash g1.sh beyondmimic prepare dance_102
bash g1.sh beyondmimic train --dry-run
bash g1.sh beyondmimic train --iterations 10000
bash g1.sh beyondmimic resume --checkpoint latest --iterations 1000
bash g1.sh beyondmimic play --checkpoint latest
bash g1.sh beyondmimic evaluate --checkpoint latest --seconds 10
bash g1.sh beyondmimic export --checkpoint latest
bash g1.sh beyondmimic deploy --bundle /完整路径/导出包 --headless --seconds 10
bash g1.sh show-sources --kind tracking
```

`--iterations` 是新增 PPO 更新数。导出包包含参考动作及策略接口信息，跟踪回放需要参考轨迹。更多命令见 [统一说明](../README.md)；G1 环境源码导航见 [源码索引](../../docs/源码导读与注释索引.md)。

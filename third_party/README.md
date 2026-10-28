# 固定版本的 mjlab 与 RSL-RL 源码

实际运行的 [SuperMimic-Reproduced](../workspace/supermimic/upstream/) 已随方法迁入 `workspace/supermimic/upstream/`，固定提交 `ba6105349069bf040f1fdad60873d13138d505a2`，保留 MIT 许可证和逐文件 [SOURCE_INFO.json](../workspace/supermimic/upstream/SOURCE_INFO.json)。它继续使用原环境，不执行依赖安装。统一接口见 [workspace/README.md](../workspace/README.md)；这里保留 mjlab/RSL-RL 两份只读导航副本。

新增 InstinctMJ 和匹配 instinct_rl 的完整源码也按功能放在 [workspace/instinct/upstream/](../workspace/instinct/upstream/)，由 `instinct` 入口实际导入，不放在本目录的只读参考副本中。两个固定提交为 CC BY-NC 4.0；原运行栈不升级，附加依赖与二进制资源单独在项目内准备。提交、兼容边界和验证见 [Instinct 接入说明](../docs/InstinctMJ接入与完整工具链.md)。

这里保存当前训练依赖的源码，方便在 IDE 中直接阅读、搜索和跳转。已与官方版本标签的源码压缩包逐文件核对：保留的源码及文本配置与上游完全一致，也与当前虚拟环境中的安装文件一致。第三方源码保留原注释；中文调用说明见 [源码导读与注释索引](../docs/源码导读与注释索引.md)。

| 依赖 | 本地源码目录 | Python 文件 | 源码及文本配置文件 | 许可证 |
| --- | --- | ---: | ---: | --- |
| mjlab 1.6.0 | [mjlab-v1.6.0/src/mjlab](mjlab-v1.6.0/src/mjlab/) | 192 | 200 | [Apache-2.0](mjlab-v1.6.0/LICENSE) |
| RSL-RL 5.4.2 | [rsl_rl-v5.4.2/rsl_rl](rsl_rl-v5.4.2/rsl_rl/) | 30 | 30 | [BSD-3-Clause](rsl_rl-v5.4.2/LICENSE) |

两份源码及文本配置合计约 1.60 MiB，另保留许可证和来源清单。RSL-RL 安装包附带的依赖许可证也保存在 [licenses/dependencies](rsl_rl-v5.4.2/licenses/dependencies/)。

## 从哪里开始看

按训练调用顺序阅读：

1. [workspace/common/training.py](../workspace/common/training.py)：从零训练与续训，准备配置、目录和模型，调用 `runner.learn()`。
2. [workspace/common/runtime.py](../workspace/common/runtime.py)：`make_env_runner()` 构造环境、包装器与任务 runner。
3. [mjlab/rl/runner.py](mjlab-v1.6.0/src/mjlab/rl/runner.py)：衔接 mjlab 与 RSL-RL，支持模型加载和导出。跟踪任务还有专用 [tracking/rl/runner.py](mjlab-v1.6.0/src/mjlab/tasks/tracking/rl/runner.py)。
4. [rsl_rl/runners/on_policy_runner.py](rsl_rl-v5.4.2/rsl_rl/runners/on_policy_runner.py)：`learn()` 完成采样、计算回报、PPO 更新和保存。
5. [rsl_rl/algorithms/ppo.py](rsl_rl-v5.4.2/rsl_rl/algorithms/ppo.py)：动作采样、损失、KL 自适应学习率、反向传播和梯度裁剪。

具体配置和数据处理可按下表查找：

| 内容 | 本地文件 |
| --- | --- |
| G1 行走任务配置 | [velocity/config/g1/env_cfgs.py](mjlab-v1.6.0/src/mjlab/tasks/velocity/config/g1/env_cfgs.py) |
| 行走观测、奖励权重、随机化和控制频率 | [velocity/velocity_env_cfg.py](mjlab-v1.6.0/src/mjlab/tasks/velocity/velocity_env_cfg.py) |
| 行走网络与 PPO 默认参数 | [velocity/config/g1/rl_cfg.py](mjlab-v1.6.0/src/mjlab/tasks/velocity/config/g1/rl_cfg.py) |
| 行走奖励函数 | [velocity/mdp/rewards.py](mjlab-v1.6.0/src/mjlab/tasks/velocity/mdp/rewards.py) |
| 动作模仿任务配置 | [tracking/config/g1/env_cfgs.py](mjlab-v1.6.0/src/mjlab/tasks/tracking/config/g1/env_cfgs.py) |
| 参考动作与自适应采样 | [tracking/mdp/commands.py](mjlab-v1.6.0/src/mjlab/tasks/tracking/mdp/commands.py) |
| 环境 step/reset 与物理子步 | [envs/manager_based_rl_env.py](mjlab-v1.6.0/src/mjlab/envs/manager_based_rl_env.py) |
| G1 机器人、驱动、默认姿态与动作尺度 | [unitree_g1/g1_constants.py](mjlab-v1.6.0/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py) |
| Actor/Critic 网络 | [models/mlp_model.py](rsl_rl-v5.4.2/rsl_rl/models/mlp_model.py) |
| 观测归一化 | [modules/normalization.py](rsl_rl-v5.4.2/rsl_rl/modules/normalization.py) |
| Rollout 缓存、GAE、minibatch | [storage/rollout_storage.py](rsl_rl-v5.4.2/rsl_rl/storage/rollout_storage.py) |

本项目的新增训练配置在 [profiles](../workspace/) 和 [workspace/common/recipes.py](../workspace/common/recipes.py)，阅读官方默认值时需结合它们查看最终覆盖值。

## 与实际训练的关系

训练仍通过 `.runtime-path` 指定的虚拟环境导入已安装的 `mjlab==1.6.0`、`rsl-rl-lib==5.4.2`。这里的 mjlab/RSL-RL 副本不加入导入路径，修改它们不会改变训练；SuperMimic-Reproduced 则由专用入口直接导入其 `src/`。实际 PPO 依赖导入位置和关键函数行号可执行：

```bash
bash g1.sh show-sources --kind all
```

这两份目录对应用户指定的上游 Python 包子树，包含其中全部 Python 源文件和文本配置。按本仓库只保存源码的约定，未复制 mjlab 的 58 个 STL 网格，也未引入模型权重、动作数据、缓存、训练日志或上游 Git 历史。因此它们是供阅读与审查的源码副本，不能直接作为包含全部资源的独立安装目录。

## 版本来源与校验

- mjlab 上游：[v1.6.0/src/mjlab](https://github.com/mujocolab/mjlab/tree/v1.6.0/src/mjlab)。当前安装元数据记录提交 `0fb8a681136be94ffc636a3dd423cabb97d91f10`，请求版本为 `v1.6.0`。
- RSL-RL 上游：[v5.4.2/rsl_rl](https://github.com/leggedrobotics/rsl_rl/tree/v5.4.2/rsl_rl)。安装版本为 `rsl-rl-lib==5.4.2`，保留的 30 个源文件已与官方标签逐文件比较一致；安装元数据没有记录 Git 提交号。
- 各包的 `SOURCE_INFO.json` 保存复制来源、版本、上游压缩包 URL、压缩包 SHA-256、核对文件数和排除项。
- [manifest.json](manifest.json) 保存每个源码文件、许可证及 `SOURCE_INFO.json` 的 SHA-256，便于识别后续本地修改。压缩包仅下载到临时目录，没有纳入本仓库。

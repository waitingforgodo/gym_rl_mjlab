# gym_rl 参考资料

本目录收录用户提供的强化学习、动作模仿和部署参考项目，作为普通目录随 `gym_rl_mjlab` 版本管理。

| 目录 | 内容 |
| --- | --- |
| `robot_lab-main/` | Isaac Lab 训练任务、算法配置与文本机器人描述 |
| `whole_body_tracking-main/` | BeyondMimic 风格动作跟踪训练和导出实现 |
| `rsl_rl-main/` | PPO、网络、采样缓存和 runner 参考实现 |
| `rl_sar-main/` | 仿真/真机部署参考程序、接口和配置 |

保留提供时的源码目录层级，包括重复解压的子目录；相同文件内容由 Git 自动去重。仓库只收录 Python/C++ 等源码、脚本、构建与任务配置、文本机器人描述、文档和许可证。

参考策略权重、机器人网格、图片、动作 CSV/NPZ、运行输出、缓存、编译产物和 Windows 下载标记 `*:Zone.Identifier` 均不上传。本地已有的原始文件保留。克隆仓库后，如需运行完整参考项目或导入动作，请自行准备对应资源；`import-reference` 的 `--source` 应指向包含原始 CSV 的资料目录。

导入前该目录自带一个本地 Git 仓库，提交为 `4b590d9`（`init`）。其元数据已完整移至本机 `.cache/git-backups/` 保存，不上传到远程，也不作为 Git 子模块提交。

各参考项目的安装方式和许可要求以各自 README、LICENSE 和源码声明为准。当前工作空间实际训练使用固定安装版本 mjlab 1.6.0 / RSL-RL 5.4.2，参考目录中的版本不替代运行时。实际训练到仿真部署流程见 [完整教程](../docs/training_to_sim_deployment.md)。

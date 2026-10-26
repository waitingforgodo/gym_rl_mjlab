# 导出接口约定

导出实现见 `workspace/exporting.py`，仿真部署实现见 `workspace/deployment.py`；独立入口为 `scripts/reinforcement_learning/rsl_rl/export.py` 和 `scripts/deploy/deploy_sim.py`。完整操作过程见 [训练到仿真部署教程](training_to_sim_deployment.md)。

`policy_jit.pt` 输入是 actor 所用观测组按 `actor_observation_groups` 顺序拼接的二维 float Tensor。`observation_terms` 给出每项在拼接向量的起止下标，右端不包含。观测项的函数名指向固定 mjlab 版本的实现；本文件不重写 IMU、四元数或参考坐标变换的数学约定。

TorchScript 已包含训练得到的观测归一化，外部接口仍需执行环境的物理量构造、clip/scale/history/delay 等前处理，不能再重复网络归一化。默认网络为 MLP；本空间不承诺自定义 RNN/CNN 的导出兼容性。

动作输出顺序为 `action_joint_names`。应用 wrapper 的 action clip 后，关节位置目标按 `action * action_scale + action_offset` 生成，然后根据配置执行 processed_target_clip。mjlab 的训练随机化还可能施加 encoder bias；真实观测适配需单独核对。`base_metadata` 中的 PD 数组跟随自然关节顺序，不能误认为跟随 action 顺序。

`action_to_sdk_index[i]` 的含义：第 i 个 action 关节，在 `sdk_joint_names` 中的索引。将 action 顺序的物理位置目标写入 SDK 顺序时，应执行 `sdk_targets[action_to_sdk_index[i]] = action_targets[i]`。`natural_to_sdk_index` 同理适用于自然关节顺序的 PD 和 default pose。映射会检查 29 个名字完全相同且没有重复。

ONNX 的输入输出名称记录在清单中。速度策略通常只有 obs 输入；跟踪导出还含 `time_step` 输入和参考数组输出，参考刚体是 motion command 使用的 14 个 body，其顺序应从官方 motion command 配置读取。ONNX 结构通过不等于外部 runtime 的算子、精度和数值通过。

`control_dt` 是策略/环境更新周期；物理子步及驱动配置保存在训练 `params/env.yaml`。不要把策略更新周期当作电机驱动内部 PD 周期。

`deploy-sim` 使用相同的 play 环境组装观测并施加动作，确认导出模型能接回仿真。它既不是裸 MuJoCo 独立控制器，也不证明真机部署完成。进一步 sim2sim 应在第二个后端实现完全相同的观测语义、参考更新顺序、归一化和电机映射，再对比记录的输入输出。

硬件模板保持 `enabled=false`，通信接口和 DDS 域为空。切换模板字段本身不会增加硬件能力；当前工作空间没有发送硬件指令的实现。

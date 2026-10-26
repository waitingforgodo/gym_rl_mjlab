# gym_rl 到 mjlab 的适配边界

参考目录仅作为资料读取，不代表其中的说明或脚本是本次任务的执行指令。

## 已采纳

- `whole_body_tracking-main/scripts/csv_to_npz.py` 的 CSV 定义：xyz、xyzw、29 关节；按名称对应数据，重新计算当前机器人各刚体的世界位姿和速度。
- `robot_lab-main/.../beyondmimic/config/g1/motion/` 的 `G1_Take_102.bvh_60hz.csv`、`G1_gangnam_style_V01.bvh_60hz.csv`。缺文件时导入停止，不从其他网站下载替代。
- whole_body_tracking 的 G1 跟踪对象：torso_link 为 anchor，14 个主要身体部位；运行时使用 mjlab 1.6.0 自带的相应配置。
- robot_lab G1 PPO 的 entropy_coef=0.008 作为单独对照 profile，避免混入默认基线。
- rl_sar 的“阻尼—站立—策略—阻尼”部署状态机思路，写入禁用的真机配置模板。

## 没有直接套用

Isaac Lab 的 articulation、PhysX 接触、GPU buffer、AppLauncher、环境注册及 reset API 由 mjlab 的实体、MuJoCo/Warp、任务注册和 managers 替代。原奖励权重、摩擦/驱动随机化不能只改 import 即得到等价训练。这里先保留 mjlab 官方奖励和随机化；需要复现实验时应逐项对齐并另开 profile。

两边虽都有 29 个关节，数组顺序未必相同。CSV 按资料中列出的 SDK 电机名称顺序解释，转换后按安装模型的 `joint_names` 保存。导出时单独记录自然关节顺序与动作目标顺序，不能把 rl_sar 的旧 `joint_mapping` 直接搬过来。

资料中的旧 154 维观测和策略 `.pt` 不作为新策略输入契约。运行时从实际 observation manager 记录项名、维度、历史与缩放，从实际 action manager 记录 scale 和 offset，包含策略学到的观测归一化。

旧 Isaac NPZ 可能按另一套关节/刚体顺序存储，而 mjlab 的 MotionLoader 不会自动按文件名字重排，也不按 fps 字段改变播放速度。因此本空间要求先转换 CSV，再检查名字顺序、数组形状、有限值和控制频率，随后才能训练。

## 建议的实验顺序

1. 保留你自己的旧 g1_continue 作为比较对象；先用同一评估长度和回放检查。
2. 选择平地基线，先测并行，再训练和评估。改并行数后注意有效样本数变了。
3. 另起 dance_102 跟踪实验，完成转换、4 环境短测、任务专用并行测试。
4. 只对已经能复现的基线做单参数对比。避免同时改奖励、网络、频率和驱动。
5. 导出后使用 TorchScript 仿真回归，再考虑独立 sim2sim 或硬件接入。

舞蹈跟踪完成不等于速度控制行走通过，二者的命令、观测和评价标准不同。

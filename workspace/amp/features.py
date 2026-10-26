"""专家和仿真共享的 AMP 特征；所有四元数均为 wxyz。

每帧 83 维：29 关节角 + 29 关节速度 + 根高度 + 去航向的 6D 姿态
+ 去航向的根线/角速度（各 3）+ 四个末端相对根的位置（共 12）。
判别器使用相邻两帧，输入 166 维；不含时间、动作编号或参考目标，
因此学习的是动作分布，不要求与某段参考轨迹逐帧同步。
"""
import torch

KEY_BODIES = ('left_ankle_roll_link', 'right_ankle_roll_link',
              'left_wrist_yaw_link', 'right_wrist_yaw_link')
FEATURE_VERSION = 'g1_heading_invariant_v1'
FEATURE_DIM = 83


def rotate(q, v):
    """用单位四元数旋转向量；支持前导批量维度广播。"""
    xyz = q[..., 1:].expand_as(v)
    uv = torch.linalg.cross(xyz, v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(xyz, uv))


def build_features(joints, joint_vel, root_pos, root_quat, root_lin_vel,
                   root_ang_vel, key_positions):
    """将两种来源的同名、同序物理量转换为相同的世界位置/航向不变特征。

    root_pos 的 z 已减去各环境原点高度；key_positions 和 root_pos 必须
    属于同一个坐标系。根部全局 xy 不入网络，避免判别器按出生位置作弊。
    """
    w, x, y, z = root_quat.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    inverse_heading = torch.stack((torch.cos(yaw/2), torch.zeros_like(yaw),
                                   torch.zeros_like(yaw), -torch.sin(yaw/2)), -1)
    axis_x = torch.zeros_like(root_pos)
    axis_x[..., 0] = 1
    axis_z = torch.zeros_like(root_pos)
    axis_z[..., 2] = 1
    tangent = rotate(inverse_heading, rotate(root_quat, axis_x))
    normal = rotate(inverse_heading, rotate(root_quat, axis_z))
    offsets = rotate(inverse_heading.unsqueeze(-2), key_positions-root_pos.unsqueeze(-2))
    return torch.cat((joints, joint_vel, root_pos[..., 2:3], tangent, normal,
                      rotate(inverse_heading, root_lin_vel),
                      rotate(inverse_heading, root_ang_vel), offsets.flatten(-2)), -1)


def make_transition(current, following):
    """按 [当前帧, 下一帧] 排列，专家与策略必须使用相同的时间方向。"""
    return torch.cat((current, following), dim=-1)

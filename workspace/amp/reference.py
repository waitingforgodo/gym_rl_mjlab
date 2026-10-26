"""从标准 G1 NPZ 加载 AMP 专家库，保存连续帧而非随机拼接的伪动作。"""
import numpy as np
import torch

from ..motion import inspect
from .features import KEY_BODIES, build_features, make_transition


class AmpReference:
    """一次加载小型动作库，供参考状态初始化和判别器独立采样使用。"""

    def __init__(self, path, device='cpu'):
        inspect(path)
        with np.load(path, allow_pickle=False) as data:
            names = data['body_names'].tolist()
            root = names.index('pelvis')
            keys = [names.index(name) for name in KEY_BODIES]
            self.arrays = {name: torch.tensor(data[name], device=device, dtype=torch.float32)
                           for name in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
                                        'body_lin_vel_w', 'body_ang_vel_w')}
            self.fps = float(data['fps'])
        self.root_index = root
        self.frames = len(self.arrays['joint_pos'])
        a = self.arrays
        self.features = build_features(a['joint_pos'], a['joint_vel'], a['body_pos_w'][:, root],
                                       a['body_quat_w'][:, root], a['body_lin_vel_w'][:, root],
                                       a['body_ang_vel_w'][:, root], a['body_pos_w'][:, keys])
        self.transitions = make_transition(self.features[:-1], self.features[1:])
        # 固定专家统计，策略和专家共用，避免训练中归一化漂移改变风格奖励。
        # 最小标准差防止专家几乎不动的维度被放大成数千倍。
        self.mean = self.features.mean(0)
        self.std = self.features.std(0, unbiased=False).clamp_min(0.1)

    def sample(self, count):
        """均匀采样完整相邻帧，绝不把动作末帧与首帧接成未经验证的循环。"""
        ids = torch.randint(len(self.transitions), (count,), device=self.transitions.device)
        return self.transitions[ids]

    def normalize(self, transitions):
        """两帧使用相同的逐特征标准化；截断极端状态以稳定判别器。"""
        return ((transitions.reshape(-1, 2, self.features.shape[-1])-self.mean)/self.std).clamp(-10, 10).flatten(1)

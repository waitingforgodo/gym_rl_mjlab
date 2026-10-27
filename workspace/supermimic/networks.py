"""论文网络：可微世界模型、运动目标编码器、确定性策略和 critic。"""
import torch
from torch import nn


def mlp(input_dim, hidden, output_dim, *, final=None):
    """构造 ELU MLP；论文只给宽度，激活沿用 BeyondMimic/mjlab 的 ELU。"""
    layers = []
    last = input_dim
    for width in hidden:
        layers.extend((nn.Linear(last, width), nn.ELU()))
        last = width
    layers.append(nn.Linear(last, output_dim))
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """确定性状态转移，以及 learned reward/observation/termination heads。

    论文允许每个 head 使用解析式或学习网络，但没有公开 learned head 宽度；
    本复现将这项未公开细节显式放入配置。状态用残差预测，改善短步展开。
    """

    def __init__(self, state_dim, action_dim, context_dim, obs_dim, goal_dim,
                 dynamics_hidden, head_hidden):
        super().__init__()
        self.state_dim = state_dim
        context_input = state_dim + action_dim + context_dim
        self.dynamics = mlp(context_input, dynamics_hidden, state_dim)
        self.reward = mlp(state_dim + action_dim + goal_dim, head_hidden, 1)
        # 观测含“上一步动作”和带 bias 的关节角，因此 head 必须看到本步
        # executed action 与 sim context，不能只从物理 q/v 猜测这两项。
        self.observation = mlp(state_dim + action_dim + context_dim, head_hidden, obs_dim)
        self.termination = mlp(state_dim + goal_dim, head_hidden, 1)

    def next_state(self, state, action, context):
        """预测残差并加到当前状态；context_dim=0 时支持空张量。"""
        return state + self.dynamics(torch.cat((state, action, context), dim=-1))

    def heads(self, next_state, action, goal, context):
        """从下一状态预测 r_{t+1}、无噪声 o_{t+1} 和终止 logit。"""
        reward = self.reward(torch.cat((next_state, action, goal), dim=-1)).squeeze(-1)
        observation = self.observation(torch.cat((next_state, action, context), dim=-1))
        termination_logit = self.termination(torch.cat((next_state, goal), dim=-1)).squeeze(-1)
        return reward, observation, termination_logit

    def step(self, state, action, context, goal):
        next_state = self.next_state(state, action, context)
        reward, observation, termination = self.heads(next_state, action, goal, context)
        return next_state, reward, observation, termination


class SuperMimicPolicy(nn.Module):
    """10 帧未来目标编码为 32 维 latent，再与 10 帧本体历史解码动作。"""

    def __init__(self, obs_dim, goal_frame_dim, action_dim, history_length,
                 future_frames, latent_dim, encoder_hidden, decoder_hidden):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_frame_dim = goal_frame_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.future_frames = future_frames
        self.motion_encoder = mlp(goal_frame_dim * future_frames, encoder_hidden,
                                  latent_dim, final=nn.Tanh())
        self.decoder = mlp(obs_dim * history_length + latent_dim, decoder_hidden,
                           action_dim, final=nn.Tanh())

    def forward(self, observation_history, future_goal):
        """输入既接受 [B,H,D]，也接受已展平的 [B,H*D]。"""
        history = observation_history.flatten(1)
        goal = future_goal.flatten(1)
        latent = self.motion_encoder(goal)
        return self.decoder(torch.cat((history, latent), dim=-1))


class SuperMimicCritic(nn.Module):
    """论文中的 V(s,g,xi)，没有 EMA target network。"""

    def __init__(self, state_dim, goal_dim, context_dim, hidden):
        super().__init__()
        self.value = mlp(state_dim + goal_dim + context_dim, hidden, 1)

    def forward(self, state, goal, context):
        return self.value(torch.cat((state, goal, context), dim=-1)).squeeze(-1)


def set_trainable(module, enabled):
    """冻结权重但保留对输入的梯度，用于通过固定世界模型做 BPTT。"""
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)

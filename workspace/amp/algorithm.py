"""在 RSL-RL 5.4.2 PPO 上实现 AMP，不复制或改写第三方 PPO 更新代码。

算法形式参考 skrl AMP：BCE 判别、专家输入梯度惩罚、logit 正则、历史
策略样本回放，风格奖励为 softplus(logit) = -log(1-sigmoid(logit))。
这是当前 mjlab 的独立实现，不是对 skrl/Isaac Lab 的运行时依赖。
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.algorithms import PPO

from .features import FEATURE_DIM, FEATURE_VERSION, make_transition
from .reference import AmpReference


def validate_amp_config(cfg):
    """在分配网络/回放池之前校验规模和系数，拒绝静默忽略的拼写错误。"""
    expected = {'motion_file', 'hidden_dims', 'learning_rate', 'batch_size', 'updates_per_iteration',
                'replay_capacity', 'style_reward_scale', 'task_reward_weight', 'gradient_penalty',
                'logit_regularization', 'weight_decay', 'max_grad_norm', 'control_dt'}
    if set(cfg) != expected:
        raise ValueError(f'AMP configuration keys differ: {set(cfg) ^ expected}')
    for key in ('batch_size', 'updates_per_iteration', 'replay_capacity'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'AMP {key} must be a positive integer')
    if not cfg['hidden_dims'] or any(type(n) is not int or n < 1 for n in cfg['hidden_dims']):
        raise ValueError('AMP hidden_dims must contain positive integers')
    for key in ('learning_rate', 'style_reward_scale', 'task_reward_weight', 'gradient_penalty',
                'logit_regularization', 'weight_decay', 'max_grad_norm', 'control_dt'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'AMP {key} must be finite and nonnegative')
    for key in ('learning_rate', 'style_reward_scale', 'max_grad_norm', 'control_dt'):
        if cfg[key] <= 0:
            raise ValueError(f'AMP {key} must be positive')


class ReplayBuffer:
    """GPU 上的有界环形回放池，仅保存真实相邻状态，不保留计算图。"""

    def __init__(self, capacity, dimension, device):
        self.data = torch.empty((capacity, dimension), device=device)
        self.size = self.cursor = 0

    def insert(self, samples):
        """一次批量写入；超过容量时只保留最新样本。"""
        samples = samples.detach()
        capacity = len(self.data)
        if len(samples) >= capacity:
            self.data.copy_(samples[-capacity:])
            self.size, self.cursor = capacity, 0
            return
        end = min(capacity-self.cursor, len(samples))
        self.data[self.cursor:self.cursor+end].copy_(samples[:end])
        if end < len(samples):
            self.data[:len(samples)-end].copy_(samples[end:])
        self.cursor = (self.cursor+len(samples)) % capacity
        self.size = min(capacity, self.size+len(samples))

    def sample(self, count):
        """只在已写入的范围采样，避免读取 torch.empty 的未初始化内存。"""
        if not self.size:
            raise ValueError('Cannot sample an empty AMP replay buffer')
        return self.data[torch.randint(self.size, (count,), device=self.data.device)]

    def state_dict(self):
        """只序列化有效数据；容量和指针随 checkpoint 恢复。"""
        return dict(data=self.data[:self.size].detach().cpu().clone(), size=self.size,
                    cursor=self.cursor, capacity=len(self.data))

    def load_state_dict(self, state):
        """回放池不兼容时明确报错，不能假装已经完整续训。"""
        if state['capacity'] != len(self.data) or state['data'].shape != (state['size'], self.data.shape[1]):
            raise ValueError('AMP replay shape/capacity differs')
        if not 0 <= state['size'] <= len(self.data) or not 0 <= state['cursor'] < len(self.data):
            raise ValueError('Invalid AMP replay pointers')
        self.size, self.cursor = state['size'], state['cursor']
        self.data[:self.size].copy_(state['data'].to(self.data.device))


class AmpPPO(PPO):
    """actor/critic 仍由官方 PPO 更新，AMP 只扩展奖励、判别器及训练状态。"""

    def __init__(self, actor, critic, storage, amp, **kwargs):
        validate_amp_config(amp)
        if kwargs.get('multi_gpu_cfg'):
            raise ValueError('This AMP implementation currently supports one GPU per training process')
        super().__init__(actor, critic, storage, **kwargs)
        self.amp_cfg = dict(amp)
        self.reference = AmpReference(amp['motion_file'], self.device)
        if self.reference.features.shape[-1] != FEATURE_DIM or not math.isclose(1/self.reference.fps, amp['control_dt']):
            raise ValueError('AMP feature dimension or expert time interval differs')
        layers, width = [], 2*FEATURE_DIM
        for hidden in amp['hidden_dims']:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.discriminator = nn.Sequential(*layers).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                       lr=amp['learning_rate'], weight_decay=amp['weight_decay'])
        self.replay = ReplayBuffer(amp['replay_capacity'], 2*FEATURE_DIM, self.device)
        self.amp_rollout = []
        self.current_amp = None

    def act(self, obs):
        """保存动作施加前的真实状态；actor 的输入组不包含 amp。"""
        self.current_amp = obs['amp'].detach().clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        """先用本轮固定判别器计算奖励，再交给 PPO 计算 GAE/return。

        下一帧必须取自动复位前的捕获值，不能用返回的 post-reset 观测。
        跌倒的风格奖励置零；时间截断继续使用正常风格奖励与 PPO bootstrap。
        rewards 原地更新让官方 logger 同时记录实际用于学习的合成回报。
        """
        if self.current_amp is None or 'amp_terminal_features' not in extras:
            raise RuntimeError('AMP requires pre-action state and pre-reset terminal features')
        transition = make_transition(self.current_amp, extras['amp_terminal_features'].to(self.device))
        if not torch.isfinite(transition).all():
            raise FloatingPointError('Non-finite AMP transition')
        with torch.no_grad():
            logits = self.discriminator(self.reference.normalize(transition)).squeeze(-1)
            style = F.softplus(logits).clamp(max=10)
            time_outs = extras.get('time_outs', torch.zeros_like(dones)).bool()
            style = style.masked_fill(dones.bool() & ~time_outs, 0)
            task_reward = rewards.detach().clone()
            style_reward = self.amp_cfg['style_reward_scale']*self.amp_cfg['control_dt']*style
            rewards.copy_(self.amp_cfg['task_reward_weight']*task_reward+style_reward)
        self.amp_rollout.append(transition.detach())
        # 含 / 的日志名由 RSL-RL 原样写入 TensorBoard，和 Episode_Reward 区分。
        extras.setdefault('log', {}).update({'AMP/style_reward': style_reward.mean(),
                                             'AMP/task_reward': task_reward.mean(),
                                             'AMP/combined_reward': rewards.mean()})
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        """每轮交替更新：PPO 策略优化 + 专家/当前策略/历史策略的判别优化。"""
        samples = torch.cat(self.amp_rollout).detach().clone()
        # super().update() 只清空 PPO storage；AMP 轨迹在本函数结束时清空。
        losses = super().update()
        totals = dict(amp_discriminator=0., amp_gradient_penalty=0.,
                      amp_expert_accuracy=0., amp_policy_accuracy=0.)
        cfg = self.amp_cfg
        for _ in range(cfg['updates_per_iteration']):
            count = cfg['batch_size']
            current_count = (count+1)//2 if self.replay.size else count
            current = samples[torch.randint(len(samples), (current_count,), device=self.device)]
            policy = (torch.cat((current, self.replay.sample(count-current_count)))
                      if current_count < count else current)
            expert = self.reference.normalize(self.reference.sample(count)).detach().clone().requires_grad_(True)
            policy = self.reference.normalize(policy).detach()
            real_logits = self.discriminator(expert).squeeze(-1)
            fake_logits = self.discriminator(policy).squeeze(-1)
            classification = 0.5*(F.softplus(-real_logits).mean()+F.softplus(fake_logits).mean())
            # R1 输入梯度惩罚：对“专家输入”求导，再反传到判别器参数。
            grad = torch.autograd.grad(real_logits.sum(), expert, create_graph=True)[0]
            penalty = grad.square().sum(-1).mean()
            logit_reg = self.discriminator[-1].weight.square().sum()
            loss = classification+cfg['gradient_penalty']*penalty+cfg['logit_regularization']*logit_reg
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite AMP discriminator loss')
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
            self.discriminator_optimizer.step()
            for name, value in (('amp_discriminator', classification), ('amp_gradient_penalty', penalty),
                                ('amp_expert_accuracy', (real_logits > 0).float().mean()),
                                ('amp_policy_accuracy', (fake_logits < 0).float().mean())):
                totals[name] += float(value.detach())/cfg['updates_per_iteration']
        self.replay.insert(samples)
        losses.update(totals, amp_replay_size=float(self.replay.size))
        self.amp_rollout.clear()
        return losses

    def train_mode(self):
        """切换训练模式，判别器和 PPO 模型一并切换。"""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self):
        """推理时冻结模型；风格判别器不参与 actor 的前向推理。"""
        super().eval_mode()
        self.discriminator.eval()

    def save(self):
        """保存真正的 AMP 续训状态：判别器、优化器、归一化、历史回放池。"""
        saved = super().save()
        saved['amp_state'] = dict(feature_version=FEATURE_VERSION,
                                  discriminator=self.discriminator.state_dict(),
                                  optimizer=self.discriminator_optimizer.state_dict(),
                                  mean=self.reference.mean, std=self.reference.std,
                                  replay=self.replay.state_dict())
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        """全量续训必须包含 AMP 状态；actor-only 回放允许跳过训练专用数据。"""
        iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get('amp', False):
            state = loaded_dict['amp_state']
            if state['feature_version'] != FEATURE_VERSION:
                raise ValueError('AMP feature contract differs')
            self.discriminator.load_state_dict(state['discriminator'], strict=strict)
            self.discriminator_optimizer.load_state_dict(state['optimizer'])
            self.reference.mean = state['mean'].to(self.device)
            self.reference.std = state['std'].to(self.device)
            self.replay.load_state_dict(state['replay'])
        return iteration

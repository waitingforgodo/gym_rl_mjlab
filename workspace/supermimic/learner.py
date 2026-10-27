"""SuperMimic 的三类更新：世界模型、BPTT actor、GAE critic。"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from .networks import SuperMimicCritic, SuperMimicPolicy, WorldModel, set_trainable


@contextmanager
def frozen(*modules):
    """临时冻结参数，但不使用 no_grad，因此仍能求输出对状态/动作的导数。"""
    old = [[p.requires_grad for p in module.parameters()] for module in modules]
    for module in modules:
        set_trainable(module, False)
    try:
        yield
    finally:
        for module, flags in zip(modules, old, strict=True):
            for parameter, flag in zip(module.parameters(), flags, strict=True):
                parameter.requires_grad_(flag)


class SuperMimicLearner:
    """论文算法的可测试实现；仿真采集由 training.py 负责。"""

    def __init__(self, cfg, state_dim, device):
        self.cfg, self.state_dim, self.device = cfg, state_dim, torch.device(device)
        goal_dim = cfg.goal_frame_dim * cfg.future_frames
        self.world = WorldModel(state_dim, cfg.action_dim, cfg.context_dim,
                                cfg.observation_dim, goal_dim,
                                cfg.world_hidden, cfg.head_hidden).to(device)
        self.policy = SuperMimicPolicy(
            cfg.observation_dim, cfg.goal_frame_dim, cfg.action_dim,
            cfg.history_length, cfg.future_frames, cfg.latent_dim,
            cfg.motion_encoder_hidden, cfg.decoder_hidden).to(device)
        self.critic = SuperMimicCritic(
            state_dim, goal_dim, cfg.context_dim, cfg.critic_hidden).to(device)
        self.world_optimizer = torch.optim.Adam(self.world.parameters(), lr=cfg.learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.update_index = 0

    @property
    def sequence_length(self):
        """策略 batch 需要历史、H 个动作以及最后一个 K 帧目标窗口。"""
        return self.cfg.history_length - 1 + self.cfg.imagination_horizon + self.cfg.future_frames

    def learning_rate(self):
        """论文：25 万次内余弦衰减到 0.03 倍，之后保持。"""
        progress = min(self.update_index / self.cfg.lr_decay_iterations, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.learning_rate * (self.cfg.lr_final_fraction +
                                         (1-self.cfg.lr_final_fraction) * cosine)

    def _set_lr(self):
        value = self.learning_rate()
        for optimizer in (self.world_optimizer, self.policy_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group['lr'] = value
        return value

    def fit_world_batch(self, batch):
        """在真实动作上自回归展开 k 步，包含四项论文损失。"""
        state = batch['state'][:, 0]
        context = batch['context'][:, 0]
        state_loss = reward_loss = observation_loss = termination_loss = state.new_zeros(())
        for step in range(self.cfg.world_window):
            action = batch['action'][:, step]
            # reward/done head的 goal 输入是从当前帧开始的 K 帧窗口；模型拟合
            # 阶段只有连续当前 goal，因此不足部分重复窗口末帧。
            end = min(step + self.cfg.future_frames, batch['goal'].shape[1])
            window = batch['goal'][:, step:end]
            if window.shape[1] < self.cfg.future_frames:
                window = torch.cat((window, window[:, -1:].expand(
                    -1, self.cfg.future_frames-window.shape[1], -1)), dim=1)
            goal = window.flatten(1)
            state = self.world.next_state(state, action, context)
            reward, observation, done_logit = self.world.heads(state, action, goal, context)
            state_loss = state_loss + F.mse_loss(state, batch['next_state'][:, step])
            reward_loss = reward_loss + F.mse_loss(reward, batch['reward'][:, step].reshape(-1))
            observation_loss = observation_loss + F.mse_loss(
                observation, batch['clean_next_observation'][:, step])
            termination_loss = termination_loss + F.binary_cross_entropy_with_logits(
                done_logit, batch['terminated'][:, step].float())
        scale = 1.0 / self.cfg.world_window
        total = (state_loss + reward_loss + observation_loss + termination_loss) * scale
        self.world_optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad = nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.max_grad_norm)
        self.world_optimizer.step()
        return dict(world_loss=float(total.detach()), world_state_loss=float(state_loss.detach()*scale),
                    world_reward_loss=float(reward_loss.detach()*scale),
                    world_observation_loss=float(observation_loss.detach()*scale),
                    world_termination_loss=float(termination_loss.detach()*scale),
                    world_grad_norm=float(grad))

    def fit_world(self, replay):
        """论文每个世界模型 update 顺序处理 10×2048 个 k-step windows。"""
        totals = {}
        # 多取 K-1 个 goal 帧，使每个转移都能形成未来窗口。
        length = self.cfg.world_window + self.cfg.future_frames - 1
        for _ in range(self.cfg.world_batches_per_update):
            batch = replay.sample(length, self.cfg.world_batch_size, self.device)
            metrics = self.fit_world_batch(batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        return {key: value/self.cfg.world_batches_per_update for key, value in totals.items()}

    def _imagine(self, batch, require_policy_grad):
        """从 replay state 出发，用固定世界模型展开 H 步。"""
        cfg = self.cfg
        start = cfg.history_length - 1
        state = batch['state'][:, start]
        context = batch['context'][:, start]
        history = batch['observation'][:, :cfg.history_length]
        survival = torch.ones(state.shape[0], device=state.device)
        discount = 1.0
        objective = state.new_zeros(state.shape[0])
        critic_rows = []
        for step in range(cfg.imagination_horizon):
            goal = batch['goal'][:, start+step:start+step+cfg.future_frames].flatten(1)
            if require_policy_grad:
                action = self.policy(history, goal)
            else:
                with torch.no_grad():
                    action = self.policy(history, goal)
            next_state, reward, next_obs, done_logit = self.world.step(
                state, action, context, goal)
            done = torch.sigmoid(done_logit)
            objective = objective + discount * survival * reward
            critic_rows.append((state.detach(), goal.detach(), context.detach(),
                                reward.detach(), done.detach()))
            survival = survival * (1.0-done)
            discount *= cfg.gamma
            # 世界模型预测无噪声观测；论文在模型外重新采样已知传感噪声。
            if cfg.observation_dim == 93:
                widths = next_obs.new_tensor(
                    [.01]*29 + [.5]*29 + [.2]*3 + [.05]*3 + [0.]*29)
                next_policy_obs = next_obs + (torch.rand_like(next_obs)*2-1)*widths
            else:  # 小维度 CPU 单元测试不伪造不存在的观测布局。
                next_policy_obs = next_obs
            history = torch.cat((history[:, 1:], next_policy_obs[:, None]), dim=1)
            state = next_state
        terminal_goal = batch['goal'][:, start+cfg.imagination_horizon:
                                      start+cfg.imagination_horizon+cfg.future_frames].flatten(1)
        objective = objective + discount * survival * self.critic(state, terminal_goal, context)
        return objective, critic_rows, (state.detach(), terminal_goal.detach(), context.detach())

    def _critic_targets(self, rows, terminal):
        """按论文式 (4) 一次性计算固定 GAE return targets。"""
        with torch.no_grad():
            terminal_value = self.critic(*terminal)
            target = terminal_value
            targets = []
            next_value = terminal_value
            for state, goal, context, reward, done in reversed(rows):
                value = self.critic(state, goal, context)
                target = reward + self.cfg.gamma*(1.0-done)*(
                    (1.0-self.cfg.gae_lambda)*next_value + self.cfg.gae_lambda*target)
                targets.append(target)
                next_value = value
            targets.reverse()
            states = torch.cat([row[0] for row in rows])
            goals = torch.cat([row[1] for row in rows])
            contexts = torch.cat([row[2] for row in rows])
            returns = torch.cat(targets)
        return states, goals, contexts, returns

    def improve_policy(self, replay):
        """总 batch=4096；用 microbatch 累积 BPTT 梯度以适应 8 GiB GPU。"""
        cfg = self.cfg
        self.policy_optimizer.zero_grad(set_to_none=True)
        all_rows, terminals = [], []
        objective_total = 0.0
        micro_count = cfg.imagination_batch_size // cfg.imagination_microbatch
        with frozen(self.world, self.critic):
            self.world.eval()
            for _ in range(micro_count):
                batch = replay.sample(self.sequence_length, cfg.imagination_microbatch, self.device,
                                      allow_final_boundary=False)
                objective, rows, terminal = self._imagine(batch, require_policy_grad=True)
                (-objective.mean()/micro_count).backward()
                objective_total += float(objective.detach().mean()) / micro_count
                all_rows.append(rows)
                terminals.append(terminal)
        policy_grad = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(-cfg.policy_element_clip, cfg.policy_element_clip)
        self.policy_optimizer.step()
        self.world.train()

        # 将各 microbatch 在同一时间步的数据拼起来，再按 B×H 展平训练 critic。
        merged_rows = []
        for step in range(cfg.imagination_horizon):
            columns = [rows[step] for rows in all_rows]
            merged_rows.append(tuple(torch.cat(items) for items in zip(*columns, strict=True)))
        terminal = tuple(torch.cat(items) for items in zip(*terminals, strict=True))
        states, goals, contexts, returns = self._critic_targets(merged_rows, terminal)
        count = states.shape[0]
        minibatch = count // cfg.critic_minibatches
        critic_loss_total, critic_grad_total, updates = 0.0, 0.0, 0
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(count, device=self.device)
            for begin in range(0, count, minibatch):
                ids = order[begin:begin+minibatch]
                loss = F.mse_loss(self.critic(states[ids], goals[ids], contexts[ids]), returns[ids])
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()
                critic_loss_total += float(loss.detach())
                critic_grad_total += float(grad)
                updates += 1
        self.update_index += 1
        lr = self._set_lr()
        return dict(policy_objective=objective_total, policy_grad_norm=float(policy_grad),
                    critic_loss=critic_loss_total/updates, critic_grad_norm=critic_grad_total/updates,
                    learning_rate=lr)

    def state_dict(self):
        return dict(update_index=self.update_index, state_dim=self.state_dim,
                    world=self.world.state_dict(), policy=self.policy.state_dict(),
                    critic=self.critic.state_dict(), world_optimizer=self.world_optimizer.state_dict(),
                    policy_optimizer=self.policy_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict())

    def load_state_dict(self, state, training=True):
        if state['state_dim'] != self.state_dim:
            raise ValueError('SuperMimic world-state dimension differs')
        self.world.load_state_dict(state['world'])
        self.policy.load_state_dict(state['policy'])
        self.critic.load_state_dict(state['critic'])
        self.update_index = state['update_index']
        if training:
            self.world_optimizer.load_state_dict(state['world_optimizer'])
            self.policy_optimizer.load_state_dict(state['policy_optimizer'])
            self.critic_optimizer.load_state_dict(state['critic_optimizer'])

"""论文 50/40/10 return-based motion reset sampler。"""
import types

import torch


def motion_done(env, command_name='motion'):
    """在参考最后一帧终止，防止 command wrap 的瞬时传送污染世界模型。"""
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total-2


def mixture_probabilities(returns, past_weights, maximum_return):
    """50% uniform + 40% softmax(-25 G/Gmax) + 10% 历史 softmax。"""
    count = returns.numel()
    uniform = torch.full_like(returns, 1.0/count)
    hard = torch.softmax(-25.0*returns/maximum_return, dim=0)
    past = past_weights/past_weights.sum().clamp_min(1e-12)
    return .5*uniform + .4*hard + .1*past


class ReturnResetSampler:
    """按起始 1 秒 bin 的 episode 折扣回报维护困难度。

    论文未给 running estimate 的更新率；本复现显式使用 EMA=0.01。
    Gmax 取正奖励理论上限 `5*dt/(1-gamma)`（六个 tracking exp 权重之和）。
    """

    def __init__(self, env, gamma, ema=0.01):
        self.env = env
        self.command = env.command_manager.get_term('motion')
        self.gamma, self.ema = gamma, ema
        count = self.command.bin_count
        self.returns = torch.zeros(count, device=env.device)
        self.seen = torch.zeros(count, dtype=torch.bool, device=env.device)
        self.past_weights = torch.ones(count, device=env.device)
        self.episode_return = torch.zeros(env.num_envs, device=env.device)
        self.discount = torch.ones(env.num_envs, device=env.device)
        self.start_bin = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.start_bin[:] = torch.clamp(
            (self.command.time_steps*self.command.bin_count)//
            max(self.command.motion.time_step_total, 1), 0, count-1)
        self.maximum_return = 5.0*env.step_dt/(1.0-gamma)

        def sample(term, env_ids):
            self._sample(env_ids)
        self.command._adaptive_sampling = types.MethodType(sample, self.command)

    def _sample(self, env_ids):
        probability = mixture_probabilities(
            self.returns, self.past_weights, self.maximum_return)
        current_softmax = torch.softmax(
            -25.0*self.returns/self.maximum_return, dim=0)
        bins = torch.multinomial(probability, len(env_ids), replacement=True)
        fraction = torch.rand(len(env_ids), device=self.env.device)
        self.command.time_steps[env_ids] = (
            (bins+fraction)/self.command.bin_count
            * (self.command.motion.time_step_total-1)).long()
        self.start_bin[env_ids] = bins
        self.past_weights += current_softmax
        entropy = -(probability*(probability+1e-12).log()).sum()
        normalizer = torch.log(torch.as_tensor(
            float(self.command.bin_count), device=self.env.device)).clamp_min(1.)
        self.command.metrics['sampling_entropy'][:] = entropy/normalizer
        value, index = probability.max(dim=0)
        self.command.metrics['sampling_top1_prob'][:] = value
        self.command.metrics['sampling_top1_bin'][:] = index.float()/self.command.bin_count

    def record(self, reward, done):
        """在 reset 前记入已完成 episode；未完成环境继续累计折扣回报。"""
        self.episode_return += self.discount*reward.detach()
        self.discount *= self.gamma
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if not len(ids):
            return
        bins = self.start_bin[ids]
        for bin_index in bins.unique():
            values = self.episode_return[ids][bins == bin_index]
            estimate = values.mean()
            index = int(bin_index)
            if self.seen[index]:
                self.returns[index].lerp_(estimate, self.ema)
            else:
                self.returns[index] = estimate
                self.seen[index] = True
        self.episode_return[ids] = 0
        self.discount[ids] = 1

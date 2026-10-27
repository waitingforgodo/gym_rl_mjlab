"""mjlab 状态与论文 SuperMimic 张量接口之间的唯一转换层。"""
import torch


class SuperMimicAdapter:
    """构造 93 维观测、未来运动目标、privileged state 与 context。

    论文没有公开 world state 的精确定义；本复现显式采用 mjlab 的 critic
    observation 作为 privileged state。context 为 29 个编码器偏置、躯干
    当前 COM 参数和脚部切向摩擦，均直接读取当前随机化后的模型。
    """

    def __init__(self, env, cfg, *, noisy):
        self.env, self.cfg, self.noisy = env, cfg, noisy
        self.robot = env.scene['robot']
        self.motion = env.command_manager.get_term('motion')
        self.history = None
        self._torso_id = self.robot.indexing.body_ids[
            self.robot.body_names.index('torso_link')]
        foot = env.cfg.events.get('foot_friction')
        selected = foot.params['asset_cfg'].geom_ids if foot is not None else None
        if isinstance(selected, slice):
            self._friction_id = self.robot.indexing.geom_ids[0]
        elif isinstance(selected, (list, tuple)):
            self._friction_id = selected[0]
        else:
            self._friction_id = selected.flatten()[0]

    def observation(self, noisy=None):
        """顺序严格按论文表 3；训练时才加入论文规定的均匀传感噪声。"""
        d = self.robot.data
        parts = (d.joint_pos_biased-d.default_joint_pos, d.joint_vel,
                 d.root_link_ang_vel_b, d.projected_gravity_b,
                 self.env.action_manager.action)
        result = torch.cat(parts, dim=-1)
        if self.noisy if noisy is None else noisy:
            widths = result.new_tensor([.01]*29 + [.5]*29 + [.2]*3 + [.05]*3 + [0.]*29)
            result = result + (torch.rand_like(result)*2-1)*widths
        if result.shape[1] != self.cfg.observation_dim:
            raise RuntimeError(f'Expected {self.cfg.observation_dim} proprioception values, got {result.shape[1]}')
        return result

    def state(self, obs_dict=None):
        """71 维 Markov 状态：根 q/v 与关节 q/v，不把参考目标混入动力学。"""
        del obs_dict
        d = self.robot.data
        return torch.cat((d.root_link_pos_w-self.env.scene.env_origins,
                          d.root_link_quat_w, d.root_link_lin_vel_w,
                          d.root_link_ang_vel_w, d.joint_pos, d.joint_vel), dim=-1)

    def _goal_at(self, indices):
        """完整参考帧：root pose/velocity + joint pose/velocity，共 71 维。"""
        motion = self.motion.motion
        root = self.motion.motion_anchor_body_index
        return torch.cat((motion.body_pos_w[indices, root],
                          motion.body_quat_w[indices, root],
                          motion.body_lin_vel_w[indices, root],
                          motion.body_ang_vel_w[indices, root],
                          motion.joint_pos[indices], motion.joint_vel[indices]), dim=-1)

    def goal(self):
        """返回当前参考帧 q/qdot；连续 K 帧窗口由 learner/replay 组合。"""
        indices = self.motion.time_steps.clamp_max(self.motion.motion.time_step_total-1)
        return self._goal_at(indices)

    def context(self):
        """29 bias + torso COM(3) + foot friction(1) = 33。"""
        model = self.env.sim.model
        com = model.body_ipos[:, self._torso_id]
        friction = model.geom_friction[:, self._friction_id, 0:1]
        result = torch.cat((self.robot.data.encoder_bias, com, friction), dim=-1)
        if result.shape[1] != self.cfg.context_dim:
            raise RuntimeError(f'Expected {self.cfg.context_dim} context values, got {result.shape[1]}')
        return result

    def initialize_history(self, observation=None):
        observation = self.observation() if observation is None else observation
        self.history = observation[:, None].expand(-1, self.cfg.history_length, -1).clone()
        return self.history

    def advance_history(self, observation, reset_ids=None):
        self.history = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
        if reset_ids is not None and len(reset_ids):
            self.history[reset_ids] = observation[reset_ids, None].expand(
                -1, self.cfg.history_length, -1)
        return self.history

    def future_goal(self):
        """直接按当前 motion time index 读取 K 帧，结尾 clamp 不循环。"""
        offsets = torch.arange(self.cfg.future_frames, device=self.env.device)
        indices = (self.motion.time_steps[:, None]+offsets).clamp_max(
            self.motion.motion.time_step_total-1)
        return self._goal_at(indices)

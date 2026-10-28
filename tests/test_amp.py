"""AMP 的算法与数据回归：验证判别器真正更新、复位边界和续训状态。"""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from workspace.common.contract import SDK_JOINTS, validate_resume


@unittest.skipUnless(importlib.util.find_spec('mjlab'), 'requires pinned training packages; no GPU needed')
class AmpTests(unittest.TestCase):
    """CPU 上执行真实 Torch/PPO 优化，不依赖本地大数据或仿真资源。"""

    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        from workspace.amp.features import KEY_BODIES
        frames, bodies = 9, ['pelvis', *KEY_BODIES]
        q = np.zeros((frames, 5, 4), np.float32)
        q[..., 0] = 1
        joint_pos = np.arange(frames, dtype=np.float32)[:, None]*np.ones((1, 29), np.float32)*.01
        pos = np.zeros((frames, 5, 3), np.float32)
        pos[:, :, 2] = .8
        self.path = self.folder/'expert.npz'
        np.savez(self.path, fps=50., joint_names=np.array(SDK_JOINTS), body_names=np.array(bodies),
                 joint_pos=joint_pos, joint_vel=np.ones_like(joint_pos)*.5,
                 body_pos_w=pos, body_quat_w=q, body_lin_vel_w=np.zeros_like(pos),
                 body_ang_vel_w=np.zeros_like(pos))

    def algorithm(self):
        """利用官方 construct_algorithm 构建小型真实 PPO，检验扩展接口兼容性。"""
        import torch
        from tensordict import TensorDict
        from workspace.amp.tasks import register, TASK_ID
        from workspace.amp.algorithm import AmpPPO
        from mjlab.tasks.registry import load_rl_cfg
        register()
        cfg = asdict(load_rl_cfg(TASK_ID))
        cfg.update(num_steps_per_env=4, multi_gpu=None)
        for key in ('actor', 'critic'):
            cfg[key]['hidden_dims'] = (32, 16)
            for opt in ('cnn_cfg', 'distribution_cfg', 'rnn_type', 'rnn_hidden_dim', 'rnn_num_layers'):
                if cfg[key].get(opt) is None or opt.startswith('rnn_'):
                    cfg[key].pop(opt, None)
        cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=2)
        cfg['algorithm']['amp'].update(motion_file=str(self.path), hidden_dims=[32, 16],
                                       batch_size=16, updates_per_iteration=2, replay_capacity=32)
        obs = TensorDict({'actor': torch.randn(4, 96), 'critic': torch.randn(4, 96),
                          'amp': torch.randn(4, 83)}, batch_size=[4])
        alg = AmpPPO.construct_algorithm(obs, SimpleNamespace(num_actions=29, num_envs=4), cfg, 'cpu')
        return alg, obs

    def rollout(self, alg, obs):
        """模拟四步，包括真实终止帧与明显不同的复位后观测。"""
        import torch
        for _ in range(4):
            with torch.inference_mode():
                alg.act(obs)
                terminal = torch.full((4, 83), .3)
                rewards = torch.ones(4)*.01
                dones = torch.tensor([1, 1, 0, 0])
                extras = {'amp_terminal_features': terminal, 'time_outs': torch.tensor([0, 1, 0, 0])}
                alg.process_env_step(obs, rewards, dones, extras)
                # 第一个失败环境的 style=0，但超时环境仍有风格奖励。
                self.assertAlmostEqual(float(rewards[0]), .01, places=6)
                self.assertGreater(float(rewards[1]), .01)
                self.assertTrue(torch.equal(alg.amp_rollout[-1][:, 83:], terminal))
        with torch.inference_mode():
            alg.compute_returns(obs)

    def test_features_ignore_world_xy_heading_and_quaternion_sign(self):
        import torch
        from workspace.amp.features import build_features, rotate
        joints, vel = torch.randn(3, 29), torch.randn(3, 29)
        pos, lin, ang = torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)
        keys = torch.randn(3, 4, 3)
        q = torch.tensor([[1., 0, 0, 0]]).repeat(3, 1)
        yaw = torch.tensor([[.70710678, 0., 0., .70710678]]).repeat(3, 1)
        translation = torch.tensor([10., -7., 0.])
        first = build_features(joints, vel, pos, q, lin, ang, keys)
        transformed = build_features(joints, vel, rotate(yaw, pos)+translation, yaw,
                                     rotate(yaw, lin), rotate(yaw, ang),
                                     rotate(yaw[:, None], keys)+translation)
        signed = build_features(joints, vel, pos, -q, lin, ang, keys)
        self.assertEqual(first.shape, (3, 83))
        self.assertTrue(torch.allclose(first, transformed, atol=2e-5))
        self.assertTrue(torch.allclose(first, signed))

    def test_expert_pairs_are_contiguous_without_end_to_start_wrap(self):
        import torch
        from workspace.amp.reference import AmpReference
        ref = AmpReference(self.path)
        self.assertEqual(ref.transitions.shape, (8, 166))
        self.assertTrue(torch.allclose(ref.transitions[:, 83:112]-ref.transitions[:, :29],
                                        torch.full((8, 29), .01)))
        self.assertTrue(torch.isfinite(ref.normalize(ref.sample(32))).all())

    def test_terminal_capture_refreshes_and_copies_state_before_reset(self):
        import torch
        from unittest.mock import Mock
        from workspace.amp.environment import capture_terminal_features
        state = torch.ones(4, 83)
        env = SimpleNamespace(sim=SimpleNamespace(forward=Mock()), extras={})
        with patch('workspace.amp.environment.amp_features', return_value=state):
            capture_terminal_features(env)
        state.zero_()  # 模拟后续自动复位覆盖底层数组。
        env.sim.forward.assert_called_once()
        self.assertTrue(torch.equal(env.extras['amp_terminal_features'], torch.ones(4, 83)))

    def test_replay_wrap_restore_and_overflow_only_keep_valid_rows(self):
        import torch
        from workspace.amp.algorithm import ReplayBuffer
        replay = ReplayBuffer(5, 2, 'cpu')
        with self.assertRaises(ValueError):
            replay.sample(1)
        replay.insert(torch.arange(6.).reshape(3, 2))
        replay.insert(torch.arange(6., 14.).reshape(4, 2))
        self.assertEqual((replay.size, replay.cursor), (5, 2))
        restored = ReplayBuffer(5, 2, 'cpu')
        restored.load_state_dict(replay.state_dict())
        self.assertTrue(torch.equal(restored.data, replay.data))
        self.assertEqual(restored.cursor, replay.cursor)
        replay.insert(torch.arange(20.).reshape(10, 2))
        self.assertTrue(torch.equal(replay.data, torch.arange(10., 20.).reshape(5, 2)))

    def test_real_ppo_and_discriminator_update_and_checkpoint_roundtrip(self):
        import torch
        alg, obs = self.algorithm()
        actor_before = [p.detach().clone() for p in alg.actor.parameters()]
        disc_before = [p.detach().clone() for p in alg.discriminator.parameters()]
        self.rollout(alg, obs)
        losses = alg.update()
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, alg.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(disc_before, alg.discriminator.parameters())))
        self.assertGreater(losses['amp_gradient_penalty'], 0)
        self.assertEqual(alg.replay.size, 16)
        checkpoint = self.folder/'state.pt'
        torch.save(alg.save(), checkpoint)
        restored, _ = self.algorithm()
        restored.load(torch.load(checkpoint, weights_only=False), None, True)
        for a, b in zip(alg.discriminator.parameters(), restored.discriminator.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(alg.reference.std, restored.reference.std))
        self.assertTrue(torch.equal(alg.replay.data[:16], restored.replay.data[:16]))
        self.assertEqual(alg.replay.cursor, restored.replay.cursor)
        self.assertEqual(len(alg.discriminator_optimizer.state), len(restored.discriminator_optimizer.state))
        # actor-only 模型回放不要求加载 discriminator/replay。
        restored.load({'actor_state_dict': alg.save()['actor_state_dict']}, {'actor': True}, True)

    def test_config_has_no_reference_targets_and_invalid_amp_settings_fail(self):
        from workspace.amp.tasks import register, TASK_ID
        from workspace.amp.algorithm import validate_amp_config
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
        register()
        cfg = load_env_cfg(TASK_ID)
        self.assertEqual(cfg.commands, {})
        self.assertFalse(any('motion' in name or 'progress' in name for name in cfg.observations['actor'].terms))
        self.assertFalse(any('motion' in name for name in cfg.rewards))
        self.assertEqual(cfg.sim.mujoco.timestep*cfg.decimation, .02)
        amp = load_rl_cfg(TASK_ID).algorithm.amp
        for key, value in [('batch_size', 0), ('gradient_penalty', -1), ('learning_rate', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_amp_config({**amp, key: value})
        with self.assertRaises(ValueError):
            validate_amp_config({**amp, 'typo': 1})

    def test_conversion_maps_joint_names_and_preserves_wxyz_root(self):
        from workspace.amp.conversion import source_to_csv_arrays
        names = list(reversed(SDK_JOINTS))
        positions = np.arange(29, dtype=float)[None, :].repeat(3, 0)
        root = np.zeros((3, 1, 3))
        rotations = np.tile([1., 0., 0., 0.], (3, 1, 1))
        source = self.folder/'source.npz'
        np.savez(source, fps=30., dof_names=names, body_names=['pelvis'],
                 dof_positions=positions, body_positions=root, body_rotations=rotations)
        csv, fps = source_to_csv_arrays(source)
        self.assertEqual(fps, 30.)
        self.assertTrue(np.array_equal(csv[0, 7:], np.arange(28, -1, -1)))
        self.assertTrue(np.array_equal(csv[0, 3:7], [0., 0., 0., 1.]))
        np.savez(source, fps=-1., dof_names=names, body_names=['pelvis'],
                 dof_positions=positions, body_positions=root, body_rotations=rotations)
        with self.assertRaises(ValueError):
            source_to_csv_arrays(source)

    def test_resume_rejects_amp_code_or_expert_change(self):
        saved = dict(amp_source_sha256='original', motion_sha256='dance')
        for current in [dict(saved, amp_source_sha256='changed'), dict(saved, motion_sha256='other')]:
            with self.assertRaises(ValueError):
                validate_resume(saved, current)


if __name__ == '__main__':
    unittest.main()

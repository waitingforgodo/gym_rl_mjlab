"""SuperMimic 核心算法 CPU 回归，不启动 MuJoCo/GPU。"""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class SuperMimicTests(unittest.TestCase):
    def config(self):
        from workspace.supermimic.config import SuperMimicConfig
        return SuperMimicConfig(
            num_envs=4, iterations=2, action_dim=3, observation_dim=6,
            goal_frame_dim=4, context_dim=2, history_length=2, future_frames=2,
            latent_dim=3, world_hidden=(16, 16), motion_encoder_hidden=(12,),
            decoder_hidden=(12,), critic_hidden=(8,), head_hidden=(8,),
            world_window=2, imagination_horizon=3, world_batch_size=4,
            world_batches_per_update=1, imagination_batch_size=4,
            imagination_microbatch=2, replay_steps_per_env=20,
            critic_epochs=1, critic_minibatches=2, save_interval=1)

    def replay(self, cfg, boundary=None):
        import torch
        from workspace.supermimic.replay import SequenceReplay
        replay = SequenceReplay(20, cfg.num_envs, storage_dtype=torch.float32)
        state = torch.randn(cfg.num_envs, 5)
        for step in range(12):
            next_state = state + .01*torch.randn_like(state)
            done = torch.zeros(cfg.num_envs, dtype=torch.bool)
            if boundary == step:
                done[0] = True
            replay.append(state=state, next_state=next_state,
                          observation=torch.randn(cfg.num_envs, cfg.observation_dim),
                          next_observation=torch.randn(cfg.num_envs, cfg.observation_dim),
                          clean_next_observation=torch.randn(cfg.num_envs, cfg.observation_dim),
                          goal=torch.randn(cfg.num_envs, cfg.goal_frame_dim),
                          action=torch.randn(cfg.num_envs, cfg.action_dim).tanh(),
                          reward=torch.randn(cfg.num_envs), terminated=done,
                          boundary=done, context=torch.randn(cfg.num_envs, cfg.context_dim))
            state = next_state
        return replay

    def test_network_shapes_and_action_bounds(self):
        import torch
        from workspace.supermimic.networks import SuperMimicPolicy
        cfg = self.config()
        policy = SuperMimicPolicy(cfg.observation_dim, cfg.goal_frame_dim,
                                  cfg.action_dim, cfg.history_length,
                                  cfg.future_frames, cfg.latent_dim,
                                  cfg.motion_encoder_hidden, cfg.decoder_hidden)
        action = policy(torch.randn(7, 2, 6), torch.randn(7, 2, 4))
        self.assertEqual(action.shape, (7, 3))
        self.assertLessEqual(float(action.detach().abs().max()), 1.)

    def test_return_reset_sampler_is_normalized_mixture(self):
        import torch
        from workspace.supermimic.reset_sampler import mixture_probabilities
        returns = torch.tensor([0., 2., 4.])
        probability = mixture_probabilities(returns, torch.tensor([1., 2., 1.]), 10.)
        self.assertTrue(torch.allclose(probability.sum(), torch.tensor(1.)))
        self.assertTrue((probability >= .5/3).all())
        self.assertGreater(float(probability[0]), float(probability[2]))

    def test_replay_never_crosses_reset_boundary(self):
        import torch
        cfg = self.config()
        replay = self.replay(cfg, boundary=5)
        generator = torch.Generator().manual_seed(3)
        for _ in range(10):
            sample = replay.sample(4, 8, 'cpu', generator=generator)
            self.assertFalse(sample['boundary'][:, :-1].any())

    def test_world_actor_critic_update_and_checkpoint_roundtrip(self):
        import torch
        from workspace.supermimic.learner import SuperMimicLearner
        cfg, replay = self.config(), self.replay(self.config())
        learner = SuperMimicLearner(cfg, state_dim=5, device='cpu')
        world_before = [p.detach().clone() for p in learner.world.parameters()]
        actor_before = [p.detach().clone() for p in learner.policy.parameters()]
        world_metrics = learner.fit_world(replay)
        frozen_world = [p.detach().clone() for p in learner.world.parameters()]
        policy_metrics = learner.improve_policy(replay)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in
                            (*world_metrics.values(), *policy_metrics.values())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(world_before, frozen_world)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, learner.policy.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(frozen_world, learner.world.parameters())))
        restored = SuperMimicLearner(cfg, state_dim=5, device='cpu')
        restored.load_state_dict(learner.state_dict())
        for a, b in zip(learner.policy.parameters(), restored.policy.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()

"""固定依赖的 CPU 配置集成：检查课程阶段、采样方式和新旧基线隔离。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from workspace.configuration import apply_agent_overrides, validate_ppo_config
from workspace.recipes import apply_recipe

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('mjlab'), 'requires the pinned mjlab runtime, but no GPU')
class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mjlab.tasks  # noqa: F401

    def configs(self, name, play=False):
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
        spec = json.loads((ROOT / 'profiles' / f'{name}.json').read_text())
        cfg, agent = load_env_cfg(spec['task'], play=play), load_rl_cfg(spec['task'])
        apply_agent_overrides(agent, spec.get('agent_overrides', {}))
        validate_ppo_config(agent, spec['num_envs'])
        apply_recipe(cfg, agent, spec.get('recipe'), play)
        return cfg, agent

    def test_new_velocity_profile_keeps_physics_and_adds_three_explicit_costs(self):
        cfg, agent = self.configs('velocity_robust')
        old, _ = self.configs('velocity_flat')
        self.assertEqual(cfg.sim.mujoco.timestep * cfg.decimation, .02)
        self.assertEqual(set(cfg.rewards) - set(old.rewards), {'failure_penalty', 'leg_torque_l2', 'leg_acc_l2'})
        self.assertEqual(len(old.rewards), 14)
        self.assertEqual(agent.actor.distribution_cfg['std_type'], 'log')
        self.assertTrue(agent.actor.obs_normalization)
        self.assertEqual(agent.algorithm.entropy_coef, .008)

    def test_curriculum_uses_control_steps_and_restored_counter(self):
        cfg, agent = self.configs('velocity_robust')
        term = cfg.curriculum['command_vel']
        self.assertEqual([s['step'] for s in term.params['velocity_stages']], [0, 36000, 96000])
        for updates, expected_x in ((0, (-.5, 1.)), (1499, (-.5, 1.)), (1500, (-1., 1.5)), (4000, (-1.5, 2.))):
            command = SimpleNamespace(cfg=cfg.commands['twist'])
            env = SimpleNamespace(common_step_counter=updates * agent.num_steps_per_env,
                                  command_manager=SimpleNamespace(get_term=lambda name: command))
            term.func(env, None, **term.params)
            self.assertEqual(command.cfg.ranges.lin_vel_x, expected_x)

    def test_play_uses_final_velocity_ranges_without_a_training_curriculum(self):
        cfg, _ = self.configs('velocity_robust', play=True)
        self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x, (-1.5, 2.))
        self.assertEqual(cfg.commands['twist'].ranges.lin_vel_y, (-1., 1.))
        self.assertNotIn('command_vel', cfg.curriculum)

    def test_both_tracking_variants_use_adaptive_training_and_start_playback(self):
        for name in ('tracking_robust', 'tracking_no_state_estimation_robust'):
            with self.subTest(profile=name):
                cfg, agent = self.configs(name)
                play, _ = self.configs(name, play=True)
                self.assertEqual(cfg.commands['motion'].sampling_mode, 'adaptive')
                self.assertEqual(play.commands['motion'].sampling_mode, 'start')
                self.assertEqual(agent.save_interval, 500)
                self.assertEqual(agent.algorithm.entropy_coef, .005)
                self.assertEqual('base_lin_vel' in cfg.observations['actor'].terms,
                                 name == 'tracking_robust')

    def test_unknown_recipe_and_wrong_task_fail_instead_of_silent_fallback(self):
        cfg, agent = self.configs('velocity_flat')
        for name in ('typo', 'tracking_robust_v1'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                apply_recipe(cfg, agent, name)


if __name__ == '__main__':
    unittest.main()

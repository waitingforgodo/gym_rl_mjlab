"""配置回归：拒绝无效批量/拼写，并保护续训参数和新实验契约。"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from workspace.common.configuration import (apply_agent_overrides, effective_agent_sha256,
                                     inherited_run_options, validate_ppo_config)
from workspace.common.contract import validate_resume
from workspace.common.profiles import profile_path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Model:
    hidden_dims: tuple = (512, 256, 128)
    obs_normalization: bool = True
    distribution_cfg: dict = field(default_factory=lambda: dict(
        class_name='GaussianDistribution', init_std=1., std_type='scalar'))


@dataclass
class Algorithm:
    schedule: str = 'adaptive'
    optimizer: str = 'adam'
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = .001
    desired_kl: float = .01
    max_grad_norm: float = 1.
    gamma: float = .99
    lam: float = .95
    clip_param: float = .2
    entropy_coef: float = .01


@dataclass
class Agent:
    actor: Model = field(default_factory=Model)
    critic: Model = field(default_factory=Model)
    algorithm: Algorithm = field(default_factory=Algorithm)
    obs_groups: dict = field(default_factory=lambda: {'actor': ('actor',), 'critic': ('critic',)})
    clip_actions: float | None = None
    num_steps_per_env: int = 24
    save_interval: int = 100
    seed: int = 42
    max_iterations: int = 3000


class ConfigurationTests(unittest.TestCase):
    def test_current_training_profiles_all_default_to_4096(self):
        """正式入口使用独立 profile；旧文件无需改写也不会被默认选中。"""
        names = (
            'velocity_flat_4096', 'velocity_gym_ppo_4096', 'velocity_robust_4096',
            'tracking_4096', 'tracking_no_state_estimation_4096',
            'tracking_robust_4096', 'tracking_no_state_estimation_robust_4096',
            'amp_dance_4096',
        )
        for name in names:
            with self.subTest(profile=name):
                profile = json.loads((profile_path(ROOT, name)).read_text(encoding='utf-8'))
                self.assertEqual(profile['num_envs'], 4096)

    def test_nested_dictionary_override_and_tuple_conversion(self):
        agent = Agent()
        apply_agent_overrides(agent, {'actor.distribution_cfg.std_type': 'log',
                                     'actor.hidden_dims': [256, 128], 'algorithm.entropy_coef': .008})
        self.assertEqual(agent.actor.distribution_cfg['std_type'], 'log')
        self.assertEqual(agent.actor.hidden_dims, (256, 128))
        validate_ppo_config(agent, 2048)

    def test_typo_wrong_type_and_nonfinite_rejected_before_simulation(self):
        for overrides in ({'algorithm.entopy_coef': .008}, {'actor.distribution_cfg.unknown': 1},
                          {'num_steps_per_env': 24.5}, {'actor.obs_normalization': 'false'},
                          {'algorithm.entropy_coef': float('nan')}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                apply_agent_overrides(Agent(), overrides)

    def test_invalid_probabilities_or_incomplete_minibatch_rejected(self):
        for field_name, value in (('gamma', 1.1), ('lam', 0.), ('num_mini_batches', 5),
                                  ('entropy_coef', -.1), ('learning_rate', 0.),
                                  ('schedule', 'adapive'), ('optimizer', 'unknown')):
            agent = Agent()
            setattr(agent.algorithm, field_name, value)
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                validate_ppo_config(agent, 8)

    def test_resume_inherits_env_count_and_seed_but_explicit_flags_win(self):
        saved = dict(num_envs=256, seed=7)
        self.assertEqual(inherited_run_options(SimpleNamespace(envs=None, seed=None), saved), (256, 7))
        self.assertEqual(inherited_run_options(SimpleNamespace(envs=512, seed=11), saved), (512, 11))
        self.assertEqual(inherited_run_options(SimpleNamespace(envs=None, seed=None)), (None, 42))

    def test_learning_fingerprint_ignores_budget_but_detects_network_or_ppo_changes(self):
        original = Agent()
        changed = deepcopy(original)
        changed.seed, changed.max_iterations = 7, 10000
        self.assertEqual(effective_agent_sha256(original), effective_agent_sha256(changed))
        changed.algorithm.desired_kl = .02
        self.assertNotEqual(effective_agent_sha256(original), effective_agent_sha256(changed))

    def test_recipe_implementation_and_effective_settings_are_part_of_resume_contract(self):
        saved = dict(recipe='velocity_robust_v1', recipe_sha256='source1', effective_agent_sha256='agent1')
        validate_resume(saved, dict(saved, num_envs=1024))
        for change in (dict(recipe='other'), dict(recipe_sha256='source2'), dict(effective_agent_sha256='agent2')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_resume(saved, dict(saved, **change))
        # 原实验没有 recipe 字段，新代码新增空字段后仍能加载旧模型。
        validate_resume({}, dict(recipe=None, recipe_sha256=None, effective_agent_sha256='new-metadata'))


if __name__ == '__main__':
    unittest.main()

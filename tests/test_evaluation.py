"""评估回归测试：覆盖这次真实日志中容易误解的统计与固定命令分支。"""
import unittest
from types import SimpleNamespace as NS
from workspace.common.evaluation import configure_fixed_velocity, is_direct_metric, survival_statistics, velocity_protocol


def velocity_cfg():
    return NS(commands={'twist': NS(ranges=NS(lin_vel_x=(-1.5, 2.), lin_vel_y=(-1., 1.),
                                            ang_vel_z=(-.7, .7), heading=(-3.14, 3.14)),
        heading_command=True, rel_standing_envs=.1, rel_heading_envs=.3,
        rel_world_envs=.2, rel_forward_envs=.2, init_velocity_prob=.5, resampling_time_range=(3., 8.))})


class EvaluationTests(unittest.TestCase):
    def test_fixed_velocity_disables_every_command_override(self):
        cfg = velocity_cfg()
        configure_fixed_velocity(cfg, [.5, 0., 0.])
        command = cfg.commands['twist']
        self.assertEqual(command.ranges.lin_vel_x, (.5, .5))
        self.assertEqual(command.ranges.lin_vel_y, (0., 0.))
        self.assertEqual(command.ranges.ang_vel_z, (0., 0.))
        self.assertFalse(command.heading_command)
        self.assertIsNone(command.ranges.heading)
        self.assertEqual(sum((command.rel_standing_envs, command.rel_heading_envs,
                              command.rel_world_envs, command.rel_forward_envs, command.init_velocity_prob)), 0.)
        self.assertEqual(velocity_protocol(cfg, [.5, 0, 0])['mode'], 'fixed_body_velocity')

    def test_invalid_or_tracking_command_rejected(self):
        for values in ([.5, 0], [.5, 0, float('nan')]):
            with self.assertRaises(ValueError):
                configure_fixed_velocity(velocity_cfg(), values)
        with self.assertRaises(ValueError):
            configure_fixed_velocity(NS(commands={'motion': object()}), [.5, 0, 0])

    def test_repeated_failure_is_not_another_failed_environment(self):
        counts = [0]*12 + [2, 1, 1, 1]
        result = survival_statistics(counts, 60.)
        self.assertEqual(result['environments_with_failures'], 4)
        self.assertEqual(result['environments_without_failures'], 12)
        self.assertEqual(result['no_failure_fraction_over_horizon'], .75)
        self.assertEqual(result['failure_events_per_environment_minute'], 5/16)
        self.assertEqual(result['total_exposure_environment_seconds'], 960)

    def test_censored_survivors_remain_successful_in_this_finite_window(self):
        result = survival_statistics([0]*16, 60.)
        self.assertEqual(result['no_failure_fraction_over_horizon'], 1.)
        self.assertEqual(result['failure_events_per_environment_minute'], 0.)

    def test_reset_log_snapshots_excluded_from_direct_metrics(self):
        for name in ('Episode_Reward/upright', 'Episode_Termination/fell_over', 'Episode_Metrics/mean_action_acc'):
            self.assertFalse(is_direct_metric(name))
        self.assertTrue(is_direct_metric('Metrics/slip_velocity_mean'))

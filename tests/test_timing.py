"""计时回归：复现系统时间倒退，并验证错误数据不会被正中位数掩盖。"""
import time
import unittest
from types import SimpleNamespace

from workspace.common.telemetry import timing_statistics
from workspace.common.timing import monotonic_runner_clock


class TimingTests(unittest.TestCase):
    """假时钟不依赖真实等待或改系统时钟，测试完成后检查引用恢复。"""
    def test_wall_clock_rollback_cannot_make_elapsed_time_negative(self):
        wall_values, monotonic_values = iter([100., 98.]), iter([20., 20.7])
        clock = SimpleNamespace(time=lambda: next(wall_values), perf_counter=lambda: next(monotonic_values))
        runner_module = SimpleNamespace(time=clock)
        real_time = time.time
        with monotonic_runner_clock(runner_module):
            start = runner_module.time.time()
            stop = runner_module.time.time()
            self.assertAlmostEqual(stop-start, .7)
            self.assertIs(time.time, real_time)
        self.assertIs(runner_module.time, clock)
        self.assertEqual(clock.time(), 100.)
        self.assertEqual(clock.time(), 98.)

    def test_clock_binding_restored_after_training_failure(self):
        runner_module = SimpleNamespace(time=time)
        with self.assertRaisesRegex(RuntimeError, 'training failed'):
            with monotonic_runner_clock(runner_module):
                raise RuntimeError('training failed')
        self.assertIs(runner_module.time, time)

    def test_positive_median_does_not_hide_negative_last_iteration(self):
        fps = {i: 10000. for i in range(30)}
        collect, learn = {i: .6 for i in fps}, {i: .1 for i in fps}
        fps[29], learn[29] = -2299., -1.905
        result = timing_statistics(fps, collect, learn)
        self.assertEqual(result['median_fps'], 10000.)
        self.assertFalse(result['timing_valid'])
        self.assertEqual(result['invalid_timing_steps'], [29])
        self.assertEqual(result['valid_timing_updates'], 24)

    def test_all_three_series_must_be_complete_and_finite(self):
        fps = {i: 10000. for i in range(12)}
        collect, learn = {i: .6 for i in fps}, {i: .1 for i in fps}
        del collect[7]
        learn[10] = float('nan')
        result = timing_statistics(fps, collect, learn)
        self.assertFalse(result['timing_valid'])
        self.assertEqual(result['missing_timing_steps'], [7])
        self.assertEqual(result['invalid_timing_steps'], [10])

    def test_clean_series_accepts_nonzero_resume_indices_and_skips_warmup(self):
        fps = {i: 17000. for i in range(3000, 3010)}
        collect, learn = {i: .6 for i in fps}, {i: .1 for i in fps}
        learn[3000] = -1.  # Warmup excluded.
        result = timing_statistics(fps, collect, learn)
        self.assertTrue(result['timing_valid'])
        self.assertEqual(result['valid_timing_updates'], 5)


if __name__ == '__main__':
    unittest.main()

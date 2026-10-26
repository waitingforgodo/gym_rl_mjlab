"""纯协议回归：数字模型选择、关节映射、续训校验和可信并行推荐。"""
import tempfile
import unittest
from pathlib import Path

from workspace.contract import (SDK_JOINTS, choose_parallelism, joint_permutation,
                                latest_checkpoint, resolve_checkpoint, validate_resume)


class ContractTests(unittest.TestCase):
    """覆盖真实工作流风险，不需要加载第三方模型或启动 GPU。"""
    def test_select_numeric_checkpoint_and_ignore_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('model_9.pt', 'model_100.pt', 'model_99.pt', 'model_best.pt', 'policy.pt'):
                (root / name).touch()
            self.assertEqual(latest_checkpoint(root).name, 'model_100.pt')
            self.assertEqual(resolve_checkpoint(folder, root), root / 'model_100.pt')

    def test_reject_policy_export_as_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.pt'
            path.touch()
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(str(path), path.parent)

    def test_latest_ignores_benchmark_sibling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for parent in ('runs/20260101_training', 'benchmarks/runs/20990101_benchmark'):
                run = root / parent
                run.mkdir(parents=True)
                (run / 'model_20.pt').touch()
            self.assertEqual(resolve_checkpoint('latest', root / 'runs').parent.name, '20260101_training')

    def test_mapping_roundtrip_with_permuted_policy_joints(self):
        source = list(SDK_JOINTS)
        target = source[::2] + source[1::2]
        values = list(range(29))
        permuted = [values[i] for i in joint_permutation(source, target)]
        restored = [permuted[i] for i in joint_permutation(target, source)]
        self.assertEqual(restored, values)

    def test_mapping_rejects_wrong_robot_or_duplicate(self):
        for target in (list(SDK_JOINTS[:-1]), list(SDK_JOINTS[:-1])+[SDK_JOINTS[0]]):
            with self.assertRaises(ValueError):
                joint_permutation(list(SDK_JOINTS), target)

    def test_resume_allows_environment_count_change_only(self):
        saved = dict(task='velocity', agent_overrides={}, profile_sha256='config', motion_sha256=None,
                     num_envs=128, mjlab_version='1.6.0', rsl_rl_version='5.4.2', control_dt=.02)
        validate_resume(saved, dict(saved, num_envs=512))
        for change in (dict(task='tracking'), dict(motion_sha256='new'), dict(profile_sha256='edited'),
                       dict(agent_overrides={'algorithm.entropy_coef': .008}), dict(control_dt=.01)):
            with self.assertRaises(ValueError):
                validate_resume(saved, dict(saved, **change))

    def test_parallelism_uses_margin_and_smallest_near_optimum(self):
        rows = [dict(num_envs=n, status='passed', timing_valid=True, median_fps=fps, min_free_vram_mib=free, min_available_ram_gib=4)
                for n, fps, free in ((128, 1000, 4000), (256, 1900, 2000), (512, 1980, 1500), (1024, 2200, 600))]
        self.assertEqual(choose_parallelism(rows), 256)

    def test_missing_telemetry_never_recommends_larger_batch(self):
        rows = [dict(num_envs=512, status='passed', timing_valid=True, median_fps=9000, min_free_vram_mib=None, min_available_ram_gib=4)]
        self.assertIsNone(choose_parallelism(rows))

    def test_corrupt_clock_row_is_not_recommended_even_with_positive_median(self):
        rows = [dict(num_envs=512, status='passed', timing_valid=False, median_fps=20000,
                     min_free_vram_mib=6000, min_available_ram_gib=10)]
        self.assertIsNone(choose_parallelism(rows))


if __name__ == '__main__':
    unittest.main()

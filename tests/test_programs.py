"""独立程序集成回归：异目录启动、参数校验、轻量导入和实际源码定位。"""
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from workspace.source_map import locate_sources

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = (
    'scripts/reinforcement_learning/rsl_rl/train.py',
    'scripts/reinforcement_learning/rsl_rl/resume.py',
    'scripts/reinforcement_learning/rsl_rl/play.py',
    'scripts/reinforcement_learning/rsl_rl/evaluate.py',
    'scripts/reinforcement_learning/rsl_rl/export.py',
    'scripts/deploy/deploy_sim.py',
    'scripts/tools/import_motions.py',
    'scripts/tools/prepare_motion.py',
    'scripts/tools/show_sources.py',
)


class ProgramTests(unittest.TestCase):
    """通过真实子进程验证用户入口，不创建仿真或优化网络。"""

    def test_all_programs_show_help_from_unrelated_directory(self):
        """模拟从 IDE 或其他目录运行绝对路径，不依赖当前目录碰巧正确。"""
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as unrelated:
            for relative in PROGRAMS:
                with self.subTest(program=relative):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / relative), '--help'],
                        cwd=unrelated, env=environment, capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_invalid_training_count_and_missing_bundle_fail_at_argument_check(self):
        """无效参数在环境初始化前失败，错误信息必须指出用户应修改的参数。"""
        for relative, arguments, expected in (
            (PROGRAMS[0], ['--envs', '0'], 'must be positive'),
            (PROGRAMS[5], [], '--bundle'),
        ):
            with self.subTest(program=relative):
                result = subprocess.run([sys.executable, str(ROOT / relative), *arguments],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_split_modules_and_compatibility_layer_import_without_gpu_frameworks(self):
        """检查导入依赖环和重依赖泄漏，保证帮助与文件工具仍是轻量入口。"""
        code = '''
import sys
from workspace import runtime, training, inference, exporting, deployment, source_map
assert 'torch' not in sys.modules
assert 'warp' not in sys.modules
assert 'mjlab' not in sys.modules
assert callable(runtime.train) and callable(runtime.deploy_sim)
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_installed_pinned_runtime_source_locations_exist(self):
        """在训练解释器中实际检查导航路径，防止文档指向不存在的第三方文件。"""
        try:
            versions = [importlib.metadata.version(p) for p in ('mjlab', 'rsl-rl-lib')]
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('训练依赖未安装；仍可运行其他 CPU 测试')
        if versions != ['1.6.0', '5.4.2']:
            self.skipTest('源码导航针对固定运行时 1.6.0 / 5.4.2')
        for row in locate_sources():
            with self.subTest(role=row['role']):
                self.assertTrue(row['available'], row.get('path'))
                self.assertTrue(row['symbols'])


if __name__ == '__main__':
    unittest.main()

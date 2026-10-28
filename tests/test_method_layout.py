"""目录迁移的关键契约：算法隔离、轻量帮助和原模型代码指纹。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from workspace.common.methods import resolve_model
from workspace.common.profiles import profile_path, method_of

ROOT = Path(__file__).resolve().parents[1]


class MethodLayoutTests(unittest.TestCase):
    def test_latest_is_filtered_by_method_and_supports_old_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models = {}
            for method, parent, spec in (
                ('amp', 'outputs/runs/old_amp', dict(learning_method='amp', task='AMP')),
                ('beyondmimic', 'outputs/beyondmimic/runs/new', dict(task='Mjlab-Tracking-Flat-Unitree-G1')),
                ('velocity', 'outputs/velocity/runs/new', dict(task='Mjlab-Velocity-Flat-Unitree-G1')),
            ):
                run = root/parent
                run.mkdir(parents=True)
                ckpt = run/'model_10.pt'
                ckpt.touch()
                (run/'run_manifest.json').write_text(json.dumps(spec))
                models[method] = ckpt
            for method, path in models.items():
                self.assertEqual(resolve_model(root, 'latest', method), path)
            with self.assertRaisesRegex(ValueError, 'received a beyondmimic'):
                resolve_model(root, str(models['beyondmimic']), 'amp')

    def test_method_help_does_not_import_gpu_frameworks(self):
        for method, commands in (
            ('amp', ['train', 'resume', 'play', 'evaluate', 'export', 'deploy', 'prepare']),
            ('beyondmimic', ['train', 'resume', 'play', 'evaluate', 'export', 'deploy', 'prepare']),
            ('supermimic', ['train', 'resume', 'play', 'evaluate', 'deploy']),
            ('velocity', ['train', 'resume', 'play', 'evaluate', 'export', 'deploy']),
        ):
            for command in commands:
                code = f'''import sys
from workspace.{method}.cli import main
try:
    main(["{command}", "--help"])
except SystemExit as e:
    assert e.code == 0
assert not {{"torch", "warp", "mjlab"}} & sys.modules.keys()
'''
                result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_profiles_are_located_by_method(self):
        for name, method in [('amp_dance_4096', 'amp'), ('tracking_4096', 'beyondmimic'),
                             ('velocity_flat_4096', 'velocity')]:
            path = profile_path(ROOT, name)
            self.assertEqual(path.parent, ROOT/'workspace'/method/'profiles')
            self.assertEqual(method_of(json.loads(path.read_text())), method)

    def test_amp_algorithm_files_still_match_existing_model_fingerprint(self):
        # 原七个学习文件的组合哈希；新增 CLI 不得让现有正式模型失效。
        digest = hashlib.sha256()
        for name in ('__init__.py', 'algorithm.py', 'conversion.py', 'environment.py',
                     'features.py', 'reference.py', 'tasks.py'):
            digest.update(name.encode())
            digest.update((ROOT/'workspace/amp'/name).read_bytes())
        self.assertEqual(digest.hexdigest(), '15d613445ce90a3674e2d46a560d7d04dd0e2eb39be696a121d27197d189cd58')


if __name__ == '__main__':
    unittest.main()

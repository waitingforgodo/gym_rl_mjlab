"""源码替换的隔离契约：不把旧 checkpoint 或换过的动作交给新网络。"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workspace.supermimic import checkpoints as bridge
from workspace.supermimic.configuration import merge_config
from workspace.supermimic.inference import RewardManagerView


class ReproducedTests(unittest.TestCase):
    def test_reward_rate_conversion_uses_control_dt_only_when_enabled(self):
        from types import SimpleNamespace
        manager = SimpleNamespace(_step_reward=3., _scale_by_dt=True, active_terms=['pose'])
        proxy = RewardManagerView(manager, .02)
        self.assertAlmostEqual(proxy.term_rewards, .06)
        self.assertEqual(proxy.active_terms, ['pose'])
        manager._scale_by_dt = False
        self.assertEqual(proxy.term_rewards, 3.)

    def test_latest_never_searches_legacy_run_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder) / 'new'
            old = Path(folder) / 'old'
            old.mkdir()
            (old / 'best.pt').touch()
            with patch.object(bridge, 'RUNS', runs):
                with self.assertRaisesRegex(FileNotFoundError, 'No SuperMimic-Reproduced'):
                    bridge.resolve_checkpoint('latest')
                (runs / 'run').mkdir(parents=True)
                ckpt = runs / 'run/ckpt_latest.pt'
                ckpt.touch()
                self.assertEqual(bridge.resolve_checkpoint('latest'), ckpt)

    def test_unsupported_manifest_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            path.touch()
            (path.parent / 'run_manifest.json').write_text(json.dumps(dict(implementation='unsupported_method')))
            with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint implementation'):
                bridge.resolve_checkpoint(path)

    def test_reference_contract_rejects_reordering_and_modified_data(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            a, b = folder / 'a.npz', folder / 'b.npz'
            a.write_bytes(b'first clip')
            b.write_bytes(b'second clip')
            manifest = dict(data=[dict(sha256=bridge.digest(p)) for p in (a, b)])
            (folder / 'run_manifest.json').write_text(json.dumps(manifest))
            bridge.check_data_contract(folder / 'ckpt_latest.pt', [a, b])
            with self.assertRaisesRegex(ValueError, 'hashes/order'):
                bridge.check_data_contract(folder / 'ckpt_latest.pt', [b, a])
            a.write_bytes(b'changed choreography')
            with self.assertRaisesRegex(ValueError, 'hashes/order'):
                bridge.check_data_contract(folder / 'ckpt_latest.pt', [a, b])

    def test_nested_config_override_rejects_typos(self):
        @dataclass
        class Schedule:
            total_iterations: int = 320000
        @dataclass
        class Config:
            schedule: Schedule = field(default_factory=Schedule)
        cfg = merge_config(Config(), dict(schedule=dict(total_iterations=3000)))
        self.assertEqual(cfg.schedule.total_iterations, 3000)
        with self.assertRaisesRegex(ValueError, 'Unknown Schedule field'):
            merge_config(cfg, dict(schedule=dict(total_iteraions=3000)))


if __name__ == '__main__':
    unittest.main()

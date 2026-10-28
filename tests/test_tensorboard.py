"""Current-run selection must not include archived experiments or stale events."""
import os
from pathlib import Path
import tempfile
import unittest

from workspace.common.tensorboard import current_logdir


class TensorBoardTests(unittest.TestCase):
    def run_folder(self, root, relative, stamp, *, manifest=True):
        run = root / relative
        run.mkdir(parents=True)
        marker = run / ('run_manifest.json' if manifest else 'events.out.tfevents.test')
        marker.write_text('{}')
        os.utime(marker, ns=(stamp, stamp))
        return run

    def test_only_latest_formal_run_excludes_newer_archived_check_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.run_folder(root, 'outputs/supermimic/runs/old', 100)
            current = self.run_folder(root, 'outputs/supermimic/runs/current', 200)
            self.run_folder(root, 'outputs/supermimic/checks/archived', 900)
            self.run_folder(root, 'outputs/benchmarks/runs/test', 1000)
            self.assertEqual(current_logdir(root), current / 'tb')

    def test_new_run_is_selected_before_events_exist_and_old_event_writes_do_not_win(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = self.run_folder(root, 'outputs/amp/runs/old', 100)
            (old / 'events.out.tfevents.old').touch()
            current = self.run_folder(root, 'outputs/instinct/runs/current', 200)
            self.assertEqual(current_logdir(root), current)

    def test_legacy_runs_without_a_manifest_remain_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = self.run_folder(root, 'outputs/runs/current', 100, manifest=False)
            self.assertEqual(current_logdir(root), current)

    def test_empty_workspace_does_not_fall_back_to_showing_all_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                current_logdir(folder)


if __name__ == '__main__':
    unittest.main()

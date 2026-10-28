"""Current-run selection must not include archived experiments or stale events."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workspace.common.tensorboard import current_logdir, existing_logdir, select_endpoint


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

    def test_same_logdir_reuses_existing_service(self):
        with patch('workspace.common.tensorboard.port_available', return_value=False), \
                patch('workspace.common.tensorboard.existing_logdir', return_value='/tmp/current'):
            self.assertEqual(select_endpoint('/tmp/current', 6006), ('reuse', 6006))

    def test_other_service_is_not_stopped_and_next_free_port_is_selected(self):
        with patch('workspace.common.tensorboard.port_available', side_effect=[False, True]), \
                patch('workspace.common.tensorboard.existing_logdir', return_value='/tmp/old'):
            self.assertEqual(select_endpoint('/tmp/current', 6006), ('start', 6007))

    def test_permission_error_is_not_misreported_as_port_conflict(self):
        with patch('workspace.common.tensorboard.port_available', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                select_endpoint('/tmp/current', 6006)

    def test_auto_port_and_port_range(self):
        self.assertEqual(select_endpoint('/tmp/current', 0), ('start', 0))
        with self.assertRaises(ValueError):
            select_endpoint('/tmp/current', 65536)

    def test_changed_server_options_start_a_new_server_instead_of_silently_reusing(self):
        with patch('workspace.common.tensorboard.port_available', side_effect=[False, True]), \
                patch('workspace.common.tensorboard.existing_logdir') as read:
            self.assertEqual(select_endpoint('/tmp/current', 6006, reuse=False), ('start', 6007))
            read.assert_not_called()

    def test_other_services_json_is_not_assumed_to_be_tensorboard_environment(self):
        import io
        with patch('urllib.request.build_opener') as build:
            build.return_value.open.return_value = io.BytesIO(b'[]')
            self.assertIsNone(existing_logdir(6006))


if __name__ == '__main__':
    unittest.main()

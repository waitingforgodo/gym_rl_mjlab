"""Best selection, destructive cleanup boundaries, atomic replacement and resume."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from workspace.supermimic import checkpoints
from workspace.supermimic.retention import BestRetention, PROTOCOL, benchmark_score, evaluate_checkpoint, retain_best

DATA = [dict(path=f'/motions/{name}.npz', sha256=name) for name in ('a', 'b')]


def report(*, failed=(True, True), steps=(20, 20), reward=.4):
    return dict(protocol=PROTOCOL, data=DATA, clips=[
        dict(name=name, frames=101, steps=step, norm_return=reward, failed=fail)
        for name, step, fail in zip(('a', 'b'), steps, failed)])


def trainer(folder):
    tr = SimpleNamespace(log_dir=Path(folder), iteration=100, num_updates=24,
                         is_main=True, device='cpu', cfg=SimpleNamespace(resume=None))
    tr.log_dir.mkdir(parents=True, exist_ok=True)
    def save(name='ckpt_latest.pt'):
        torch.save(dict(iteration=tr.iteration, num_updates=tr.num_updates, config={},
                        wm={'weight': torch.tensor(float(tr.iteration))},
                        policy={'weight': torch.ones(1)}, critic={'weight': torch.ones(1)}), tr.log_dir / name)
    tr.save = save
    return tr


class RetentionTests(unittest.TestCase):
    def test_success_outranks_coverage_and_reward_then_coverage_breaks_failed_ties(self):
        success = benchmark_score(report(failed=(False, True), steps=(100, 1), reward=.01), DATA)
        failed = benchmark_score(report(steps=(99, 99), reward=.99), DATA)
        self.assertGreater(success, failed)
        self.assertGreater(benchmark_score(report(steps=(40, 40), reward=.1), DATA),
                           benchmark_score(report(steps=(20, 20), reward=.9), DATA))

    def test_improvements_replace_best_worse_and_equal_keep_the_previous_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            tr = trainer(folder)
            responses = iter([report(steps=(40, 40)), report(steps=(10, 10), reward=.99),
                              report(steps=(40, 40)), report(failed=(False, True), steps=(100, 10))])
            with retain_best(tr, dict(data=DATA), evaluator=lambda *a: next(responses)):
                tr.save()
                saved = (tr.log_dir/'best.pt').read_bytes()
                for iteration in (200, 300):
                    tr.iteration = iteration
                    tr.save()
                    self.assertEqual((tr.log_dir/'best.pt').read_bytes(), saved)
                tr.iteration = 400
                tr.save()
            best = checkpoints.load_checkpoint(tr.log_dir/'best.pt')
            self.assertEqual(best['iteration'], 400)
            self.assertEqual(best['selection']['score'][0], .5)
            self.assertEqual(list(tr.log_dir.glob('*.pt')), [tr.log_dir/'best.pt'])

    def test_duplicate_saves_evaluate_once_and_cleanup_only_known_local_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            tr = trainer(Path(folder)/'run')
            outside = Path(folder)/'other_run.pt'
            outside.write_bytes(b'preserve')
            (tr.log_dir/'notes.pt').write_bytes(b'preserve')
            (tr.log_dir/'config.json').write_text('{}')
            for name in ('ckpt_latest.pt', 'ckpt_0000050.pt'):
                (tr.log_dir/name).write_bytes(b'intermediate')
            (tr.log_dir/'ckpt_0000051.pt').symlink_to(outside)
            calls = []
            original = tr.save
            def evaluate(*args):
                calls.append(args)
                return report()
            with retain_best(tr, dict(data=DATA), evaluator=evaluate):
                tr.save()
                with self.assertRaises(ValueError):
                    tr.save('../other_run.pt')
                tr.save('ckpt_0000100.pt')
                tr.save()
            self.assertEqual(len(calls), 1)
            self.assertIs(tr.save, original)
            self.assertEqual(outside.read_bytes(), b'preserve')
            self.assertEqual((tr.log_dir/'notes.pt').read_bytes(), b'preserve')
            self.assertTrue((tr.log_dir/'config.json').is_file())
            self.assertFalse(list(tr.log_dir.glob('ckpt_*.pt')))

    def test_failed_evaluation_retains_previous_best_and_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            tr = trainer(folder)
            original = tr.save
            def evaluate(*args):
                if tr.iteration > 100:
                    raise RuntimeError('evaluator failed')
                return report()
            with self.assertRaisesRegex(RuntimeError, 'evaluator failed'):
                with retain_best(tr, dict(data=DATA), evaluator=evaluate):
                    tr.save()
                    previous = (tr.log_dir/'best.pt').read_bytes()
                    tr.iteration = 200
                    tr.save()
            self.assertEqual((tr.log_dir/'best.pt').read_bytes(), previous)
            self.assertTrue((tr.log_dir/'.candidate.pt').is_file())
            self.assertTrue((tr.log_dir/'selection_error.json').is_file())
            self.assertIs(tr.save, original)

    def test_incomplete_mismatched_or_nonfinite_reports_cannot_select_or_prune(self):
        bad_reports = [dict(report(), clips=[]), dict(report(), data=list(reversed(DATA))),
                       report(reward=float('nan')), dict(report(), protocol='different protocol')]
        for bad in bad_reports:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                tr = trainer(folder)
                with self.assertRaises(ValueError):
                    with retain_best(tr, dict(data=DATA), evaluator=lambda *a: bad):
                        tr.save()
                self.assertFalse((tr.log_dir/'best.pt').exists())
                self.assertTrue((tr.log_dir/'.candidate.pt').exists())

    def test_failed_atomic_replace_does_not_damage_best(self):
        with tempfile.TemporaryDirectory() as folder:
            tr = trainer(folder)
            with retain_best(tr, dict(data=DATA), evaluator=lambda *a: report(steps=(tr.iteration//5, 20))):
                tr.save()
                before = (tr.log_dir/'best.pt').read_bytes()
                tr.iteration = 200
                import os
                original_replace = os.replace
                def replace(src, dst):
                    if Path(dst).name == 'best.pt':
                        raise OSError('replace failed')
                    return original_replace(src, dst)
                with patch('workspace.supermimic.retention.os.replace', side_effect=replace):
                    with self.assertRaisesRegex(OSError, 'replace failed'):
                        tr.save()
                self.assertEqual((tr.log_dir/'best.pt').read_bytes(), before)
                self.assertFalse(list(tr.log_dir.glob('.best.*.tmp')))

    def test_checkpoint_carries_its_own_selection_if_json_index_is_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            tr = trainer(folder)
            with retain_best(tr, dict(data=DATA), evaluator=lambda *a: report()):
                tr.save()
            (tr.log_dir/'best.json').unlink()
            manager = BestRetention(tr, dict(data=DATA))
            self.assertEqual(manager.score, (.0, .2, .4))

    def test_resume_inherits_source_best_before_new_candidates_are_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            source = trainer(Path(folder)/'source')
            with retain_best(source, dict(data=DATA), evaluator=lambda *a: report(steps=(80, 80))):
                source.save()
            old = (source.log_dir/'best.pt').read_bytes()
            resumed = trainer(Path(folder)/'resumed')
            resumed.cfg.resume = source.log_dir/'best.pt'
            resumed.iteration = 200
            with retain_best(resumed, dict(data=DATA), evaluator=lambda *a: report()):
                resumed.save()
            self.assertEqual(checkpoints.load_checkpoint(resumed.log_dir/'best.pt')['iteration'], 100)
            self.assertEqual((source.log_dir/'best.pt').read_bytes(), old)

    def test_latest_and_run_directory_prefer_best_over_legacy_latest(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder)
            run = runs/'run'; run.mkdir()
            best = run/'best.pt'; best.touch()
            (run/'ckpt_latest.pt').touch()
            with patch.object(checkpoints, 'RUNS', runs):
                self.assertEqual(checkpoints.resolve_checkpoint('latest'), best)
                self.assertEqual(checkpoints.resolve_checkpoint('best'), best)
                self.assertEqual(checkpoints.resolve_checkpoint(run), best)

    def test_evaluator_uses_an_independent_process_and_the_full_clip_count(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'evaluation.json'
            output.write_text(json.dumps(report()))
            with patch('workspace.supermimic.retention.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                evaluate_checkpoint(Path(folder)/'candidate.pt', output, 'cuda:0', 40)
            command = run.call_args.args[0]
            self.assertEqual(command[1:5], ['-m', 'workspace.cli', 'supermimic', 'evaluate'])
            self.assertEqual(command[command.index('--envs')+1], '40')


if __name__ == '__main__':
    unittest.main()

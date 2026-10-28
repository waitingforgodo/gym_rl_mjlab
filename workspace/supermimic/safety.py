"""Fail on nonfinite training values without changing valid upstream updates."""
from contextlib import contextmanager
from functools import wraps
import math
import os
from pathlib import Path
import tempfile
import time


def require_finite(value, label='state'):
    import torch
    checks = []

    def collect(item, name):
        if torch.is_tensor(item) and (item.is_floating_point() or item.is_complex()):
            checks.append((name, torch.isfinite(item).all()))
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(child, f'{name}.{key}')
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                collect(child, f'{name}[{index}]')
        elif isinstance(item, float) and not math.isfinite(item):
            raise FloatingPointError(f'Nonfinite {name}: {item}')

    collect(value, label)
    # One synchronization per device, then identify the failing tensor only on
    # the error path. Checkpoint tensors may be split across CPU and CUDA.
    devices = {check.device for _, check in checks}
    for device in devices:
        selected = [(name, check) for name, check in checks if check.device == device]
        if not bool(torch.stack([check for _, check in selected]).all()):
            bad = next(name for name, check in selected if not bool(check))
            raise FloatingPointError(f'Nonfinite {bad}; refuse to update, save or load this model')


def trainer_state(trainer):
    return {**{name: getattr(trainer, name).state_dict() for name in ('wm', 'policy', 'critic')},
            'wm_opt': trainer.wm_trainer.opt.state_dict(),
            'policy_opt': trainer.policy_trainer.opt.state_dict(),
            'critic_opt': trainer.policy_trainer.critic_opt.state_dict()}


@contextmanager
def guard_training(trainer):
    """Keep the upstream loop and losses intact; add checks at its boundaries."""
    import torch
    originals = []
    handles = []
    stage = {'name': 'initialization'}
    require_finite(trainer_state(trainer), 'initial training state')

    def replace(obj, name, value):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def checked_update(name, original):
        first = True
        @wraps(original)
        def update(*args, **kwargs):
            nonlocal first
            stage['name'] = name
            started = time.perf_counter()
            if first and trainer.is_main:
                print(f'[INFO] First {name} starting; first call may include graph compilation', flush=True)
            metrics = original(*args, **kwargs)
            require_finite(metrics, name + ' metrics')
            require_finite(trainer_state(trainer), name + ' resulting state')
            if first and trainer.is_main:
                print(f'[INFO] First {name} completed in {time.perf_counter() - started:.1f}s; metrics and state are finite', flush=True)
            first = False
            return metrics
        return update

    def optimizer_guard(name):
        def check(optimizer, args, kwargs):
            gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            require_finite(gradients, name + ' gradients')
        return check

    original_clip = torch.nn.utils.clip_grad_norm_

    @wraps(original_clip)
    def clip(parameters, max_norm, norm_type=2., error_if_nonfinite=False, foreach=None):
        try:
            return original_clip(parameters, max_norm, norm_type,
                                 error_if_nonfinite=True, foreach=foreach)
        except RuntimeError as exc:
            if 'non-finite' not in str(exc):
                raise
            raise FloatingPointError(f'Nonfinite gradient norm during {stage["name"]}; optimizer step was blocked') from exc

    original_act = trainer.act

    @wraps(original_act)
    def act(*args, **kwargs):
        stage['name'] = 'simulation action'
        actions = original_act(*args, **kwargs)
        require_finite(actions, 'simulation actions')
        return actions

    original_save = trainer.save

    @wraps(original_save)
    def save(name='ckpt_latest.pt'):
        if not trainer.is_main:
            return
        stage['name'] = 'checkpoint save'
        if Path(name).name != name:
            raise ValueError('Checkpoint name must be a plain filename')
        require_finite(trainer_state(trainer), 'checkpoint state')
        require_finite({'bin_return': trainer.command.bin_return,
                        'bin_memory': trainer.command.bin_memory}, 'sampler state')
        with tempfile.NamedTemporaryFile(dir=trainer.log_dir, prefix=f'.{name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            original_save(temporary.name)
            os.replace(temporary, Path(trainer.log_dir) / name)
        finally:
            temporary.unlink(missing_ok=True)

    try:
        for name, optimizer in (('world model', trainer.wm_trainer.opt),
                                ('policy', trainer.policy_trainer.opt),
                                ('critic', trainer.policy_trainer.critic_opt)):
            handles.append(optimizer.register_step_pre_hook(optimizer_guard(name)))
        replace(torch.nn.utils, 'clip_grad_norm_', clip)
        replace(trainer.wm_trainer, 'update', checked_update('world model update', trainer.wm_trainer.update))
        replace(trainer.policy_trainer, 'update', checked_update('policy/critic update', trainer.policy_trainer.update))
        replace(trainer, 'act', act)
        replace(trainer, 'save', save)
        yield stage
    finally:
        for handle in handles:
            handle.remove()
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)

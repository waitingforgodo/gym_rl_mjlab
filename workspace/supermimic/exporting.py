"""将 SuperMimic actor 导出为只含推理图的 TorchScript 部署包。"""
from datetime import datetime
import shutil

from .inference import load_checkpoint


def export(root, args):
    import torch
    from workspace.contract import sha256, write_json
    env, adapter, policy, manifest = load_checkpoint(
        root, args.checkpoint, args.device, args.motion)
    destination = root/'outputs'/'supermimic'/'deploy'/(
        f'{args.checkpoint.parent.name}_{datetime.now():%H%M%S_%f}')
    destination.mkdir(parents=True)
    try:
        history, goal = adapter.history, adapter.future_goal()
        traced = torch.jit.trace(policy, (history, goal), check_trace=True)
        traced.save(str(destination/'policy_jit.pt'))
        loaded = torch.jit.load(str(destination/'policy_jit.pt'), map_location=args.device)
        with torch.inference_mode():
            error = float((policy(history, goal)-loaded(history, goal)).abs().max())
        if error > 1e-5:
            raise RuntimeError(f'TorchScript parity failed: {error}')
        shutil.copyfile(manifest['motion_file'], destination/'motion.npz')
        bundle = dict(manifest, checkpoint=str(args.checkpoint),
                      checkpoint_sha256=sha256(args.checkpoint),
                      motion_file='motion.npz', jit_max_abs_error=error,
                      inputs={'observation_history': [1, 10, 93],
                              'future_goal': [1, 10, 71]}, output={'action': [1, 29]},
                      deployment_status='mjlab simulation only')
        write_json(destination/'manifest.json', bundle)
        write_json(destination/'hashes.json', {
            name: sha256(destination/name) for name in ('policy_jit.pt', 'motion.npz', 'manifest.json')})
        print(f'DEPLOY_BUNDLE={destination}', flush=True)
        return destination
    finally:
        env.close()

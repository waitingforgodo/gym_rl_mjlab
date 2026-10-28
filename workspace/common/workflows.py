"""方法接口复用的功能流程，避免为 AMP/BeyondMimic 复制 PPO 实现。"""
from datetime import datetime
from .contract import write_json, validate_resume
from .methods import output_dir


def train(root, args, resume=False):
    from .training import train as implementation
    from .cli import training_lock
    from .telemetry import summarize
    checkpoint = args.checkpoint if resume else None
    if args.dry_run:
        from .configuration import inherited_run_options
        from .runtime import initialize, build_configs, checkpoint_context
        initialize(root)
        saved = None
        profile, motion = args.profile, args.motion
        if checkpoint:
            saved, profile, motion = checkpoint_context(root, checkpoint, profile, motion, False)
        envs, seed = inherited_run_options(args, saved)
        _, _, _, manifest = build_configs(root, profile, envs, motion, seed=seed,
            profile_spec=saved.get('profile_spec') if saved else None,
            profile_sha256=saved.get('profile_sha256') if saved else None)
        if saved:
            validate_resume(saved, manifest)
        return manifest
    with training_lock():
        run = implementation(root, args, checkpoint)
    return summarize(run)


def play(root, args):
    from .inference import load_for_inference, view
    env, _, policy, _ = load_for_inference(root, args, args.checkpoint)
    try:
        view(env, policy, args.viewer, args.port)
    finally:
        env.close()


def evaluate(root, args, method):
    from .inference import load_for_inference, evaluate as implementation
    env, _, policy, manifest = load_for_inference(root, args, args.checkpoint, args.envs)
    try:
        result = implementation(env, policy, args.seconds)
        result.update(checkpoint=str(args.checkpoint), task=manifest['task'],
                      evaluation_conditions=manifest['evaluation_conditions'])
        output = output_dir(root, method, 'evaluations')/f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        write_json(output, result)
        print(f'EVALUATION={output}', flush=True)
        return result
    finally:
        env.close()


def export(root, args):
    from .exporting import export_bundle
    return export_bundle(root, args, args.checkpoint)


def deploy(root, args):
    from .deployment import deploy_sim
    return deploy_sim(root, args)

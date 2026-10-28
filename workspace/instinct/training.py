"""上游 Instinct 环境 + instinct_rl runner 的训练与续训连接层。"""
from datetime import datetime
import json
from .paths import ROOT,OUTPUTS
from .runtime import activate,provenance
from .configuration import profile,build,runner_configuration
from .checkpoints import checkpoint,validate_source
from workspace.common.contract import write_json,sha256


TEACHER_TASKS = {
    'Instinct-Perceptive-Shadowing-G1-v0',
    'Instinct-Perceptive-Shadowing-G1-OneMotion-v0',
}
MOTION_TASKS = {
    'Instinct-Shadowing-WholeBody-Plane-G1-v0',
    'Instinct-BeyondMimic-Plane-G1-v0',
    'Instinct-Parkour-Target-Amp-G1-v0',
}


def configured_profile(name, config):
    """Load a profile and accept only documented env/agent override maps."""
    custom = {}
    if config:
        custom = json.loads(config.read_text())
        if not isinstance(custom, dict):
            raise ValueError('Custom config must be a JSON object')
        unknown = set(custom) - {'profile', 'env_overrides', 'agent_overrides'}
        if unknown:
            raise ValueError(f'Unknown custom config fields: {sorted(unknown)}')
    selected = custom.get('profile', name or 'locomotion')
    if not isinstance(selected, str):
        raise ValueError('Custom config profile must be a string')
    if name and custom.get('profile', name) != name:
        raise ValueError('Command --profile differs from custom config profile')
    spec = profile(selected)
    for key in ('env_overrides', 'agent_overrides'):
        if key in custom:
            if not isinstance(custom[key], dict):
                raise ValueError(f'{key} must be a JSON object')
            values = dict(spec.get(key, {}))
            values.update(custom[key])
            spec[key] = values
    return spec


def validate_data_contract(spec):
    """Reject data switches that the selected upstream task cannot consume."""
    motion = spec.get('motion')
    motion_dir = spec.get('motion_dir')
    scene = spec.get('scene_dataset')
    task = spec['task']
    scene_task = 'Perceptive' in task or 'HOI' in task
    if motion and motion_dir:
        raise ValueError('Choose only one of motion or motion_dir')
    if scene and not scene_task:
        raise ValueError('scene_dataset is only valid for perceptive or HOI tasks')
    if scene and (motion or motion_dir):
        raise ValueError('Scene-backed tasks get motions from scene_dataset; do not also set motion or motion_dir')
    if (motion or motion_dir) and task not in MOTION_TASKS:
        raise ValueError('This task does not accept a generic motion or motion_dir')
    if task in MOTION_TASKS and not (motion or motion_dir):
        raise ValueError('This motion-reference task requires motion or motion_dir')
    if scene_task and not scene:
        raise ValueError('This scene-backed task requires scene_dataset')


def prepare_teacher(spec, source, *, resume=False):
    """Resolve and pin the VAE teacher before any simulation is created."""
    value = spec.get('teacher_checkpoint')
    if 'Vae' not in spec['task']:
        if value:
            raise ValueError('--teacher-checkpoint is only valid for the perceptive VAE task')
        return
    if not value:
        raise ValueError('Perceptive VAE training requires --teacher-checkpoint from a trained perceptive policy')
    path, saved = checkpoint(value)
    validate_source(saved, source)
    teacher_task = saved.get('spec', {}).get('task')
    if teacher_task not in TEACHER_TASKS:
        raise ValueError('VAE teacher must be a perceptive shadowing policy, not another Instinct task')
    models = [candidate for candidate in path.parent.glob('model_*.pt')
              if candidate.stem.split('_')[-1].isdigit()]
    latest = max(models, key=lambda candidate:int(candidate.stem.split('_')[-1]))
    if path != latest:
        raise ValueError(f'VAE teacher must be the latest checkpoint in its run ({latest.name})')
    teacher_params = path.parent/'params/agent.yaml'
    if not teacher_params.is_file():
        raise FileNotFoundError(f'{teacher_params}; incomplete teacher run')
    digest = sha256(path)
    params_digest = sha256(teacher_params)
    if resume and (digest != spec.get('teacher_sha256') or
                   params_digest != spec.get('teacher_params_sha256')):
        raise ValueError('VAE teacher checkpoint differs from the saved run')
    spec['teacher_checkpoint'] = str(path)
    spec['teacher_sha256'] = digest
    spec['teacher_params_sha256'] = params_digest
    spec['teacher_profile'] = saved['profile']


def train(args,resume=False):
    activate()
    source=provenance()
    previous=None
    if resume:
        previous,saved=checkpoint(args.checkpoint,args.profile)
        validate_source(saved,source)
        spec=dict(saved['spec'])
        spec['iterations']=args.iterations or 1000
        if args.config:raise ValueError('Start a new run to change task configuration')
    else:
        spec=configured_profile(args.profile,args.config)
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',spec['profile']):raise ValueError('Invalid profile name')
    for key in ('num_envs','iterations'):
        if type(spec[key]) is not int or spec[key]<1:raise ValueError(f'{key} must be positive')
    for key,value in [('num_envs',args.envs),('iterations',args.iterations),('seed',args.seed)]:
        if value is not None:spec[key]=value
    if args.motion:
        if resume:raise ValueError('Use a new training run to change reference motion')
        spec['motion']=str(args.motion.expanduser().resolve())
        spec.pop('motion_dir',None)
    if args.motion_dir:
        if resume:raise ValueError('Use a new training run to change the motion dataset')
        spec['motion_dir']=str(args.motion_dir.expanduser().resolve())
        spec['motion']=None
    if args.scene_dataset:
        if resume:raise ValueError('Use a new run to change scene data')
        spec['scene_dataset']=str(args.scene_dataset.expanduser().resolve())
        spec['scene_demo']=False
    if args.teacher_checkpoint:
        if resume:raise ValueError('Use a new run to change VAE teacher')
        spec['teacher_checkpoint']=str(args.teacher_checkpoint.expanduser().resolve())
    validate_data_contract(spec)
    prepare_teacher(spec,source,resume=resume)
    if spec.get('motion'):
        path=(ROOT/spec['motion']).resolve()
        if not path.is_file():raise FileNotFoundError(f'{path}; run instinct prepare first')
        if resume and sha256(path)!=spec['motion_sha256']:raise ValueError('Reference motion differs')
        spec['motion']=str(path);spec['motion_sha256']=sha256(path)
    if spec.get('motion_dir'):
        directory=(ROOT/spec['motion_dir']).resolve()
        hashes={p.name:sha256(p) for p in sorted(directory.glob('*retargeted.npz'))}
        if not hashes:raise FileNotFoundError(f'{directory}; run instinct prepare-lafan')
        if resume and hashes!=spec.get('motion_files'):raise ValueError('Motion dataset differs')
        spec['motion_dir']=str(directory);spec['motion_files']=hashes
    if spec.get('scene_dataset'):
        from .scenes import validate_dataset
        report=validate_dataset(ROOT/spec['scene_dataset'],'hoi' if 'HOI' in spec['task'] else 'perceptive')
        if resume and report['files']!=spec.get('scene_files'):raise ValueError('Scene dataset differs')
        spec['scene_dataset']=report['dataset'];spec['scene_files']=report['files']
    cfg,agent=build(spec)
    if args.dry_run:
        return dict(**source,spec=spec,control_dt=cfg.sim.mujoco.timestep*cfg.decimation,simulation_created=False)
    from workspace.common.cli import training_lock
    from instinct_mj.tasks.registry import load_env_cls,load_vecenv_cls,load_runner_cls
    from instinct_rl.runners import OnPolicyRunner
    from instinct_mj.utils.dict import class_to_dict
    from mjlab.utils.os import dump_yaml
    name=args.name or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if name in ('.','..') or '/' in name or '\\' in name:raise ValueError('Use a plain run name')
    run=OUTPUTS/'runs'/f'{name}_{spec["profile"]}'
    with training_lock():
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'run_manifest.json',dict(**source,profile=spec['profile'],spec=spec,source_checkpoint=str(previous) if previous else None))
        from workspace.common.telemetry import ResourceSampler
        sampler=ResourceSampler();sampler.start()
        env=None
        runner=None
        status='failed'
        start_iteration=0
        try:
            env=load_env_cls(spec['task'])(cfg,device=args.device)
            wrapped=load_vecenv_cls(spec['task'])(env,policy_group=agent.policy_observation_group,critic_group=agent.critic_observation_group)
            dump_yaml(run/'params/env.yaml',class_to_dict(cfg));dump_yaml(run/'params/agent.yaml',class_to_dict(agent))
            runner_cls=load_runner_cls(spec['task']) or OnPolicyRunner
            runner=runner_cls(wrapped,runner_configuration(agent),str(run),device=args.device)
            if previous:runner.load(str(previous))
            start_iteration=runner.current_learning_iteration
            print(f'RUN_DIR={run}',flush=True)
            # 与现有 PPO 工具链一致，仅为当前 runner 的计时改用单调时钟。
            # 不修改系统时钟或上游源码，避免 WSL 校时令耗时/吞吐变成负值。
            import instinct_rl.runners.on_policy_runner as runner_module
            from workspace.common.timing import monotonic_runner_clock
            with monotonic_runner_clock(runner_module):
                runner.learn(spec['iterations'],init_at_random_ep_len=False)
            status='passed'
        except KeyboardInterrupt:
            status='interrupted'
            if runner:runner.save(str(run/f'model_{runner.current_learning_iteration}.pt'))
        finally:
            if runner and runner.writer:runner.writer.close()
            if env:env.close()
            resources=sampler.finish()
            resources.update(status=status,num_envs=cfg.scene.num_envs,
                             requested_updates=spec['iterations'],
                             completed_updates=(runner.current_learning_iteration-start_iteration) if runner else 0)
            write_json(run/'resources.json',resources)
    return dict(run=str(run),backend='instinct_mj',status=status,
                requested_updates=spec['iterations'],completed_updates=resources['completed_updates'])

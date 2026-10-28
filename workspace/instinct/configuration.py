"""创建上游任务配置；所有本地覆盖写入 profile 快照，方便续训与部署重建。"""
import json
from pathlib import Path
import re
from .paths import ROOT, HERE
from workspace.common.contract import SDK_JOINTS,sha256
from workspace.common.motion import scalar_float


def apply_overrides(cfg,values):
    """按点号路径修改现有字段；不使用 eval，也不接受未知字段。"""
    for name,value in values.items():
        target=cfg
        parts=name.split('.')
        if any(p.startswith('_') for p in parts):raise ValueError(f'Private config field: {name}')
        try:
            for part in parts[:-1]:
                target=target[part] if isinstance(target,dict) else getattr(target,part)
            last=parts[-1]
            original=target[last] if isinstance(target,dict) else getattr(target,last)
        except (KeyError,AttributeError):raise ValueError(f'Unknown config field: {name}') from None
        if isinstance(original,tuple) and isinstance(value,list):value=tuple(value)
        if isinstance(target,dict):target[last]=value
        else:setattr(target,last,value)


def profile(name):
    if not isinstance(name,str) or not re.fullmatch(r'[a-zA-Z0-9_-]+',name):
        raise ValueError(f'Invalid Instinct profile name: {name}')
    path=HERE/'profiles'/f'{name}.json'
    if not path.is_file():
        raise ValueError(f'Unknown Instinct profile: {name}')
    return json.loads(path.read_text())


def bind_teacher(agent, spec, *, play=False):
    """Bind VAE distillation to an explicit local teacher run.

    The pinned upstream configuration contains a developer-machine path.  It
    must never leak into this workspace.  A teacher is needed while training,
    but not when loading the finished student for playback or export.
    """
    teacher = spec.get('teacher_checkpoint')
    is_vae = 'Vae' in spec['task']
    if not is_vae:
        if teacher:
            raise ValueError('--teacher-checkpoint is only valid for the perceptive VAE task')
        return
    if play:
        agent.algorithm.teacher_logdir = None
        return
    if not teacher:
        raise ValueError('Perceptive VAE training requires --teacher-checkpoint from a trained perceptive policy')
    agent.algorithm.teacher_logdir = str(Path(teacher).parent)


def runner_configuration(agent):
    """Return the native runner dict while preserving TPPO observation shapes.

    The upstream dataclass serializer turns nested teacher-policy tuples into
    lists. TPPO later concatenates those shapes with ``torch.Size`` and thus
    requires tuples.
    """
    result = agent.to_dict()
    teacher = result.get('algorithm', {}).get('teacher_policy', {})
    for group in teacher.get('obs_format', {}).values():
        for name, shape in group.items():
            if isinstance(shape, list):
                group[name] = tuple(shape)
    return result


def build(spec, *, play=False, envs=None):
    """使用同一任务工厂，数据路径通过配置覆盖，不改上游个人目录常量。"""
    from instinct_mj.tasks.registry import load_env_cfg,load_instinct_rl_cfg
    from instinct_mj.motion_reference import MotionReferenceManagerCfg
    from .scenes import factory_data, apply_scene
    task=spec['task']
    with factory_data(spec):
        cfg=load_env_cfg(task,play=play)
    apply_scene(cfg,spec,play=play)
    agent=load_instinct_rl_cfg(task)
    apply_overrides(cfg,spec.get('env_overrides',{}))
    apply_overrides(agent,spec.get('agent_overrides',{}))
    # Enforce the local teacher contract after user overrides so a config file
    # cannot restore the upstream developer-machine path.
    bind_teacher(agent,spec,play=play)
    cfg.scene.num_envs=envs if envs is not None else spec['num_envs']
    cfg.seed=agent.seed=spec['seed']
    motion=spec.get('motion')
    motion_dir=spec.get('motion_dir')
    if motion or motion_dir:
        motion=Path(motion).expanduser().resolve() if motion else None
        directory=(ROOT/motion_dir).resolve() if motion_dir else motion.parent
        files=[motion] if motion else sorted(directory.glob('*retargeted.npz'))
        if not files:raise FileNotFoundError(f'No retargeted motions under {directory}; run instinct prepare-lafan')
        for motion in files:
            validate_motion(motion)
        hashes={p.name:sha256(p) for p in files}
        if spec.get('motion_files') is not None and spec['motion_files']!=hashes:
            raise ValueError('Reference motion dataset differs from saved profile')
        motion=files[0]
        if not motion.is_file():
            raise FileNotFoundError(f'{motion}; run instinct prepare --source motions/processed/amp_dance.npz')
        expected=spec.get('motion_sha256')
        if expected and sha256(motion)!=expected:
            raise ValueError('Reference motion differs from saved profile')
        import hashlib
        dataset_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        selection=ROOT/'.cache/instinct-motion-selections'/f'{dataset_hash}.yaml'
        selection.parent.mkdir(parents=True,exist_ok=True)
        import yaml
        selection.write_text(yaml.safe_dump(dict(selected_files=[p.name for p in files])))
        found=False
        for sensor in cfg.scene.sensors:
            if isinstance(sensor,MotionReferenceManagerCfg):
                found=True
                for buffer in sensor.motion_buffers.values():
                    buffer.path=str(directory)
                    buffer.filtered_motion_selection_filepath=str(selection)
                    if play:buffer.motion_start_from_middle_range=[0.,0.]
        if not found:
            raise ValueError('This task has no motion reference sensor')
    agent.max_iterations=spec['iterations']
    agent.save_interval=spec.get('save_interval',100)
    agent.log_interval=spec.get('log_interval',1)
    return cfg,agent


def validate_motion(motion):
    """本地 torso-root 数据格式检查；把坏数据阻止在 GPU 环境创建之前。"""
    import numpy as np
    if not motion.is_file():raise FileNotFoundError(motion)
    if not motion.name.endswith(('retargetted.npz','retargeted.npz')):
        raise ValueError('Instinct motion filename must end with retargeted.npz')
    with np.load(motion,allow_pickle=False) as data:
        required={'joint_names','joint_pos','base_pos_w','base_quat_w','framerate'}
        if not required<=set(data.files):raise ValueError('Instinct motion format differs; convert with instinct prepare')
        n=len(data['joint_pos'])
        if n<3 or data['joint_pos'].shape!=(n,29) or len(data['joint_names'])!=29:
            raise ValueError('Invalid motion joint dimensions')
        names=data['joint_names'].tolist()
        if len(set(names))!=29:raise ValueError('Duplicate motion joint names')
        if set(names)!=set(SDK_JOINTS):raise ValueError('Motion joint names differ from the G1 contract')
        if scalar_float(data['framerate'],'framerate')<=0:
            raise ValueError('Invalid motion framerate')
        for key,shape in [('joint_pos',(n,29)),('base_pos_w',(n,3)),('base_quat_w',(n,4))]:
            if data[key].shape!=shape or not np.isfinite(data[key]).all():raise ValueError(f'Invalid motion {key}')
        if not np.allclose(np.linalg.norm(data['base_quat_w'],axis=-1),1.,atol=1e-4):
            raise ValueError('Motion quaternions must be normalized wxyz')

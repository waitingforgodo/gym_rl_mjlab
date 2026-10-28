"""导出含归一化器的 actor，并用 JIT/ONNX 进行独立仿真推理。

部署包保存参考动作、关节顺序、PD 和观测规格；不构造 PPO runner。
仍依赖本机固定的仿真后端与源码，不等同于已完成真机部署。
"""
from datetime import datetime
import json
from pathlib import Path
import shutil
from .paths import OUTPUTS
from .runtime import activate,provenance
from .configuration import build
from .checkpoints import validate_source
from workspace.common.contract import sha256,write_json


def copy_dataset(source, destination):
    """Vendor referenced task data while excluding reproducible caches."""
    source=Path(source).expanduser().resolve()
    if not source.is_dir():raise FileNotFoundError(source)
    shutil.copytree(source,destination,
                    ignore=shutil.ignore_patterns('.coacd_cache','__pycache__','*.pyc'))


def export(args):
    from .inference import load
    env,runner,policy,saved,checkpoint=load(args)
    import copy
    import torch
    from torch import nn
    if getattr(runner.alg.actor_critic,'is_recurrent',False):
        env.close()
        raise ValueError('This exporter supports feedforward policies; recurrent policies need explicit hidden-state export')

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.network=copy.deepcopy(runner.alg.actor_critic).cpu().eval()
            normalizer=runner.normalizers.get('policy')
            self.normalizer=copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else nn.Identity()
            self.vae=(hasattr(self.network,'_vae_input_indices') and
                      hasattr(self.network.actor,'encoder') and hasattr(self.network.actor,'decoder'))
        def forward(self,obs):
            obs=self.normalizer(obs)
            if not self.vae:return self.network.act_inference(obs)
            # Upstream VAE act_inference samples z. Export the conventional
            # deterministic posterior-mean policy for reproducible deployment.
            if hasattr(self.network,'encoders'):obs=self.network.encoders(obs)
            input_indices=self.network._vae_input_indices
            aux_indices=self.network._vae_aux_indices
            vae_input=(torch.gather(obs,1,input_indices.unsqueeze(0).expand(obs.shape[0],-1))
                       if input_indices is not None else obs)
            mean,_=self.network.actor.encoder(vae_input).chunk(2,dim=-1)
            if aux_indices is not None:
                aux=torch.gather(obs,1,aux_indices.unsqueeze(0).expand(obs.shape[0],-1))
                mean=torch.cat((mean,aux),dim=-1)
            return self.network.actor.decoder(mean)

    destination=OUTPUTS/'deploy'/f'{checkpoint.parent.name}_{checkpoint.stem}_{datetime.now():%H%M%S_%f}'
    destination.mkdir(parents=True,exist_ok=False)
    try:
        observations,outputs=[],[]
        obs,_=env.get_observations()
        with torch.inference_mode():
            for _ in range(16):
                action=policy(obs)
                observations.append(obs.cpu());outputs.append(action.cpu())
                obs,_,_,_=env.step(action)
            x,y=torch.cat(observations),torch.cat(outputs)
            module=Actor().eval()
            expected=module(x)
            if not module.vae and not torch.allclose(expected,y,atol=1e-4,rtol=1e-4):
                raise ValueError('Export actor differs from loaded inference policy')
            jit=torch.jit.freeze(torch.jit.trace(module,x[:1]))
            predicted=jit(x)
            if not torch.allclose(predicted,expected,atol=1e-4,rtol=1e-4):
                raise ValueError('JIT actor parity failed')
            jit.save(str(destination/'policy_jit.pt'))
            torch.onnx.export(module,x[:1],str(destination/'policy.onnx'),dynamo=False,opset_version=17,
                              input_names=['obs'],output_names=['actions'],
                              dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}})
        import onnx
        import onnxruntime as ort
        onnx.checker.check_model(onnx.load(str(destination/'policy.onnx')))
        session=ort.InferenceSession(str(destination/'policy.onnx'),providers=['CPUExecutionProvider'])
        import numpy as np
        onnx_actions=session.run(None,{'obs':x.numpy()})[0]
        if not np.allclose(onnx_actions,expected.numpy(),atol=1e-4,rtol=1e-4):raise ValueError('ONNX actor parity failed')
        manifest=dict(saved)
        manifest['spec']=dict(saved['spec'])
        if manifest['spec'].get('motion'):
            shutil.copyfile(manifest['spec']['motion'],destination/'motion_retargeted.npz')
            manifest['spec']['motion']='motion_retargeted.npz'
        if manifest['spec'].get('motion_dir'):
            copy_dataset(manifest['spec']['motion_dir'],destination/'motion_dataset')
            manifest['spec']['motion_dir']='motion_dataset'
        if manifest['spec'].get('scene_dataset'):
            copy_dataset(manifest['spec']['scene_dataset'],destination/'scene_dataset')
            manifest['spec']['scene_dataset']='scene_dataset'
        if manifest['spec'].get('teacher_checkpoint'):
            manifest['teacher_provenance']={
                'profile':manifest['spec'].get('teacher_profile'),
                'checkpoint_sha256':manifest['spec'].get('teacher_sha256'),
                'params_sha256':manifest['spec'].get('teacher_params_sha256'),
            }
            for key in ('teacher_checkpoint','teacher_profile','teacher_sha256','teacher_params_sha256'):
                manifest['spec'].pop(key,None)
        robot=env.unwrapped.scene['robot']
        action=env.unwrapped.action_manager.get_term('joint_pos')
        def first(value):
            if torch.is_tensor(value):
                return (value[0] if value.ndim>1 else value).detach().cpu().tolist()
            return value
        manifest.update(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                        actor_observations=env.get_obs_format()['policy'],
                        joint_names=list(robot.joint_names),action_dim=env.num_actions,
                        robot_root_body='torso_link',action_joint_names=list(action.target_names),
                        action_scale=first(action.scale),action_offset=first(action.offset),
                        default_joint_pos=first(robot.data.default_joint_pos),
                        control_dt=env.unwrapped.step_dt,
                        jit_max_abs_error=float((predicted-expected).abs().max()),
                        onnx_max_abs_error=float(np.max(np.abs(onnx_actions-expected.numpy()))),
                        normalization='included in policy_jit.pt and policy.onnx',deployment_scope='same-backend simulation',
                        hardware_enabled=False,hardware_joint_sign_mapping='not validated against physical SDK')
        if module.vae:manifest['vae_latent_policy']='posterior mean (deterministic deployment)'
        write_json(destination/'manifest.json',manifest)
        write_json(destination/'hashes.json',{
            str(p.relative_to(destination)):sha256(p)
            for p in destination.rglob('*') if p.is_file()
        })
        print(f'DEPLOY_BUNDLE={destination}',flush=True)
        return dict(bundle=str(destination),jit_max_abs_error=manifest['jit_max_abs_error'],onnx_max_abs_error=manifest['onnx_max_abs_error'])
    finally:env.close()


def deploy(args):
    bundle=args.bundle.expanduser().resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest.get('backend')!='instinct_mj':raise ValueError('Bundle is not from InstinctMJ')
    hashes=json.loads((bundle/'hashes.json').read_text())
    if not isinstance(hashes,dict):raise ValueError('Bundle integrity differs: hashes.json must be an object')
    for name,expected in hashes.items():
        relative=Path(name)
        path=(bundle/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(bundle) or not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'Bundle integrity differs: {name}')
    required={'manifest.json','policy_jit.pt','policy.onnx'}
    if not required<=set(hashes):
        raise ValueError(f'Bundle integrity differs: missing required hashes {sorted(required-set(hashes))}')
    actual={str(path.relative_to(bundle)) for path in bundle.rglob('*') if path.is_file()
            and path.relative_to(bundle).as_posix() not in {'hashes.json','simulation_result.json'}}
    if set(hashes)!=actual:
        raise ValueError('Bundle integrity differs: payload file set does not match hashes.json')
    spec=dict(manifest['spec'])
    for key in ('motion','motion_dir','scene_dataset'):
        if spec.get(key):
            relative=Path(spec[key])
            path=(bundle/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(bundle):
                raise ValueError(f'Bundle {key} path must stay inside the bundle: {relative}')
            spec[key]=str(path)
    activate();validate_source(manifest,provenance())
    cfg,agent=build(spec,play=True,envs=args.envs)
    from instinct_mj.tasks.registry import load_env_cls,load_vecenv_cls
    from .inference import rollout,show
    import torch
    env=load_env_cls(spec['task'])(cfg,device=args.device)
    try:
        wrapped=load_vecenv_cls(spec['task'])(env,policy_group=agent.policy_observation_group,critic_group=agent.critic_observation_group)
        if list(env.scene['robot'].joint_names)!=manifest['joint_names']:raise ValueError('Bundle joint order differs')
        if args.engine=='jit':
            policy=torch.jit.load(str(bundle/'policy_jit.pt'),map_location=args.device).eval()
        else:
            import onnxruntime as ort
            session=ort.InferenceSession(str(bundle/'policy.onnx'),providers=['CPUExecutionProvider'])
            def policy(obs):
                return torch.as_tensor(session.run(None,{'obs':obs.detach().cpu().numpy()})[0],device=obs.device)
        torch.manual_seed(spec['seed'])
        if args.headless:
            report=rollout(wrapped,policy,args.steps)
            write_json(bundle/'simulation_result.json',report)
            return report
        show(wrapped,policy,args.viewer,args.port)
    finally:env.close()

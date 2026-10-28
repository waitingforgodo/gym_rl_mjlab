"""Instinct checkpoint 回放/评测；与原工作空间模型格式完全分开。"""
from datetime import datetime
import json
from pathlib import Path
from .runtime import activate,provenance
from .paths import OUTPUTS
from .configuration import build,runner_configuration
from .checkpoints import checkpoint,validate_source
from workspace.common.contract import write_json


def load(args):
    path,saved=checkpoint(args.checkpoint,args.profile)
    activate()
    validate_source(saved,provenance())
    cfg,agent=build(saved['spec'],play=True,envs=args.envs)
    from instinct_mj.tasks.registry import load_env_cls,load_vecenv_cls,load_runner_cls
    from instinct_rl.runners import OnPolicyRunner
    env=load_env_cls(saved['spec']['task'])(cfg,device=args.device)
    try:
        wrapped=load_vecenv_cls(saved['spec']['task'])(env,policy_group=agent.policy_observation_group,critic_group=agent.critic_observation_group)
        runner=(load_runner_cls(saved['spec']['task']) or OnPolicyRunner)(
            wrapped,runner_configuration(agent),log_dir=None,device=args.device)
        runner.load(str(path))
        policy=runner.get_inference_policy(device=args.device)
        print(f'CHECKPOINT={path}',flush=True)
        return wrapped,runner,policy,saved,path
    except BaseException:
        env.close();raise


def rollout(env,policy,steps):
    """报告首回合存活，并另列失败后 reset 的次数，避免混算成功。"""
    import torch
    obs,_=env.get_observations()
    active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    alive=torch.zeros(env.num_envs,device=env.device)
    failures=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
    total=0.
    with torch.inference_mode():
        for _ in range(steps):
            actions=policy(obs)
            if not torch.isfinite(actions).all():raise FloatingPointError('Nonfinite Instinct action')
            obs,rewards,dones,info=env.step(actions)
            if not torch.isfinite(obs).all() or not torch.isfinite(rewards).all():
                raise FloatingPointError('Nonfinite Instinct rollout')
            alive+=active
            # timeout / 动作结束不等同跌倒；首回合结束统一停止存活计时。
            terminated=env.unwrapped.termination_manager.terminated
            failures+=terminated.long()
            active&=~dones.bool()
            total+=float(rewards.mean())
    return dict(steps=steps,num_envs=env.num_envs,control_dt=env.unwrapped.step_dt,
                mean_step_reward=total/steps,failure_events=int(failures.sum()),
                first_episode_seconds=(alive*env.unwrapped.step_dt).cpu().tolist(),
                episodes_not_ended=int(active.sum()),
                protocol='fixed-duration rollout with first-episode lengths; auto resets continue')


def show(env,policy,viewer='viser',port=8080):
    from instinct_mj.scripts.instinct_rl.play import _ViewerEnvAdapter
    from mjlab.viewer import NativeMujocoViewer,ViserPlayViewer
    import torch
    adapter=_ViewerEnvAdapter(env)
    server=None
    try:
        with torch.inference_mode():
            if viewer=='native':NativeMujocoViewer(adapter,policy).run()
            else:
                import viser
                server=viser.ViserServer(host='0.0.0.0',port=port)
                print(f'网页回放：http://localhost:{server.get_port()}',flush=True)
                ViserPlayViewer(adapter,policy,viser_server=server).run()
    finally:
        if server:server.stop()


def run(args,evaluate=False):
    env,runner,policy,saved,path=load(args)
    try:
        # 构造训练 runner 会消耗随机数，而独立 JIT 部署不构造 runner。
        # 统一推理开始时的 RNG 起点，避免由加载方式改变后续观测噪声。
        import torch
        torch.manual_seed(saved['spec']['seed'])
        if evaluate or args.headless:
            report=rollout(env,policy,args.steps)
            report.update(checkpoint=str(path),profile=saved['profile'],backend='instinct_mj')
            output=OUTPUTS/'evaluations'/f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
            write_json(output,report)
            print(f'EVALUATION={output}',flush=True)
            return report
        show(env,policy,args.viewer,args.port)
    finally:env.close()

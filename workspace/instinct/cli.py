"""统一 InstinctMJ 工具链：检查、数据、训练、续训、评测、导出、仿真。"""
import argparse
import json
from pathlib import Path
from .paths import HERE,ROOT
from workspace.common.cli import emit,positive


def parser():
    p=argparse.ArgumentParser(prog='g1.sh instinct',description='InstinctMJ + pinned instinct_rl; existing runtime, separate checkpoints')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('tasks',help='List upstream tasks and local ready-to-use profiles')
    sub.add_parser('doctor',help='Report sources, runtime, local dependencies and assets')
    sub.add_parser('setup',help='Install only project-local supplementary dependencies and retrieve pinned meshes')
    summary=sub.add_parser('summary',help='Read latest native run curves/resources, without simulation')
    summary.add_argument('--checkpoint',default='latest')
    summary.add_argument('--profile')
    data=sub.add_parser('prepare',help='Convert existing named NPZ motion into Instinct torso-root format')
    data.add_argument('--source',type=Path,default=ROOT/'motions/processed/amp_dance.npz')
    data.add_argument('--output',type=Path)
    sub.add_parser('prepare-lafan',help='Download/convert upstream LAFAN1 G1 40 clips into both backend formats')
    scene=sub.add_parser('prepare-scene',help='Prepare flat perceptive demo or validate a real terrain/HOI dataset')
    scene_mode=scene.add_mutually_exclusive_group(required=True)
    scene_mode.add_argument('--demo',action='store_true',help='Generate a clearly labelled flat-ground perceptive demo')
    scene_mode.add_argument('--dataset',type=Path)
    scene.add_argument('--source',type=Path)
    scene.add_argument('--kind',choices=('perceptive','hoi'),default='perceptive')
    profiles=[p.stem for p in sorted((HERE/'profiles').glob('*.json'))]
    for command in ('train','resume','play','evaluate','export'):
        child=sub.add_parser(command)
        child.add_argument('--profile',choices=profiles)
        child.add_argument('--device',default='cuda:0')
        if command!='train':child.add_argument('--checkpoint',default='latest')
        if command in ('train','resume'):
            child.add_argument('--envs',type=positive)
            child.add_argument('--iterations',type=positive,help='ADDITIONAL PPO updates; resume default 1000')
            child.add_argument('--seed',type=int)
            motion=child.add_mutually_exclusive_group()
            motion.add_argument('--motion',type=Path)
            motion.add_argument('--motion-dir',type=Path,help='Dataset of native torso-root motions')
            child.add_argument('--scene-dataset',type=Path,help='Matched terrain metadata / HOI objects.json dataset')
            child.add_argument('--teacher-checkpoint',type=Path,help='Trained perceptive PPO teacher for VAE distillation')
            child.add_argument('--name')
            child.add_argument('--dry-run',action='store_true')
            child.add_argument('--config',type=Path,help='JSON profile snapshot plus env_overrides/agent_overrides')
        else:child.add_argument('--envs',type=positive,default=1)
        if command in ('play','evaluate'):
            child.add_argument('--steps',type=positive,default=500)
            child.add_argument('--headless',action='store_true')
            child.add_argument('--viewer',choices=('viser','native'),default='viser')
            child.add_argument('--port',type=positive,default=8080)
    child=sub.add_parser('deploy',help='Run an exported Instinct JIT/ONNX bundle; no PPO runner/checkpoint')
    child.add_argument('--bundle',required=True,type=Path)
    child.add_argument('--engine',choices=('jit','onnx'),default='jit')
    child.add_argument('--envs',type=positive,default=1)
    child.add_argument('--device',default='cuda:0')
    child.add_argument('--steps',type=positive,default=500)
    child.add_argument('--headless',action='store_true')
    child.add_argument('--viewer',choices=('viser','native'),default='viser')
    child.add_argument('--port',type=positive,default=8080)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        if args.command=='tasks':
            result=dict(profiles={path.stem:json.loads(path.read_text()) for path in (HERE/'profiles').glob('*.json')},
                        upstream_tasks=json.loads((HERE/'tasks.json').read_text()),
                        note=('Perceptive profiles include a flat integration demo; real benchmarks need matched terrain data. '
                              'Parkour defaults to an 18-clip LAFAN locomotion engineering baseline, not paper data. '
                              'HOI requires user-supplied object trajectories, meshes and objects.json.'))
        elif args.command in ('doctor','setup'):
            from .bootstrap import doctor,setup
            result=(doctor if args.command=='doctor' else setup)()
        elif args.command=='summary':
            from .checkpoints import checkpoint
            path,manifest=checkpoint(args.checkpoint,args.profile)
            from workspace.common.telemetry import summarize
            result=dict(checkpoint=str(path),manifest=manifest,summary=summarize(path.parent))
        elif args.command=='prepare':
            # 转换只需 CPU MuJoCo FK 和文件校验，不加载 GPU 或学习框架。
            from .data import prepare
            result=prepare(args.source,args.output)
        elif args.command=='prepare-lafan':
            from workspace.common.datasets import prepare_lafan1,STANDARD
            result=prepare_lafan1()
            source_names=sorted(result['outputs'])
            if len(source_names)!=40:
                raise ValueError(f'Pinned LAFAN conversion must contain exactly 40 clips, got {len(source_names)}')
            from .data import prepare
            directory=ROOT/'motions/processed/instinct/lafan1'
            locomotion_directory=ROOT/'motions/processed/instinct/lafan1_locomotion'
            converted_names=[];locomotion_names=[]
            for name in source_names:
                path=STANDARD/name
                converted=directory/f'{path.stem}_retargeted.npz'
                prepare(path,converted)
                converted_names.append(converted.name)
                if path.stem.startswith(('walk','run','sprint')):
                    import shutil
                    locomotion=locomotion_directory/converted.name
                    locomotion.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(converted,locomotion)
                    locomotion_names.append(locomotion.name)
                print(f'INSTINCT_MOTION={path.stem}',flush=True)
            if len(locomotion_names)!=18:
                raise ValueError(f'LAFAN locomotion subset must contain exactly 18 clips, got {len(locomotion_names)}')
            actual={path.name for path in directory.glob('*retargeted.npz')}
            if actual!=set(converted_names):
                raise ValueError('Instinct LAFAN directory contains unexpected clips; move them aside before training')
            actual_locomotion={path.name for path in locomotion_directory.glob('*retargeted.npz')}
            if actual_locomotion!=set(locomotion_names):
                raise ValueError('Instinct LAFAN locomotion directory contains unexpected clips; move them aside before training')
            result.update(instinct_motion_dir=str(directory),instinct_clips=len(converted_names),
                          instinct_locomotion_dir=str(locomotion_directory),
                          instinct_locomotion_clips=len(locomotion_names))
        elif args.command=='prepare-scene':
            from .scenes import prepare_demo,validate_dataset
            if args.demo:result=prepare_demo(args.source)
            elif args.dataset:result=validate_dataset(args.dataset,args.kind)
            else:raise ValueError('Choose --demo or --dataset DIRECTORY')
        elif args.command in ('train','resume'):
            from .training import train
            result=train(args,resume=args.command=='resume')
        elif args.command in ('play','evaluate'):
            from .inference import run
            result=run(args,evaluate=args.command=='evaluate')
        else:
            from .deployment import export,deploy
            result=(export if args.command=='export' else deploy)(args)
        if result is not None:emit(result)
    except (ValueError,FileNotFoundError,ImportError) as exc:
        p.error(f'{exc}\nUse "g1.sh instinct doctor" or "g1.sh instinct setup" to inspect local prerequisites.')
    except FileExistsError as exc:
        p.error(f'Run already exists: {exc.filename}; choose another --name (existing runs are preserved)')

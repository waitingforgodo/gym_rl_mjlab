"""Instinct 模型选择；严格拒绝当前 RSL-RL、AMP、SuperMimic 权重。"""
import json
import re
from pathlib import Path
from .paths import OUTPUTS


def checkpoint(value='latest', profile=None):
    if str(value)=='latest':
        runs=list((OUTPUTS/'runs').glob('*/run_manifest.json'))
        candidates=[]
        for manifest in runs:
            data=json.loads(manifest.read_text())
            if data.get('backend')!='instinct_mj':continue
            if profile and data.get('profile')!=profile:continue
            # 每个运行按更新编号选模型，跨运行再按保存时间选择。
            # 避免复制/触碰旧模型后，latest 倒退到更早的更新。
            files=[p for p in manifest.parent.glob('model_*.pt')
                   if re.fullmatch(r'model_\d+\.pt',p.name)]
            if files:candidates.append(max(files,key=lambda p:int(p.stem.split('_')[-1])))
        if not candidates:
            raise FileNotFoundError('No Instinct checkpoint; run instinct train first')
        path=max(candidates,key=lambda p:p.stat().st_mtime_ns)
    else:
        path=Path(value).expanduser().resolve()
        if path.is_dir():
            files=[p for p in path.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt',p.name)]
            if not files:raise FileNotFoundError(f'No model_N.pt in {path}')
            path=max(files,key=lambda p:int(p.stem.split('_')[-1]))
    if not path.is_file():raise FileNotFoundError(path)
    manifest=path.parent/'run_manifest.json'
    if not manifest.is_file() or json.loads(manifest.read_text()).get('backend')!='instinct_mj':
        raise ValueError('Checkpoint is not from this InstinctMJ backend; existing AMP/PPO/SuperMimic weights are incompatible')
    saved=json.loads(manifest.read_text())
    if profile and saved['profile']!=profile:
        raise ValueError('Explicit checkpoint profile differs from --profile')
    return path,saved


def validate_source(saved,current):
    for key in ('backend','sources','runtime_versions'):
        if saved[key]!=current[key]:raise ValueError(f'Instinct model contract differs: {key}')

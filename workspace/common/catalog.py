"""可用方法与后端的轻量目录，不导入仿真或学习框架。"""
from pathlib import Path


def methods(root):
    result={}
    for name in ('amp','beyondmimic','supermimic','velocity','instinct'):
        package=Path(root)/'workspace'/name
        result[name]=dict(path=str(package),help=f'bash g1.sh {name} --help',
                          profiles=[p.stem for p in sorted((package/'profiles').glob('*.json'))])
    result['amp']['backend']='mjlab + RSL-RL + local AMP'
    result['beyondmimic']['backend']='mjlab + RSL-RL tracking'
    result['velocity']['backend']='mjlab + RSL-RL velocity'
    result['supermimic']['backend']='SuperMimic-Reproduced world model/BPTT'
    result['instinct']['backend']='InstinctMJ + pinned instinct_rl; separate robot/checkpoint contract'
    return result

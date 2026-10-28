"""安装项目内附加包并恢复被 Git 忽略的上游网格；原虚拟环境保持原样。"""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from .paths import HERE, ROOT, MJ_SOURCE, RL_SOURCE, ADDONS
from workspace.common.contract import sha256


def setup():
    """仅向 --target 目录安装锁定的辅助包；明确禁止下载核心运行栈。"""
    requirements = HERE/'requirements-addons.txt'
    blocked = {'torch','torchvision','numpy','scipy','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib'}
    for line in requirements.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            if line.split('==')[0].lower() in blocked or '==' not in line:
                raise ValueError(f'Only pinned supplementary dependencies allowed: {line}')
    python = shutil.which('python3')
    if python is None:
        raise RuntimeError('python3 with pip is required to install project-local supplementary dependencies')
    # g1.sh 禁止训练加载用户 site-packages；安装器本身可能恰好位于系统
    # Python 的用户目录，仅在这个子进程恢复可见性，目标仍强制为项目缓存。
    installer_env=dict(os.environ)
    installer_env.pop('PYTHONNOUSERSITE',None)
    subprocess.run([python,'-m','pip','--python',sys.executable,'install','--target',str(ADDONS),
                    '--upgrade','--no-deps','-r',str(requirements)],check=True,env=installer_env)
    for base in (MJ_SOURCE, RL_SOURCE):
        info = json.loads((base/'SOURCE_INFO.json').read_text())
        for relative, expected in info['assets'].items():
            destination = base/relative
            if destination.is_file() and sha256(destination) == expected:
                continue
            url = info['repository'].replace('https://github.com/','https://raw.githubusercontent.com/')
            url += '/'+info['commit']+'/'+relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary = destination.with_suffix(destination.suffix+'.download')
            with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response,out)
            if sha256(temporary) != expected:
                temporary.unlink()
                raise ValueError(f'Download hash differs: {relative}')
            temporary.replace(destination)
    return dict(addons=str(ADDONS), source_assets='verified', shared_environment_modified=False)


def doctor():
    """只读报告来源、辅助包、资产与原解释器，不启动仿真。"""
    from .runtime import provenance
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(ADDONS)])}
    missing=[]
    for base in (MJ_SOURCE, RL_SOURCE):
        info=json.loads((base/'SOURCE_INFO.json').read_text())
        missing.extend(str(base/name) for name in info['assets'] if not (base/name).is_file())
    numpy_version=importlib.metadata.version('numpy')
    notes=[]
    if tuple(int(p) for p in numpy_version.split('.')[:2])>=(2,5):
        notes.append(f'Existing NumPy {numpy_version} exceeds upstream <2.5 declaration; only the documented local tasks are smoke-tested.')
    installed={name.lower().replace('_','-') for name in packages}
    required=[line.split('==')[0] for line in (HERE/'requirements-addons.txt').read_text().splitlines()
              if line and not line.startswith('#')]
    missing_packages=[name for name in required if name.lower().replace('_','-') not in installed]
    try:importlib.metadata.version('pyvista');optional=[]
    except importlib.metadata.PackageNotFoundError:optional=['pyvista (advanced visualization/terrain utilities)']
    from workspace.common.telemetry import gpu_free_mib
    return dict(python=sys.executable, **provenance(verify=False), addons=packages,
                compatibility_notes=notes,
                missing_supplementary_packages=missing_packages,optional_upstream_packages_not_installed=optional,
                gpu_free_mib=gpu_free_mib(),missing_assets=missing, setup_command='bash g1.sh instinct setup')

"""定位实际安装版本的源码，回答“环境/奖励/PPO 的完整代码在哪里”。

按 distribution 元数据找到包文件，避免 import mjlab.tasks 时触发重依赖。
返回路径指向运行时本身；参考资料中的 RSL-RL 不一定是训练使用的版本。
"""
import ast
from importlib.metadata import PackageNotFoundError, distribution


def locate_sources(kind='all'):
    """返回公共源码和指定任务源码的路径、版本、顶层函数/类行号。"""
    if kind not in ('all', 'velocity', 'tracking'):
        raise ValueError(f'Unknown source kind: {kind}')
    files = [
        ('环境 step/reset', 'mjlab', 'mjlab/envs/manager_based_rl_env.py'),
        ('任务注册/加载', 'mjlab', 'mjlab/tasks/registry.py'),
        ('G1 模型与 PD/动作尺度', 'mjlab', 'mjlab/asset_zoo/robots/unitree_g1/g1_constants.py'),
        ('通用观测', 'mjlab', 'mjlab/envs/mdp/observations.py'),
        ('通用奖励', 'mjlab', 'mjlab/envs/mdp/rewards.py'),
        ('动作处理', 'mjlab', 'mjlab/envs/mdp/actions/actions.py'),
        ('RSL-RL 环境包装器', 'mjlab', 'mjlab/rl/vecenv_wrapper.py'),
        ('mjlab runner/导出', 'mjlab', 'mjlab/rl/runner.py'),
        ('PPO 采样/更新循环', 'rsl-rl-lib', 'rsl_rl/runners/on_policy_runner.py'),
        ('PPO 算法', 'rsl-rl-lib', 'rsl_rl/algorithms/ppo.py'),
        ('Actor/Critic 网络', 'rsl-rl-lib', 'rsl_rl/models/mlp_model.py'),
        ('采样缓存与 GAE', 'rsl-rl-lib', 'rsl_rl/storage/rollout_storage.py'),
        ('观测归一化', 'rsl-rl-lib', 'rsl_rl/modules/normalization.py'),
    ]
    for task in ('velocity', 'tracking'):
        if kind not in ('all', task):
            continue
        prefix = f'mjlab/tasks/{task}'
        files.extend([
            (f'{task} G1 配置', 'mjlab', f'{prefix}/config/g1/env_cfgs.py'),
            (f'{task} PPO 超参数', 'mjlab', f'{prefix}/config/g1/rl_cfg.py'),
            (f'{task} 基础环境/奖励权重', 'mjlab', f'{prefix}/{task}_env_cfg.py'),
            (f'{task} 奖励函数', 'mjlab', f'{prefix}/mdp/rewards.py'),
            (f'{task} 观测函数', 'mjlab', f'{prefix}/mdp/observations.py'),
            (f'{task} 命令', 'mjlab', f'{prefix}/mdp/{"velocity_command" if task == "velocity" else "commands"}.py'),
            (f'{task} 终止条件', 'mjlab', f'{prefix}/mdp/terminations.py'),
        ])
    rows = []
    for role, package, relative in files:
        row = dict(role=role, package=package, relative_path=relative)
        try:
            installed = distribution(package)
        except PackageNotFoundError:
            # 工具 Python 未安装训练依赖时仍可打印预期相对路径帮助查找。
            rows.append(dict(row, available=False, reason='package not installed'))
            continue
        path = installed.locate_file(relative).resolve()
        row.update(version=installed.version, path=str(path), available=path.is_file())
        if path.is_file():
            tree = ast.parse(path.read_text(encoding='utf-8'))
            row['symbols'] = {node.name: node.lineno for node in tree.body
                              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        rows.append(row)
    return rows

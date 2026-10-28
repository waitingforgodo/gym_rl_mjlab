"""按算法定位配置；原 JSON 字节保留，已有模型的配置哈希不变。"""
import json
from pathlib import Path

METHODS = ('amp', 'beyondmimic', 'velocity')


def profile_paths(root, method=None):
    methods = (method,) if method else METHODS
    return sorted(p for name in methods for p in (Path(root)/'workspace'/name/'profiles').glob('*.json'))


def profile_path(root, name):
    found = [p for p in profile_paths(root) if p.stem == name]
    if len(found) != 1:
        raise ValueError(f'Unknown or ambiguous profile: {name}')
    return found[0]


def method_of(spec):
    """以清单内容辨别算法，不能仅凭模型目录名猜测。"""
    if spec.get('learning_method') == 'amp':
        return 'amp'
    if 'Tracking' in spec.get('task', ''):
        return 'beyondmimic'
    return 'velocity'


def read_profile(root, name):
    return json.loads(profile_path(root, name).read_text(encoding='utf-8'))

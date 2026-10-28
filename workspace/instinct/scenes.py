"""高级场景的数据和配置连接，不修改固定的 InstinctMJ 源文件。

感知任务需要 motion↔terrain metadata；HOI 还需要真实物体轨迹和网格。
提供平地感知示例以便验证深度相机链路，同时支持用户导入实际采集场景。
示例明确标为 demo，不把普通 LAFAN 动作冒充 OMOMO/跑酷专用数据。
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

from .paths import ROOT
from workspace.common.contract import sha256, write_json

PERCEPTIVE = 'instinct_mj.tasks.shadowing.perceptive.config.g1.'


def prepare_demo(source=None):
    """用现有 G1 动作与平地生成匹配数据，实际训练仍走上游深度相机环境。"""
    import numpy as np
    import trimesh
    import yaml
    from .data import prepare
    base = ROOT / 'motions/processed/instinct/scenes/perceptive_demo'
    base.mkdir(parents=True, exist_ok=True)
    motion = base / 'demo_retargeted.npz'
    prepare(Path(source or ROOT/'motions/processed/amp_dance.npz'), motion)
    floor = base / 'floor.stl'
    if not floor.exists():
        mesh = trimesh.creation.box(extents=(30., 16., .2))
        mesh.apply_translation((0., 0., -.1))
        mesh.export(floor)
    with np.load(motion, allow_pickle=False) as data:
        positions = data['base_pos_w']
        if (np.abs(positions[:, 0]) > 14.).any() or (np.abs(positions[:, 1]) > 7.).any():
            raise ValueError('Demo motion leaves flat terrain; supply matched real terrain data instead')
    # Upstream TerrainMotion maps terrain difficulty to a zero-based integer
    # index and parses every metadata ID with int().  Keep the demo on that
    # native contract instead of using a descriptive string identifier.
    (base/'metadata.yaml').write_text(yaml.safe_dump(dict(
        terrains=[dict(terrain_id=0, terrain_file=floor.name)],
        motion_files=[dict(terrain_id=0, motion_file=motion.name)]), sort_keys=False))
    write_json(base/'scene_info.json', dict(kind='perceptive', purpose='flat-ground integration demo',
               real_scanned_terrain=False, source=str(source or 'motions/processed/amp_dance.npz')))
    return validate_dataset(base, 'perceptive')


def checked_file(root, relative):
    """数据包内部引用保持相对路径，移动目录后仍然可用。"""
    relative = Path(relative)
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Dataset reference must stay inside its directory: {relative}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_dataset(path, kind):
    """检查地形配对/物体引用并生成完整数据哈希，缺数据在创建仿真前报错。"""
    import numpy as np
    import yaml
    from .configuration import validate_motion
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {}
    meshes = {}
    if kind == 'perceptive':
        metadata = root/'metadata.yaml'
        if not metadata.is_file():
            raise FileNotFoundError(f'{metadata}; use instinct prepare-scene --demo, or import a matched dataset')
        try:
            data = yaml.safe_load(metadata.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f'{metadata}: invalid YAML') from exc
        if not isinstance(data, dict):
            raise ValueError(f'{metadata}: expected a YAML object')
        terrain_entries = data.get('terrains')
        motion_entries = data.get('motion_files')
        if not isinstance(terrain_entries, list) or not terrain_entries:
            raise ValueError('Terrain dataset must contain a non-empty terrains list')
        if not isinstance(motion_entries, list) or not motion_entries:
            raise ValueError('Terrain dataset must contain a non-empty motion_files list')
        for entry in terrain_entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get('terrain_file'), str)
                    or not entry['terrain_file']):
                raise ValueError('Each terrain entry needs terrain_id and a non-empty terrain_file')
        terrain_ids = [entry.get('terrain_id') for entry in terrain_entries]
        if any(type(terrain_id) is not int or terrain_id < 0 for terrain_id in terrain_ids):
            raise ValueError('terrain_id values must be zero-based non-negative integers')
        if terrain_ids != list(range(len(terrain_ids))):
            raise ValueError('terrain_id values must be unique, ordered and contiguous from zero')
        terrains = {entry['terrain_id']: entry for entry in terrain_entries}
        terrain_files = [checked_file(root, entry['terrain_file']) for entry in terrain_entries]
        motions = []
        for entry in motion_entries:
            if (not isinstance(entry, dict) or type(entry.get('terrain_id')) is not int
                    or not isinstance(entry.get('motion_file'), str) or not entry['motion_file']):
                raise ValueError('Each motion entry needs an integer terrain_id and non-empty motion_file')
            if entry['terrain_id'] not in terrains:
                raise ValueError(f'Unknown terrain_id: {entry["terrain_id"]}')
            motions.append(checked_file(root, entry['motion_file']))
        referenced = [metadata, *motions, *terrain_files]
    elif kind == 'hoi':
        metadata = root/'objects.json'
        if not metadata.is_file():
            raise FileNotFoundError(f'{metadata}; HOI needs retargeted object trajectories and an objects.json mesh map')
        objects = json.loads(metadata.read_text())
        if not isinstance(objects, dict) or not objects:
            raise ValueError('objects.json must contain a non-empty object map')
        for name,item in objects.items():
            if not isinstance(name, str) or not name or '_' in name or not isinstance(item, dict):
                raise ValueError('HOI object names must be non-empty strings without underscores')
            if not isinstance(item.get('mesh'), str) or not item['mesh']:
                raise ValueError(f'HOI object {name} needs a non-empty mesh path')
            try:
                scale = np.asarray(item.get('scale', [1.,1.,1.]), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'HOI object {name} scale must contain three positive numbers') from exc
            if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f'HOI object {name} scale must contain three positive finite numbers')
            meshes[name] = dict(path=str(checked_file(root, item['mesh'])), scale=scale.tolist())
        motions = sorted(root.glob('*retargeted.npz'))
        if not motions:
            raise ValueError('HOI requires object meshes and subXX_OBJECTNAME_..._retargeted.npz motions')
        referenced = [metadata, *motions, *[Path(item['path']) for item in meshes.values()]]
    else:
        raise ValueError(f'Unknown scene data kind: {kind}')
    for motion in motions:
        try:
            validate_motion(motion)
            if kind == 'hoi':
                parts = motion.stem.split('_')
                if len(parts)<3 or parts[1] not in meshes:
                    raise ValueError(f'{motion}: object name must match objects.json')
                with np.load(motion, allow_pickle=False) as data:
                    n = len(data['joint_pos'])
                    for key,shape in [('object_pos_w',(n,3)),('object_quat_w',(n,4))]:
                        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
                            raise ValueError(f'{motion}: invalid {key}')
                    if not np.allclose(np.linalg.norm(data['object_quat_w'],axis=-1),1.,atol=1e-4):
                        raise ValueError(f'{motion}: object_quat_w must contain normalized wxyz quaternions')
        except (OSError, TypeError, KeyError, IndexError) as exc:
            raise ValueError(f'{motion}: invalid motion archive') from exc
    for file in referenced:
        files[str(file.relative_to(root))] = sha256(file)
    return dict(dataset=str(root), kind=kind, clips=len(motions), files=files, objects=meshes)


@contextmanager
def factory_data(spec):
    """临时为工厂绑定本地数据；深拷贝配置后恢复模块全局值，避免跨任务污染。

OneMotion 工厂在 __post_init__ 就读取 metadata，必须在调用工厂之前绑定，
不能等配置创建完毕后再覆盖开发者个人目录。
"""
    dataset = spec.get('scene_dataset')
    if not dataset:
        yield
        return
    kind = 'hoi' if 'HOI' in spec['task'] else 'perceptive'
    report = validate_dataset(ROOT/dataset, kind)
    expected = spec.get('scene_files')
    if expected is not None and expected != report['files']:
        raise ValueError('Scene dataset differs from saved model')
    if kind == 'hoi':
        module = importlib.import_module('instinct_mj.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg')
    else:
        module = importlib.import_module(PERCEPTIVE + ('perceptive_vae_cfg' if 'Vae' in spec['task'] else 'perceptive_shadowing_cfg'))
    saved = {}
    def replace(name, value):
        saved[name] = getattr(module,name)
        setattr(module,name,value)
    try:
        replace('OMOMO_DATASET_PATH' if kind=='hoi' else 'MOTION_FOLDER', report['dataset'])
        if kind=='hoi':
            replace('MESH_FILE_PATHS', {name:item['path'] for name,item in report['objects'].items()})
            replace('MESH_FILE_SCALES', {name:tuple(item['scale']) for name,item in report['objects'].items()})
        else:
            replace('PLAY_MOTION_FOLDER_OVERRIDE',None)
        for name in ('motion_reference_cfg','motion_reference_cfg_play'):
            cfg = deepcopy(getattr(module,name))
            if kind=='hoi':cfg.scene_object_names=list(report['objects'])
            for buffer in cfg.motion_buffers.values():
                buffer.path=report['dataset']
                if hasattr(buffer,'metadata_yaml'):buffer.metadata_yaml=str(Path(report['dataset'])/'metadata.yaml')
            replace(name,cfg)
        yield
    finally:
        for name,value in saved.items():setattr(module,name,value)


def apply_scene(cfg, spec, *, play=False):
    """仅示例模式缩小地形 tile 数；真实场景保留上游碰撞/传感器实现。"""
    if spec.get('scene_demo'):
        generator = cfg.scene.terrain.terrain_generator
        if generator is not None:
            generator.num_rows=generator.num_cols=1
            generator.curriculum=False
    if play:
        # Upstream play configs enable an OpenCV debug window unconditionally.
        # The workspace uses opencv-python-headless, and evaluate/export/deploy
        # must remain display-independent. World viewers still work normally.
        for group in cfg.observations.values():
            for term in group.terms.values():
                if 'debug_vis' in term.params:
                    term.params['debug_vis']=False
        for sensor in cfg.scene.sensors:
            if getattr(sensor,'name',None)=='camera' and hasattr(sensor,'debug_vis'):
                sensor.debug_vis=False
    # 相机任务不能沿用普通任务的未分组射线配置；所有传感器维持上游类型。

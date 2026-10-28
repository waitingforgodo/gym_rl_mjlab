"""Instinct 后端的模型隔离、动作根转换和独立部署契约回归。"""
from dataclasses import dataclass,field
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from workspace.common.contract import SDK_JOINTS
from workspace.instinct import checkpoints
from workspace.instinct.configuration import apply_overrides,bind_teacher,runner_configuration
from workspace.instinct.paths import MJ_SOURCE
from workspace.instinct.scenes import validate_dataset
from workspace.instinct.training import configured_profile,prepare_teacher,validate_data_contract

ROOT=Path(__file__).resolve().parents[1]


class InstinctToolchainTests(unittest.TestCase):
    def test_foreign_checkpoint_is_rejected_without_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model_1.pt';p.touch()
            (p.parent/'run_manifest.json').write_text(json.dumps(dict(learning_method='amp')))
            with self.assertRaisesRegex(ValueError,'not from this InstinctMJ'):
                checkpoints.checkpoint(p)

    def test_latest_is_filtered_by_profile_and_explicit_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            expected={}
            for name in ('locomotion','beyondmimic'):
                run=root/'runs'/name;run.mkdir(parents=True)
                p=run/'model_2.pt';p.touch();expected[name]=p
                (run/'run_manifest.json').write_text(json.dumps(dict(backend='instinct_mj',profile=name)))
                # 时间更晚的旧编号和非数字临时文件都不应覆盖本运行的最终模型。
                (run/'model_1.pt').touch()
                (run/'model_latest.pt').touch()
            foreign=root/'runs'/'foreign';foreign.mkdir()
            (foreign/'run_manifest.json').write_text(json.dumps(dict(backend='other')))
            (foreign/'model_100.pt').touch()
            with patch.object(checkpoints,'OUTPUTS',root):
                for name,p in expected.items():
                    self.assertEqual(checkpoints.checkpoint('latest',name)[0],p)
            with self.assertRaisesRegex(ValueError,'profile differs'):
                checkpoints.checkpoint(expected['beyondmimic'],'locomotion')

    def test_configuration_override_rejects_typos_and_private_fields(self):
        @dataclass
        class Algorithm:
            lr:float=.001
        @dataclass
        class Config:
            algorithm:Algorithm=field(default_factory=Algorithm)
        cfg=Config()
        apply_overrides(cfg,{'algorithm.lr':.0003})
        self.assertEqual(cfg.algorithm.lr,.0003)
        for name in ('algorithm.laerning_rate','algorithm.__dict__'):
            with self.assertRaises(ValueError):apply_overrides(cfg,{name:1.})

    def test_vae_teacher_binding_never_uses_upstream_developer_path(self):
        @dataclass
        class Algorithm:
            teacher_logdir:str='developer/path'
        @dataclass
        class Agent:
            algorithm:Algorithm=field(default_factory=Algorithm)
        spec=dict(task='Instinct-Perceptive-Vae-G1-v0',teacher_checkpoint='/runs/teacher/model_7.pt')
        agent=Agent();bind_teacher(agent,spec)
        self.assertEqual(agent.algorithm.teacher_logdir,'/runs/teacher')
        bind_teacher(agent,spec,play=True)
        self.assertIsNone(agent.algorithm.teacher_logdir)
        with self.assertRaisesRegex(ValueError,'requires --teacher-checkpoint'):
            bind_teacher(Agent(),dict(task=spec['task']))
        with self.assertRaisesRegex(ValueError,'only valid'):
            bind_teacher(Agent(),dict(task='Instinct-Locomotion-Flat-G1-v0',teacher_checkpoint='model.pt'))

    def test_teacher_observation_shapes_survive_runner_serialization(self):
        class Agent:
            def to_dict(self):
                return {'algorithm':{'teacher_policy':{'obs_format':{
                    'policy':{'depth_image':[1,18,32]},'critic':{'joint_pos':[232]}}}}}
        result=runner_configuration(Agent())
        teacher=result['algorithm']['teacher_policy']['obs_format']
        self.assertEqual(teacher['policy']['depth_image'],(1,18,32))
        self.assertEqual(teacher['critic']['joint_pos'],(232,))

    def test_task_data_contract_rejects_ignored_or_conflicting_inputs(self):
        with self.assertRaisesRegex(ValueError,'only one'):
            validate_data_contract(dict(task='Instinct-BeyondMimic-Plane-G1-v0',motion='a',motion_dir='b'))
        with self.assertRaisesRegex(ValueError,'scene_dataset is only valid'):
            validate_data_contract(dict(task='Instinct-Locomotion-Flat-G1-v0',scene_dataset='scene'))
        with self.assertRaisesRegex(ValueError,'does not accept'):
            validate_data_contract(dict(task='Instinct-Locomotion-Flat-G1-v0',motion='a'))
        with self.assertRaisesRegex(ValueError,'do not also set'):
            validate_data_contract(dict(task='Instinct-Perceptive-Shadowing-G1-v0',
                                        scene_dataset='scene',motion='a'))
        with self.assertRaisesRegex(ValueError,'requires motion'):
            validate_data_contract(dict(task='Instinct-BeyondMimic-Plane-G1-v0'))
        with self.assertRaisesRegex(ValueError,'requires scene_dataset'):
            validate_data_contract(dict(task='Instinct-Perceptive-Vae-G1-v0'))

    def test_custom_config_cannot_replace_task_or_data_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            path.write_text(json.dumps({'profile':'parkour','env_overrides':{'episode_length_s':12.}}))
            spec=configured_profile(None,path)
            self.assertEqual(spec['profile'],'parkour')
            self.assertEqual(spec['env_overrides']['episode_length_s'],12.)
            self.assertIn('scene.terrain.terrain_generator.num_rows',spec['env_overrides'])
            path.write_text(json.dumps({'profile':'parkour','task':'Instinct-Locomotion-Flat-G1-v0'}))
            with self.assertRaisesRegex(ValueError,'Unknown custom config fields'):
                configured_profile(None,path)
            path.write_text(json.dumps({'profile':'parkour'}))
            with self.assertRaisesRegex(ValueError,'differs'):
                configured_profile('locomotion',path)
            path.write_text(json.dumps({'profile':'../../outside'}))
            with self.assertRaisesRegex(ValueError,'Invalid Instinct profile name'):
                configured_profile(None,path)

    def test_vae_teacher_is_source_checked_and_hash_pinned(self):
        source=dict(backend='instinct_mj',sources=[{'commit':'abc'}],runtime_versions={'mjlab':'1'})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'teacher';run.mkdir()
            model=run/'model_7.pt';model.write_bytes(b'teacher')
            (run/'params').mkdir();(run/'params/agent.yaml').touch()
            manifest=dict(**source,profile='perceptive',spec={
                'task':'Instinct-Perceptive-Shadowing-G1-v0'})
            (run/'run_manifest.json').write_text(json.dumps(manifest))
            spec=dict(task='Instinct-Perceptive-Vae-G1-v0',teacher_checkpoint=str(model))
            prepare_teacher(spec,source)
            self.assertEqual(spec['teacher_checkpoint'],str(model))
            self.assertEqual(spec['teacher_sha256'],hashlib.sha256(b'teacher').hexdigest())
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'differs from the saved run'):
                prepare_teacher(spec,source,resume=True)

    def test_help_and_task_catalog_do_not_import_gpu_frameworks(self):
        commands=('tasks','doctor','setup','summary','prepare','prepare-lafan','prepare-scene',
                  'train','resume','play','evaluate','export','deploy')
        for command in commands:
            code=f'''import sys
from workspace.instinct.cli import main
try:main(["{command}","--help"])
except SystemExit as exc:assert exc.code==0
assert not {{"torch","warp","mjlab","instinct_rl"}} & sys.modules.keys()
'''
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_corrupt_bundle_is_rejected_before_runtime_initialization(self):
        from types import SimpleNamespace
        from workspace.instinct.deployment import deploy
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            manifest=p/'manifest.json';manifest.write_text(json.dumps(dict(backend='instinct_mj')))
            (p/'policy_jit.pt').write_bytes(b'changed')
            (p/'policy.onnx').write_bytes(b'onnx')
            (p/'hashes.json').write_text(json.dumps({
                'manifest.json':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'policy_jit.pt':hashlib.sha256(b'original').hexdigest(),
                'policy.onnx':hashlib.sha256(b'onnx').hexdigest()}))
            with self.assertRaisesRegex(ValueError,'integrity differs'):
                deploy(SimpleNamespace(bundle=p))

    def test_bundle_hash_paths_cannot_escape_bundle(self):
        from types import SimpleNamespace
        from workspace.instinct.deployment import deploy
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'manifest.json').write_text(json.dumps(dict(backend='instinct_mj')))
            (p/'policy_jit.pt').touch();(p/'policy.onnx').touch()
            (p/'hashes.json').write_text(json.dumps({'../outside':hashlib.sha256(b'').hexdigest()}))
            with self.assertRaisesRegex(ValueError,'integrity differs'):
                deploy(SimpleNamespace(bundle=p))

    def test_bundle_data_paths_cannot_escape_bundle(self):
        from types import SimpleNamespace
        from workspace.instinct.deployment import deploy
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            manifest=p/'manifest.json'
            manifest.write_text(json.dumps(dict(
                backend='instinct_mj',spec={'motion':'../outside_retargeted.npz'})))
            jit=p/'policy_jit.pt';jit.touch()
            onnx=p/'policy.onnx';onnx.touch()
            (p/'hashes.json').write_text(json.dumps({
                'manifest.json':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'policy_jit.pt':hashlib.sha256(jit.read_bytes()).hexdigest(),
                'policy.onnx':hashlib.sha256(onnx.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError,'must stay inside'):
                deploy(SimpleNamespace(bundle=p))

    def test_supplementary_requirements_do_not_replace_the_shared_stack(self):
        req=(ROOT/'workspace/instinct/requirements-addons.txt').read_text().splitlines()
        blocked={'torch','torchvision','numpy','scipy','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib'}
        for line in req:
            if line and not line.startswith('#'):
                self.assertIn('==',line)
                self.assertNotIn(line.split('==')[0].lower(),blocked)

    def test_perceptive_metadata_requires_native_contiguous_integer_ids(self):
        import numpy as np
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'floor.stl').touch()
            np.savez(root/'demo_retargeted.npz',framerate=50.,
                     joint_names=np.array(SDK_JOINTS),
                     joint_pos=np.zeros((3,29)),base_pos_w=np.zeros((3,3)),
                     base_quat_w=np.tile([1.,0.,0.,0.],(3,1)))
            metadata=dict(terrains=[dict(terrain_id='flat_demo',terrain_file='floor.stl')],
                          motion_files=[dict(terrain_id='flat_demo',motion_file='demo_retargeted.npz')])
            (root/'metadata.yaml').write_text(yaml.safe_dump(metadata))
            with self.assertRaisesRegex(ValueError,'zero-based non-negative integers'):
                validate_dataset(root,'perceptive')
            metadata['terrains'][0]['terrain_id']=0
            metadata['motion_files'][0]['terrain_id']=0
            (root/'metadata.yaml').write_text(yaml.safe_dump(metadata))
            self.assertEqual(validate_dataset(root,'perceptive')['clips'],1)

    def test_hoi_dataset_validates_mesh_scale_and_object_quaternions(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'box.obj').touch()
            objects=root/'objects.json'
            objects.write_text(json.dumps({'box':{'mesh':'box.obj','scale':[1.,0.,1.]}}))
            with self.assertRaisesRegex(ValueError,'positive finite'):
                validate_dataset(root,'hoi')
            objects.write_text(json.dumps({'box':{'mesh':'box.obj','scale':[1.,1.,1.]}}))
            motion=root/'sub01_box_lift_retargeted.npz'
            arrays=dict(framerate=np.array([50.]),joint_names=np.array(SDK_JOINTS),
                        joint_pos=np.zeros((3,29)),base_pos_w=np.zeros((3,3)),
                        base_quat_w=np.tile([1.,0.,0.,0.],(3,1)),
                        object_pos_w=np.zeros((3,3)),object_quat_w=np.zeros((3,4)))
            np.savez(motion,**arrays)
            with self.assertRaisesRegex(ValueError,'normalized wxyz'):
                validate_dataset(root,'hoi')
            arrays['object_quat_w']=np.tile([1.,0.,0.,0.],(3,1))
            np.savez(motion,**arrays)
            self.assertEqual(validate_dataset(root,'hoi')['clips'],1)

    @unittest.skipUnless(importlib.util.find_spec('mujoco'),'requires pinned MuJoCo; no GPU needed')
    def test_torso_root_conversion_inverts_waist_and_preserves_source(self):
        info=json.loads((MJ_SOURCE/'SOURCE_INFO.json').read_text())
        if any(not (MJ_SOURCE/name).is_file() for name in info['assets']):
            self.skipTest('Instinct asset setup has not been run')
        import mujoco
        import numpy as np
        import mjlab
        from workspace.instinct.data import prepare
        old_model=mujoco.MjModel.from_xml_path(str(Path(mjlab.__file__).parent/'asset_zoo/robots/unitree_g1/xmls/g1.xml'))
        old=mujoco.MjData(old_model)
        joints=[mujoco.mj_id2name(old_model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,old_model.njnt)]
        bodies=[mujoco.mj_id2name(old_model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,old_model.nbody)]
        q=np.zeros((4,29));q[:,joints.index('waist_yaw_joint')]=.3
        q[:,joints.index('waist_roll_joint')]=.12;q[:,joints.index('waist_pitch_joint')]=-.2
        pos=[];quat=[]
        for frame in range(4):
            old.qpos[:3]=[.01*frame,0,.8];old.qpos[3:7]=[1,0,0,0];old.qpos[7:]=q[frame]
            mujoco.mj_forward(old_model,old)
            pos.append(old.xpos[1:].copy());quat.append(old.xquat[1:].copy())
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.npz';target=Path(folder)/'motion_retargeted.npz'
            np.savez(source,fps=50.,joint_names=np.array(joints),body_names=np.array(bodies),
                     joint_pos=q,joint_vel=np.zeros_like(q),body_pos_w=np.array(pos),body_quat_w=np.array(quat),
                     body_lin_vel_w=np.zeros_like(pos),body_ang_vel_w=np.zeros_like(pos))
            original=source.read_bytes()
            result=prepare(source,target)
            with np.load(target,allow_pickle=False) as data:
                names=data['joint_names'].tolist()
                for name in ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint'):
                    np.testing.assert_allclose(data['joint_pos'][:,names.index(name)],-q[:,joints.index(name)])
                np.testing.assert_allclose(data['base_pos_w'],np.array(pos)[:,bodies.index('torso_link')])
            self.assertLess(result['sampled_fk_max_position_difference_m'],.03)
            self.assertTrue(prepare(source,target)['reused'])
            self.assertEqual(source.read_bytes(),original)


if __name__=='__main__':unittest.main()

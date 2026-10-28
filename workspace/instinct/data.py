"""把现有标准 NPZ 转为 Instinct 的 torso-root 格式，按名字映射关节。

当前模型的自由根为 pelvis，Instinct 模型的自由根为 torso_link；转换必须
同时处理根刚体及 waist 关节，不能仅重排 29 列或改文件键名。
"""
from pathlib import Path
from .paths import ROOT,MJ_SOURCE
from workspace.common.contract import sha256,write_json
from workspace.common.motion import scalar_float


def prepare(source,output=None):
    import numpy as np
    import mujoco
    from workspace.common.motion import inspect
    inspect(source,expected_fps=50.)
    target=Path(output or ROOT/'motions/processed/instinct'/f'{Path(source).stem}_retargeted.npz').resolve()
    if target==Path(source).resolve():raise ValueError('Do not overwrite the original motion')
    if target.exists():
        import json
        info=target.with_suffix('.json')
        cached=json.loads(info.read_text()) if info.is_file() else {}
        if (cached.get('source_sha256')==sha256(source) and cached.get('output_sha256')==sha256(target)
                and len(cached.get('inverted_joints',[]))==3
                and cached.get('sampled_fk_max_position_difference_m',1.)<=.03):
            return dict(cached,reused=True)
        raise ValueError('Output already exists with a different source/conversion; choose --output for a new motion')
    model=mujoco.MjModel.from_xml_path(str(MJ_SOURCE/'src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml'))
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,model.njnt)]
    with np.load(source,allow_pickle=False) as old:
        old_names=old['joint_names'].tolist()
        if set(names)!=set(old_names):raise ValueError('Joint names differ between the two G1 models')
        torso=old['body_names'].tolist().index('torso_link')
        q=old['joint_pos'][:,[old_names.index(n) for n in names]].copy()
        # 原模型：pelvis→yaw→roll→pitch→torso。
        # 新模型：torso→pitch→roll→yaw→pelvis，轴方向仍为原正方向。
        # 链条反向后，三个位姿变换取逆，因此 waist 三角度必须取负。
        for i,name in enumerate(names):
            if name.startswith('waist_'):q[:,i]*=-1.
        positions=old['body_pos_w'][:,torso].copy()
        quaternions=old['body_quat_w'][:,torso].copy()
        fps=scalar_float(old['fps'],'fps')
        old_bodies=old['body_names'].tolist()
        sample_ids=np.unique(np.linspace(0,len(q)-1,min(32,len(q)),dtype=int))
        model_data=mujoco.MjData(model)
        comparable=[(i,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)) for i,n in enumerate(old_bodies)]
        comparable=[(i,j) for i,j in comparable if j>=0]
        errors=[]
        for frame in sample_ids:
            model_data.qpos[:3]=positions[frame]
            model_data.qpos[3:7]=quaternions[frame]
            model_data.qpos[7:]=q[frame]
            mujoco.mj_forward(model,model_data)
            errors.extend(float(np.linalg.norm(model_data.xpos[j]-old['body_pos_w'][frame,i])) for i,j in comparable)
        # 模型骨架/关节方向必须确实对应，超过 3 cm 就拒绝此快捷转换；
        # 真正不同的机器人需要重新 retarget，不能靠列名掩盖模型差异。
        max_error=max(errors)
        if max_error>.03:
            raise ValueError(f'Model FK differs by {max_error:.4f} m; retarget motion to Instinct G1 instead')
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,framerate=fps,joint_names=np.array(names),joint_pos=q,
                        base_pos_w=positions,base_quat_w=quaternions)
    metadata=dict(source=str(Path(source).resolve()),source_sha256=sha256(source),
                  output=str(target),output_sha256=sha256(target),root_body='torso_link',
                  joint_names=names,frames=len(q),fps=fps,quaternion='wxyz',
                  inverted_joints=[name for name in names if name.startswith('waist_')],
                  sampled_fk_max_position_difference_m=max_error)
    write_json(target.with_suffix('.json'),metadata)
    return metadata

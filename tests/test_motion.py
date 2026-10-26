"""动作数据回归：时间尺度、xyzw/wxyz、SLERP、世界系角速度和 NPZ 契约。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workspace.motion import (angular_velocity_world, continuous_quat, inspect,
                              quat_multiply, resample_csv)


class MotionTests(unittest.TestCase):
    """用有解析答案的小轨迹检查数学语义，避免把代码输出本身作为答案。"""
    def test_resample_preserves_seconds_and_xyzw_conversion(self):
        data = np.zeros((61, 36))
        data[:, 0] = np.arange(61) / 60
        data[:, 6] = 1  # CSV xyzw identity.
        data[:, 7:] = data[:, 0, None] * np.arange(29)
        xyz, quat, joints = resample_csv(data, 60, 50)
        self.assertEqual(len(xyz), 51)
        np.testing.assert_allclose(xyz[:, 0], np.arange(51)/50)
        np.testing.assert_allclose(quat, np.tile([1, 0, 0, 0], (51, 1)))
        np.testing.assert_allclose(joints[-1], np.arange(29))

    def test_quaternion_sign_change_does_not_create_rotation(self):
        data = np.zeros((10, 36))
        data[:, 6] = 1
        data[1::2, 6] = -1
        _, q, _ = resample_csv(data, 60, 50)
        np.testing.assert_allclose(angular_velocity_world(q, 50), 0, atol=1e-8)

    def test_slerp_constant_yaw_rate(self):
        data = np.zeros((61, 36))
        angle = np.arange(61) / 60 * .8
        data[:, 5], data[:, 6] = np.sin(angle/2), np.cos(angle/2)
        _, q, _ = resample_csv(data, 60, 50)
        omega = angular_velocity_world(q, 50)
        np.testing.assert_allclose(omega[:, 2], .8, atol=1e-8)
        np.testing.assert_allclose(omega[:, :2], 0, atol=1e-8)

    def test_angular_velocity_is_world_not_body_frame(self):
        # Start with 90 degrees around X, then rotate about WORLD Z.
        angle = np.arange(101) / 100
        world_z = np.stack((np.cos(angle/2), np.zeros(101), np.zeros(101), np.sin(angle/2)), axis=-1)
        base = np.broadcast_to(np.array([2**-.5, 2**-.5, 0, 0]), world_z.shape)
        q = quat_multiply(world_z, base)
        omega = angular_velocity_world(q, 100)
        np.testing.assert_allclose(omega[:, 2], 1, atol=1e-8)
        np.testing.assert_allclose(omega[:, :2], 0, atol=1e-8)

    def test_invalid_csv_rejected(self):
        for data in (np.zeros((5, 35)), np.zeros((1, 36)), np.zeros((5, 36)), np.full((5, 36), np.nan)):
            with self.assertRaises(ValueError):
                resample_csv(data, 60, 50)

    def test_named_motion_rejects_wrong_order_or_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'motion.npz'
            arrays = dict(fps=np.array(50), joint_names=np.array(['a', 'b']), body_names=np.array(['base']),
                          joint_pos=np.zeros((3, 2)), joint_vel=np.zeros((3, 2)), body_pos_w=np.zeros((3, 1, 3)),
                          body_quat_w=np.tile([[[1., 0, 0, 0]]], (3, 1, 1)), body_lin_vel_w=np.zeros((3, 1, 3)),
                          body_ang_vel_w=np.zeros((3, 1, 3)))
            np.savez(file, **arrays)
            self.assertEqual(inspect(file, ['a', 'b'], ['base'], 50)['frames'], 3)
            with self.assertRaises(ValueError):
                inspect(file, ['b', 'a'], ['base'], 50)
            with self.assertRaises(ValueError):
                inspect(file, ['a', 'b'], ['base'], 60)


if __name__ == '__main__':
    unittest.main()

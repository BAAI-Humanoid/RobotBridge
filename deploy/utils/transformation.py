import numpy as np
import transforms3d
from scipy.spatial.transform import Rotation as sRot
from typing import Tuple


def matrix_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    return sRot.from_quat(quat_xyzw).as_matrix()

def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray, q02: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r01 = sRot.from_quat(q01)
    r10 = r01.inv()
    r02 = sRot.from_quat(q02)
    r12 = r10 * r02
    t_rel = r10.apply(t02 - t01)
    return t_rel.astype(np.float32), r12.as_quat().astype(np.float32)

def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Keep consistent with LEVEL deployment script for quaternion inverse rotation."""
    w, x, y, z = q_wxyz
    q_conj = np.array([w, -x, -y, -z])
    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ],
        dtype=np.float32,
    )

def pos_quat_to_T(pos, quat, quat_format='xyzw'):
    """
    using transforms3d to convert pos + quat to 4x4 transformation matrix (support batch)
    
    Args:
        pos: (..., 3) array
        quat: (..., 4) array, quaternion
        quat_format: 'xyzw' or 'wxyz'
    Returns:
        T: (..., 4, 4) transformation matrix
    """
    pos = np.array(pos)
    quat = np.array(quat)
    
    # ensure shape consistency
    assert pos.shape[-1] == 3
    assert quat.shape[-1] == 4
    
    # flatten leading dimensions for looping
    leading_shape = pos.shape[:-1]
    pos_flat = pos.reshape(-1, 3)
    quat_flat = quat.reshape(-1, 4)
    
    # batch construct transformation matrix
    T_flat = np.zeros((len(pos_flat), 4, 4))
    for i in range(len(pos_flat)):
        # using transforms3d.quaternions.quat2mat
        R = transforms3d.quaternions.quat2mat(quat_flat[i])  # [3,3]
        T_flat[i] = transforms3d.affines.compose(
            pos_flat[i], R, np.ones(3)  # translation, rotation, scale (set to 1)
        )
    
    # restore original shape
    T = T_flat.reshape(*leading_shape, 4, 4)
    return T

def T_to_pos_quat(T):
    """
    from T to pos and quat
    """
    pos = T[:3, 3]
    rot_matrix = T[:3, :3]
    quat = transforms3d.quaternions.mat2quat(rot_matrix)
    quat = np.roll(quat, -1)  # transfer to xyzw format

    return pos, quat

def pelvis2root(T_world_pelvis, only_yaw=False):
        """only apply when self.only_yaw == True"""
        if only_yaw:
            # trans T_hip's R to q
            R_world_pelvis = T_world_pelvis[:3, :3]
            q_world_pelvis = transforms3d.quaternions.mat2quat(R_world_pelvis)  # wxyz
            
            # rotate R_hips's z axis to world z namely (0,0,1)           
            # rotate theta | cos(theta) = z_axis_hip . z_axis_world = z3
            # cos(theta/2) = sqrt((1+z3)/2) | sin(theta/2) = sqrt((1-z3)/2)
            # rotate axis = z_axis_hip cross z_axis_world = (z2, -z1, 0)
            # q = [x,y,z,w]
            # q = [sin(theta/2)*z2, sin(theta/2)*(-z1), 0, cos(theta/2)]
            # q = [z2/sqrt(2(1+z3)), -z1/sqrt(2(1+z3)), 0, sqrt((1+z3)/2)]
            q_world_pelvis = q_world_pelvis / np.linalg.norm(q_world_pelvis)  # normalize
            z1 = q_world_pelvis[1]  # x component
            z2 = q_world_pelvis[2]  # y component
            z3 = q_world_pelvis[3]  # z component

            # compute diff quaternion
            denom = np.sqrt(2 * (1 + z3))
            x = z2 / denom
            y = -z1 / denom
            z = 0.0
            w = np.sqrt((1 + z3) / 2)
            q_diff = np.array([w, x, y, z]) # transforms3d uses [w,x,y,z]
            
            # Normalize diff quaternion
            q_diff = q_diff / np.linalg.norm(q_diff)
            
            # q_world_root = q_diff * q_world_pelvis
            q_world_root = quaternions.qmult(q_diff, q_world_pelvis)

            # Convert back to rotation matrix
            R_world_root = quaternions.quat2mat(q_world_root)

            # Build output transform
            T_world_root = np.eye(4)
            T_world_root[:3, :3] = R_world_root
            T_world_root[:3, 3] = T_world_pelvis[:3, 3]  # same position
            return T_world_root
        else:
            return T_world_pelvis
    
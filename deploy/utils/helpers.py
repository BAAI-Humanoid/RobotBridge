import numpy as np

def parse_observation(cls, key_list, buf_dict, obs_scales):
    """
    Enhanced parse_observation with noise support (similar to URCIRobotResidual)
    """
    for obs_key in key_list:
        actor_obs = getattr(cls, f'_get_obs_{obs_key}')().copy()
        obs_scale = np.array(obs_scales[obs_key], dtype=np.float32)
        
        # Apply scaling
        scaled_obs = actor_obs * obs_scale
        buf_dict[obs_key] = scaled_obs

def get_gravity(quat,w_last=True):
    if w_last:
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    else:
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    gravity = np.zeros(3)
    gravity[0] = 2*(-qz*qx + qw*qy)
    gravity[1] = -2*(qz*qy + qw*qx)
    gravity[2] = 1 - 2*(qw*qw + qz*qz)

    return gravity

def quaternion_to_euler_array(quat:np.ndarray)->np.ndarray:
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # w, x, y, z = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])



def get_rpy(quat, w_last=True):
    if w_last:
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    else:
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    
    sinr_cosp = 2.0 * (qw*qx+qy*qz)
    cosr_cosp = qw*qw - qx*qx - qy*qy +qz*qz
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2*(qw*qy - qz*qx)
    pitch = np.where(
        np.abs(sinp)>=1, np.abs(np.pi/2)*np.sign(sinp), np.arcsin(sinp)
    )
    
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = qw*qw + qx*qx - qy*qy -qz*qz
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.stack([roll, pitch, yaw])
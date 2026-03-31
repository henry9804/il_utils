import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point, Pose, Transform

class TF_mat:
    def __init__(self, T=None):
        if T is None:
            self.T = np.identity(4)
        else:
            self.T = T

    @classmethod
    def from_vectors(cls, pos, quat):
        pos = np.array(pos)
        quat = np.array(quat)
        if len(pos.shape) == 2:
            tf_mat = cls(np.zeros([pos.shape[0], 4, 4]))
            tf_mat.T[:,3,3] = 1
        else:
            tf_mat = cls()
        tf_mat.T[...,:3,:3] = Rotation.from_quat(quat).as_matrix()
        tf_mat.T[...,:3,3] = pos

        return tf_mat

    @classmethod
    def from_mat(cls, pos, rotm):
        pos = np.array(pos)
        rotm = np.array(rotm)
        if len(pos.shape) == 2:
            tf_mat = cls(np.zeros([pos.shape[0], 4, 4]))
            tf_mat.T[:,3,3] = 1
        else:
            tf_mat = cls()
        tf_mat.T[...,:3,:3] = rotm
        tf_mat.T[...,:3,3] = pos

        return tf_mat
    
    @classmethod
    def from_msg(cls, msg):
        if isinstance(msg, Pose):
            pos = np.array([msg.position.x, msg.position.y, msg.position.z])
            quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        elif isinstance(msg, Transform):
            pos = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
            quat = [msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]
        elif isinstance(msg, Point):
            pos = np.array([msg.x, msg.y, msg.z])
            quat = [0.0, 0.0, 0.0, 1.0]
        else:
            raise NotImplementedError

        tf_mat = cls()
        tf_mat.T[:3,:3] = Rotation.from_quat(quat).as_matrix()
        tf_mat.T[:3,3] = pos

        return tf_mat
    
    @classmethod
    def mul(cls, tf1, tf2):
        tf_mat = cls(np.matmul(tf1.T, tf2.T))

        return tf_mat
    
    def inverse(self):
        p = self.T[...,:3,3:]
        R = self.T[...,:3,:3]
        inv = np.zeros_like(self.T)
        inv[...,:3,:3] = R.swapaxes(-1, -2)
        inv[...,:3,3:] = -np.matmul(R.swapaxes(-1, -2), p)
        inv[...,3,3] = 1

        return TF_mat(inv)
    
    def as_matrix(self):
        return self.T
    
    def as_vectors(self):
        p = self.T[...,:3,3]
        R = self.T[...,:3,:3]
        q = Rotation.from_matrix(R).as_quat()

        return p, q
    
    def as_pose_msg(self):
        assert len(self.T.shape) == 2, 'as_pose_msg() is not available for batched TF_mat'
        p = self.T[:3,3]
        R = self.T[:3,:3]
        q = Rotation.from_matrix(R).as_quat()

        pose = Pose()
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = p[2]
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose
    
    def get_pos(self):
        return self.T[..., :3,3].copy()
    
    def get_rotm(self):
        return self.T[..., :3,:3].copy()

def cubic(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    if time < time_0:
        return x_0
    elif time > time_f:
        return x_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x = x_f - x_0

        x_t = x_0 + x_dot_0 * elapsed_time \
              + (3 * total_x / total_time2 - 2 * x_dot_0 / total_time - x_dot_f / total_time) \
              * elapsed_time * elapsed_time \
              + (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2) \
              * elapsed_time * elapsed_time * elapsed_time
        
        return x_t

def cubicVector(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    res = np.zeros_like(x_0)
    for i in range(len(x_0)):
        res[i] = cubic(time, time_0, time_f, x_0[i], x_f[i], x_dot_0[i], x_dot_f[i])
    return res
    
def tf_cubic_spline(t, t_0, t_f, TF_0: TF_mat, TF_f: TF_mat):
    p_0 = TF_0.get_pos()
    p_f = TF_f.get_pos()
    p = cubicVector(t, t_0, t_f, p_0, p_f, np.zeros_like(p_0), np.zeros_like(p_f))

    R_0 = TF_0.get_rotm()
    R_f = TF_f.get_rotm()
    R_diff = R_0.transpose() @ R_f
    rotvec = Rotation.from_matrix(R_diff).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle == 0:
        R = R_f
    else:
        axis = rotvec / angle
        
        angle_t = cubic(t, t_0, t_f, 0.0, angle, 0.0, 0.0)
        R_diff_t = Rotation.from_rotvec(angle_t * axis).as_matrix()
        R = R_0 @ R_diff_t

    return TF_mat.from_mat(p, R)
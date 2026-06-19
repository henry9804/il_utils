import numpy as np
from scipy.linalg import logm, expm

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

def cubicDot(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    if time < time_0:
        return x_dot_0
    elif time > time_f:
        return x_dot_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x = x_f - x_0

        x_t = x_dot_0 \
              + 2 * (3 * total_x / total_time2 - 2 * x_dot_0 / total_time - x_dot_f / total_time) \
              * elapsed_time \
              + 3 * (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2) \
              * elapsed_time * elapsed_time
        
        return x_t

def cubicVector(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    res = np.zeros_like(x_0)
    for i in range(len(x_0)):
        res[i] = cubic(time, time_0, time_f, x_0[i], x_f[i], x_dot_0[i], x_dot_f[i])
    return res

def cubicDotVector(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    res = np.zeros_like(x_0)
    for i in range(len(x_0)):
        res[i] = cubicDot(time, time_0, time_f, x_0[i], x_f[i], x_dot_0[i], x_dot_f[i])
    return res

def rotationCubic(time, time_0, time_f, rotation_0, rotation_f):
    if time >= time_f:
        return rotation_f
    elif time < time_0:
        return rotation_0
    tau = cubic(time, time_0, time_f, 0, 1, 0, 0)
    rot_scaler_skew = logm(rotation_0.T @ rotation_f)
    return rotation_0 @ expm(rot_scaler_skew * tau)

def rotationCubicDot(time, time_0, time_f, rotation_0, rotation_f):
    result = np.zeros(3)
    if time >= time_f or time < time_0:
        return result
    rotation_d = rotation_f @ rotation_0.T
    theta = np.arccos(np.clip((rotation_d[0, 0] + rotation_d[1, 1] + rotation_d[2, 2] - 1) / 2, -1.0, 1.0))
    theta_dot = cubicDot(time, time_0, time_f, 0, theta, 0, 0)
    w = np.array([
        1 / (2 * np.sin(theta)) * (rotation_d[2, 1] - rotation_d[1, 2]),
        1 / (2 * np.sin(theta)) * (rotation_d[0, 2] - rotation_d[2, 0]),
        1 / (2 * np.sin(theta)) * (rotation_d[1, 0] - rotation_d[0, 1]),
    ])
    return theta_dot * w

def getPhi(current_rotation, desired_rotation):
    s = [np.cross(current_rotation[:, i], desired_rotation[:, i]) for i in range(3)]
    return -0.5 * (s[0] + s[1] + s[2])
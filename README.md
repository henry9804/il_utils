# il_utils

Utilities for imitation learning data collection and preprocessing with ROS 2.

## Installation

```bash
pip install -e .
```

> ROS 2 packages (`rclpy`, `geometry_msgs`, etc.) must be installed separately via your ROS 2 distribution.

## Package structure

```
il_utils/
├── transform.py          # TF_mat, cubic spline interpolation
├── dyros_math_python.py  # cubic / cubicDot scalar & vector helpers
└── ros2_configs/         # robot-specific ROS 2 topic configs
    ├── dual_fr3.py
    ├── tocabi.py
    └── panda_maniskill.py

scripts/
├── lerobot_record_ros2.py   # record episodes from ROS 2 topics → LeRobot dataset
└── hdf5_to_lerobot.py       # convert ACT-style HDF5 files → LeRobot dataset
```

## Usage

### Transform utilities

```python
from il_utils.transform import TF_mat, tf_cubic_spline

# construct from position + quaternion [x, y, z, w]
tf = TF_mat.from_vectors(pos, quat)

# construct from position + rotation matrix
tf = TF_mat.from_mat(pos, rotm)

# construct from ROS 2 Pose / Transform / Point message
tf = TF_mat.from_msg(msg)

# compose transforms
tf_result = TF_mat.mul(tf_a, tf_b)

# invert
tf_inv = tf.inverse()

# extract position and quaternion
pos, quat = tf.as_vectors()

# interpolate between two transforms
tf_t = tf_cubic_spline(t, t_0, t_f, TF_0, TF_f)
```

### Math utilities

```python
import il_utils.dyros_math_python as dyros_math

x   = dyros_math.cubic(t, t0, tf, x0, xf, dx0, dxf)
dx  = dyros_math.cubicDot(t, t0, tf, x0, xf, dx0, dxf)
xv  = dyros_math.cubicVector(t, t0, tf, x0, xf, dx0, dxf)
dxv = dyros_math.cubicDotVector(t, t0, tf, x0, xf, dx0, dxf)
```

### Robot configs

```python
from il_utils import ros2_configs

cfg = ros2_configs.load("dual_fr3")
cfg = ros2_configs.load("tocabi")
cfg = ros2_configs.load("panda_maniskill")
```

Each config defines `fps`, `state`, `action`, `end_effector`, and `image` fields with ROS 2 topic mappings.

## Scripts

### Record episodes from ROS 2

```bash
python scripts/lerobot_record_ros2.py --robot-type dual_fr3 --lerobot-id <repo/dataset-id> [options]
```

### Convert HDF5 to LeRobot dataset

```bash
python scripts/hdf5_to_lerobot.py --dataset-dir <path> --lerobot-id <repo/dataset-id> [--task <str>] [--fps 30]
```

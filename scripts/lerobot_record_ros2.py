import os
import argparse
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, Int32MultiArray, Float32, Float32MultiArray
from sensor_msgs.msg import JointState, Joy, Image, CompressedImage
from geometry_msgs.msg import Point, PoseStamped, PoseArray
import tf2_ros

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from il_utils import ros2_configs
from il_utils.transform import TF_mat


# ------------------------------------------------------------------ #
# utility functions
# ------------------------------------------------------------------ #
def extract_data_from_msg(msg):
    if isinstance(msg, JointState):
        data = msg.position
    elif isinstance(msg, (Int32MultiArray, Float32MultiArray)):
        data = msg.data
    elif isinstance(msg, (Int32, Float32)):
        data = [msg.data]
    else:
        print(f"[ERROR] extracting data array from msg type {msg} is not implemented!")
        raise NotImplementedError
    
    return data

def img_msg_to_cv(img_msg) -> np.ndarray:
    """Decode a CompressedImage or Image message to an RGB numpy array."""
    if isinstance(img_msg, CompressedImage):
        buf = np.frombuffer(img_msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(img_msg, Image):
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
        if img_msg.encoding.startswith('bgr'):
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr


class ROS2LeRobotRecord(Node):
    def __init__(self, args):
        super().__init__("ros2_lerobot_record")

        # ------------------------------------------------------------------ #
        # load robot type and config
        # ------------------------------------------------------------------ #
        robot_type = args.robot_type
        lerobot_id = args.lerobot_id
        self.task = args.task
        if self.task is None:
            self.task = lerobot_id
        self.config = ros2_configs.load(robot_type)
        self.fps = float(self.config.fps)

        # ------------------------------------------------------------------ #
        # set features & create lerobot dataset
        # ------------------------------------------------------------------ #
        features = {
            'observation.state': {
                'dtype': "float32",
                'shape': tuple(self.config.state['shape']),
                'names': {
                    "motors": self.config.state['names']
                }
            },
            'action': {
                'dtype': "float32",
                'shape': tuple(self.config.action['shape']),
                'names': {
                    "motors": self.config.action['names']
                }
            }
        }
        self.state_name_to_index = {
            name: k for k, name in enumerate(self.config.state['names'])
        }
        self.action_name_to_index = {
            name: k for k, name in enumerate(self.config.action['names'])
        }

        for ee_name in self.config.end_effector['names']:
            features[f'observation.{ee_name}.pos'] = {
                'dtype': "float32",
                'shape': (3,),
                'names': {
                    "axes": ["x", "y", "z"]
                }
            }
            features[f'observation.{ee_name}.rot6d'] = {
                'dtype': "float32",
                'shape': (6,),
                'names': {
                    "axes": ["R11", "R21", "R31", "R12", "R22", "R32"]
                }
            }
            features[f'action.{ee_name}.pos'] = {
                'dtype': "float32",
                'shape': (3,),
                'names': {
                    "axes": ["x", "y", "z"]
                }
            }
            features[f'action.{ee_name}.rot6d'] = {
                'dtype': "float32",
                'shape': (6,),
                'names': {
                    "axes": ["R11", "R21", "R31", "R12", "R22", "R32"]
                }
            }

        for topic_name, topic_config in self.config.image.items():
            features[f'observation.images.{topic_config['cam_name']}'] = {
                'dtype': "video",
                'shape': topic_config['shape'],
                'names': ["height", "width", "channels"]
            }

        root = args.root
        if root is not None:    # if None, dataset will be saved in $HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot
            root = os.path.join(root, lerobot_id)
        self.dataset = LeRobotDataset.create(
            repo_id=lerobot_id,
            fps = int(self.fps),
            features=features,
            root=root,
            robot_type=robot_type
        )
        self.num_episode = 0

        # ------------------------------------------------------------------ #
        # create subscribers and initialize variables
        # ------------------------------------------------------------------ #
        
        # state
        self.state = np.zeros(self.config.state['shape'])
        for topic_name, topic_config in self.config.state['topics'].items():
            self.create_subscription(
                topic_config['msg_type'], topic_name, 
                lambda msg, names=topic_config['state_names']: self._state_callback(msg, names), 1)
        
        # action
        self.action = np.zeros(self.config.action['shape'])
        self.prev_joy = {}
        for topic_name, topic_config in self.config.action['topics'].items():
            if 'init_value' in topic_config.keys():
                for i, action_name in enumerate(topic_config['action_names']):
                    self.action[self.action_name_to_index[action_name]] = topic_config['init_value'][i]
            if isinstance(topic_config['msg_type'], Joy):
                self.prev_joy[topic_name] = None
                self.create_subscription(
                    Joy, topic_name,
                    lambda msg, name=topic_name, config=topic_config: self._joy_callback(msg, name, config), 1
                )
            else:
                self.create_subscription(
                    topic_config['msg_type'], topic_name, 
                    lambda msg, names=topic_config['action_names']: self._action_callback(msg, names), 1)

        # end-effectors
        self.ee_states = {}
        for topic_name, topic_config in self.config.end_effector['state_topics'].items():
            if topic_name == "/tf":
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
                for i, ee_name in enumerate(topic_config['ee_names']):
                    self.ee_states[ee_name] = {
                        'base_frame': topic_config['base_frames'][i],
                        'ee_frame': topic_config['ee_frames'][i]
                    }
            else:
                self.create_subscription(
                    topic_config['msg_type'], topic_name,
                    lambda msg, names=topic_config['ee_names']: self._ee_state_callback(msg, names), 1
                )
                for ee_name in topic_config['ee_names']:
                    self.ee_states[ee_name] = None

        self.ee_actions = {}
        for topic_name, topic_config in self.config.end_effector['action_topics'].items():
            if topic_name == "/tf":
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
                for i, ee_name in enumerate(topic_config['ee_names']):
                    self.ee_actions[ee_name] = {
                        'base_frame': topic_config['base_frames'][i],
                        'ee_frame': topic_config['ee_frames'][i]
                    }
            else:
                self.create_subscription(
                    topic_config['msg_type'], topic_name,
                    lambda msg, names=topic_config['ee_names']: self._ee_action_callback(msg, names), 1
                )
                for ee_name in topic_config['ee_names']:
                    self.ee_actions[ee_name] = None

        # images
        self.images = {}
        for topic_name, topic_config in self.config.image.items():
            self.images[topic_config['cam_name']] = None
            self.create_subscription(
                topic_config['msg_type'], topic_name,
                lambda msg, name=topic_config['cam_name']: self._img_callback(msg, name), 10
            )

        # recording
        self.recording = False
        self.create_subscription(Bool, '/lerobot/start_recording', self._start_recording_callback, 1)
        self.create_subscription(Bool, '/lerobot/save_recording', self._save_recording_callback, 1)
        self.create_timer(1.0 / self.fps, self._timer_callback)


    # ------------------------------------------------------------------ #
    # callback functions
    # ------------------------------------------------------------------ #    
    def _lookup_tf(self, base_frame, ee_frame):
        try:
            t = self.tf_buffer.lookup_transform(base_frame, ee_frame, rclpy.time.Time())
            return TF_mat.from_msg(t.transform)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({base_frame} -> {ee_frame}): {e}")
            return None

    def _state_callback(self, msg, names):
        data = extract_data_from_msg(msg)
        for i, state_name in enumerate(names):
            self.state[self.state_name_to_index[state_name]] = data[i]

    def _joy_callback(self, msg: Joy, name, config):
        if self.prev_joy[name] is None:
            self.prev_joy[name] = msg
        for i, action_name in enumerate(config['action_names']):
            action_idx = self.action_name_to_index[action_name]
            (field, idx) = config['index'][i]
            if field == "buttons":
                if self.prev_joy[name].buttons[idx] == 0 and msg.buttons[idx] == 1: # button pressed
                    self.action[action_idx] = 1 - self.action[action_idx]           # 0->1, 1->0
            elif field == "axes":
                self.action[action_idx] = msg.axes[idx]
            else:
                self.get_logger().warn(f"wrong field name is given for {action_name} {name} msg config")
        self.prev_joy[name] = msg

    def _action_callback(self, msg, names):
        data = extract_data_from_msg(msg)
        for i, action_name in enumerate(names):
            self.action[self.action_name_to_index[action_name]] = data[i]

    def _ee_state_callback(self, msg, names):
        if isinstance(msg, Point):
            self.ee_states[names[0]] = TF_mat.from_msg(msg)
        elif isinstance(msg, PoseStamped):
            self.ee_states[names[0]] = TF_mat.from_msg(msg.pose)
        elif isinstance(msg, PoseArray):
            for i, ee_name in enumerate(names):
                self.ee_states[ee_name] = TF_mat.from_msg(msg.poses[i])
        else:
            self.get_logger().error(f"TF_mat from msg type {msg} is not implemented")
            raise NotImplementedError

    def _ee_action_callback(self, msg, names):
        if isinstance(msg, Point):
            self.ee_actions[names[0]] = TF_mat.from_msg(msg)
        elif isinstance(msg, PoseStamped):
            self.ee_actions[names[0]] = TF_mat.from_msg(msg.pose)
        elif isinstance(msg, PoseArray):
            for i, ee_name in enumerate(names):
                self.ee_actions[ee_name] = TF_mat.from_msg(msg.poses[i])
        else:
            self.get_logger().error(f"TF_mat from msg type {msg} is not implemented")
            raise NotImplementedError

    def _img_callback(self, msg, name):
        self.images[name] = img_msg_to_cv(msg)

    def _start_recording_callback(self, msg: Bool):
        self.recording = msg.data

    def _save_recording_callback(self, msg: Bool):
        if msg.data:
            self.dataset.save_episode()
            self.num_episode += 1
            self.get_logger().info(f"saved {self.num_episode} episodes")
        else:
            self.dataset.clear_episode_buffer()
            self.get_logger().info("discard current recording")
        
        # reset variables
        self.state = np.zeros(self.config.state['shape'])

        self.action = np.zeros(self.config.action['shape'])
        for _, topic_config in self.config.action['topics'].items():
            if 'init_value' in topic_config.keys():
                for i, action_name in enumerate(topic_config['action_names']):
                    self.action[self.action_name_to_index[action_name]] = topic_config['init_value'][i]

        for joy_name in self.prev_joy.keys():
            self.prev_joy[joy_name] = None
        
        for ee_name in self.ee_states.keys():
            if isinstance(self.ee_states[ee_name], dict):
                continue
            self.ee_states[ee_name] = None

        for ee_name in self.ee_actions.keys():
            if isinstance(self.ee_actions[ee_name], dict):
                continue
            self.ee_actions[ee_name] = None
            
        for cam_name in self.images.keys():
            self.images[cam_name] = None

    def _timer_callback(self):
        if not self.recording:
            return
        
        # state, action
        frame = {
            'observation.state': self.state.astype(np.float32),
            'action': self.action.astype(np.float32)
        }
        # end-effectors
        for ee_name, ee_TF in self.ee_states.items():
            if isinstance(ee_TF, dict):
                ee_TF = self._lookup_tf(ee_TF['base_frame'], ee_TF['ee_frame'])
            if ee_TF is None:
                return
            frame[f'observation.{ee_name}.pos'] = ee_TF.get_pos().astype(np.float32)
            frame[f'observation.{ee_name}.rot6d'] = ee_TF.get_rotm()[:,:2].transpose().reshape(6).astype(np.float32)
        for ee_name, ee_TF in self.ee_actions.items():
            if isinstance(ee_TF, dict):
                ee_TF = self._lookup_tf(ee_TF['base_frame'], ee_TF['ee_frame'])
            if ee_TF is None:
                return
            frame[f'action.{ee_name}.pos'] = ee_TF.get_pos().astype(np.float32)
            frame[f'action.{ee_name}.rot6d'] = ee_TF.get_rotm()[:,:2].transpose().reshape(6).astype(np.float32)
        # images
        for cam_name, img in self.images.items():
            if img is None:
                return
            frame[f'observation.images.{cam_name}'] = img
        frame['task'] = self.task
        self.dataset.add_frame(frame)


def main():
    parser = argparse.ArgumentParser(description="Script for recording lerobot dataset")
    parser.add_argument("--robot-type", type=str, default=True)
    parser.add_argument("--lerobot-id", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    rclpy.init()
    node = ROS2LeRobotRecord(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
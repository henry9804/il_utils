from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from .base_config import BaseConfig


class dualFR3Config(BaseConfig, name="dual_fr3"):
    fps = 30
    
    state = {
        "shape": [16],
        "names": [
            "left_fr3_joint1", "left_fr3_joint2", "left_fr3_joint3", "left_fr3_joint4", "left_fr3_joint5", "left_fr3_joint6", "left_fr3_joint7", 
            "left_fr3_gripper", 
            "right_fr3_joint1", "right_fr3_joint2", "right_fr3_joint3", "right_fr3_joint4", "right_fr3_joint5", "right_fr3_joint6", "right_fr3_joint7", 
            "right_fr3_gripper"
        ],

        "topics": {
            "/jointstates": {
                "msg_type": JointState, # make sure state_names orders match with joint orders
                "state_names": [
                    "left_fr3_joint1", "left_fr3_joint2", "left_fr3_joint3", "left_fr3_joint4", "left_fr3_joint5", "left_fr3_joint6", "left_fr3_joint7", 
                    "left_fr3_gripper", 
                    "right_fr3_joint1", "right_fr3_joint2", "right_fr3_joint3", "right_fr3_joint4", "right_fr3_joint5", "right_fr3_joint6", "right_fr3_joint7", 
                    "right_fr3_gripper"
                ]
            },
        }
    }

    action = {
        "shape": [16],
        "names": [
            "left_fr3_joint1", "left_fr3_joint2", "left_fr3_joint3", "left_fr3_joint4", "left_fr3_joint5", "left_fr3_joint6", "left_fr3_joint7", 
            "left_fr3_gripper", 
            "right_fr3_joint1", "right_fr3_joint2", "right_fr3_joint3", "right_fr3_joint4", "right_fr3_joint5", "right_fr3_joint6", "right_fr3_joint7", 
            "right_fr3_gripper"
        ],

        "topics": {
            "/joint_actions": {
                "msg_type": JointState,
                "action_names": [
                    "left_fr3_joint1", "left_fr3_joint2", "left_fr3_joint3", "left_fr3_joint4", "left_fr3_joint5", "left_fr3_joint6", "left_fr3_joint7", 
                    "right_fr3_joint1", "right_fr3_joint2", "right_fr3_joint3", "right_fr3_joint4", "right_fr3_joint5", "right_fr3_joint6", "right_fr3_joint7"
                ],
            },

            "/left_controller/gripper": {
                "msg_type": Int32,
                "action_names": ["left_fr3_gripper"],
                "init_value": [0],
            },

            "right_controller/gripper": {
                "msg_type": Int32,
                "action_names": ["right_fr3_gripper"],
                "init_value": [0],
            }
        }
    }

    end_effector = {
        "names": ["left_fr3_gripper", "right_fr3_gripper"],

        "state_topics": {
            "/tf": {
                "ee_names": ["left_fr3_gripper", "right_fr3_gripper"],
                "base_frames": ["base", "base"],
                "ee_frames": ["left_fr3_hand", "right_fr3_hand"],
            },
        },
    
        "action_topics": {
            "/left_fr3/target_pose": {
                "msg_type": PoseStamped,
                "ee_names": ["left_fr3_gripper"],
            },

            "/right_fr3/target_pose": {
                "msg_type": PoseStamped,
                "ee_names": ["right_fr3_gripper"],
            },
        }
    }

    image = {
        "/left_realsense/color/image_raw": {
            "msg_type": Image,
            "cam_name": "left",
            "shape": (480, 640, 3),
        },
        "/azure/color/image": {
            "msg_type": Image,
            "cam_name": "base",
            "shape": (480, 640, 3),
        },
        "/right_realsense/color/image_raw": {
            "msg_type": Image,
            "cam_name": "right",
            "shape": (480, 640, 3),
        }
    }

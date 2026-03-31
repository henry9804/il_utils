from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from .base_config import BaseConfig


class PandaManiskillConfig(BaseConfig, name="panda_maniskill"):
    fps = 20
    
    state = {
        "shape": [8],
        "names": [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", 
            "panda_hand"
        ],

        "topics": {
            "/joint_states": {
                "msg_type": JointState, # make sure state_names orders match with joint orders
                "state_names": [
                    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", 
                    "panda_hand"
                ]
            },
        }
    }

    action = {
        "shape": [8],
        "names": [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", 
            "panda_hand"
        ],

        "topics": {
            "/panda/gripper_action": {
                "msg_type": Float32,
                "action_names": ["panda_hand"],
                "init_value": [1.0],
            },
        }
    }

    end_effector = {
        "names": ["panda_hand"],

        "state_topics": {
            "/tf": {
                "ee_names": ["panda_hand"],
                "base_frames": ["panda_link0"],
                "ee_frames": ["panda_hand_tcp"],
            },
        },
    
        "action_topics": {
            "/panda/ee_action": {
                "msg_type": PoseStamped,
                "ee_names": ["panda_hand"],
            },
        }
    }

    image = {
        "/panda/hand_camera": {
            "msg_type": Image,
            "cam_name": "hand",
            "shape": (224, 224, 3),
        },
    }

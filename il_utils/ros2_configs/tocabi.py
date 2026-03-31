from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
from .base_config import BaseConfig


class TOCABIConfig(BaseConfig, name="tocabi"):
    fps = 30
    
    state = {
        "shape": [35],
        "names": [
            "L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
            "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
            "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
            "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
            "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
            "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
            "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint", "L_Hand",
            "Neck_Joint", "Head_Joint",
            "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
            "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint", "R_Hand"
        ],

        "topics": {
            "/tocabi/jointstates": {
                "msg_type": JointState, # make sure state_names orders match with joint orders
                "state_names": [
                    "L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
                    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                    "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
                    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
                    "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint", 
                    "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
                    "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                    "Neck_Joint", "Head_Joint",
                    "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
                    "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint",
                ]
            },

            "/tocabi_hand/left/state": {
                "msg_type": Int32,
                "state_names": ["L_Hand"],
            },

            "/tocabi_hand/right/state": {
                "msg_type": Int32,
                "state_names": ["R_Hand"],
            },
        }
    }
    
    action = {
        "shape": [35],
        "names": [
            "L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
            "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
            "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
            "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
            "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
            "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
            "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint", "L_Hand",
            "Neck_Joint", "Head_Joint",
            "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
            "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint", "R_Hand"
        ],

        "topics": {
            "/tocabi/joint_target": {
                "msg_type": JointState,
                "action_names": [
                    "L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
                    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                    "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
                    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
                    "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint", 
                    "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
                    "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                    "Neck_Joint", "Head_Joint",
                    "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
                    "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"
                ],
            },

            "/tocabi_hand/left/action": {
                "msg_type": Int32,
                "action_names": ["L_Hand"],
                "init_value": [0],
            },

            "/tocabi_hand/right/action": {
                "msg_type": Int32,
                "action_names": ["R_Hand"],
                "init_value": [0],
            },
        }
    }

    end_effector = {
        "names": ["L_Hand", "Head", "R_Hand"],

        "state_topics": {
            # "/tf": {
            #     "ee_names": ["L_Hand", "Head", "R_Hand"],
            #     "base_frames": ["Pelvis_Link", "Pelvis_Link", "Pelvis_Link"],
            #     "ee_frames": ["L_Wrist2_Link", "Head_Link", "R_Wrist2_Link"],
            # },

            "/tocabi/robot_poses": {
                "msg_type": PoseArray,
                "ee_names": ["L_Hand", "Head", "R_Hand"],
            },
        },

        "action_topics": {
            "/tocabi/target_poses": {
                "msg_type": PoseArray,
                "ee_names": ["L_Hand", "Head", "R_Hand"],
            },
        }
    }

    image = {
        "/cam_LEFT/image_raw/compressed": {
            "msg_type": CompressedImage,
            "cam_name": "left",
            "shape": (480, 640, 3),
        },

        "/cam_RIGHT/image_raw/compressed": {
            "msg_type": CompressedImage,
            "cam_name": "right",
            "shape": (480, 640, 3),
        }
    }

import argparse
import h5py
import json
import os

import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _infer_features(demo: h5py.Group) -> tuple[dict, dict]:
    """Return (features_dict, feature_map) inferred from a single demo group."""
    features: dict = {}
    feature_map: dict = {}  # lerobot_key -> hdf5 path relative to demo group

    # actions
    actions = demo["actions"]
    features["action"] = {
        "dtype": "float32",
        "shape": list(actions.shape[1:]),
        "names": None,
    }
    feature_map["action"] = "actions"

    # observations
    for obs_key in demo["obs"].keys():
        sample = demo[f"obs/{obs_key}"][0]
        if obs_key.endswith('image'):
            lerobot_key = f"observation.images.{obs_key}"
            features[lerobot_key] = {
                "dtype": "video",
                "shape": list(sample.shape),
                "names": ["height", "width", "channels"],
            }
        else:
            lerobot_key = f"observation.{obs_key}"
            features[lerobot_key] = {
                "dtype": "float32",
                "shape": list(sample.shape) if sample.ndim > 0 else [1],
                "names": None,
            }
        feature_map[lerobot_key] = f"obs/{obs_key}"

    return features, feature_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert a robomimic HDF5 dataset to a LeRobot dataset"
    )
    parser.add_argument("--hdf5", type=str, required=True, help="Path to robomimic .hdf5 file")
    parser.add_argument("--lerobot-id", type=str, required=True, help="LeRobot repo/dataset id")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--root", type=str, default=None, help="Root directory for the dataset")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--robot-type", type=str, default=None)
    parser.add_argument("--num-episode", type=int, default=None)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        data_group = f["data"]
        demo_keys = sorted(
            [k for k in data_group.keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_")[1]),
        )
        if not demo_keys:
            raise ValueError("No demo groups found under 'data/'")
        if args.num_episode is not None:
            demo_keys = demo_keys[:args.num_episode]

        # infer features from first demo
        features, feature_map = _infer_features(data_group[demo_keys[0]])

        # print env_args if present
        if "env_args" in data_group.attrs:
            env_args = json.loads(data_group.attrs["env_args"])
            print("env_args:", json.dumps(env_args, indent=2))

        # save features for user review
        features_json_path = "features.json"
        with open(features_json_path, "w") as jf:
            json.dump(features, jf, indent=4)
        print(
            f"\nAuto-detected features saved to '{features_json_path}'.\n"
            "Edit if needed, then press Enter to continue..."
        )
        input()

        with open(features_json_path, "r") as jf:
            features_loaded = json.load(jf)
        features = {k: {**v, "shape": tuple(v["shape"])} for k, v in features_loaded.items()}
        os.remove(features_json_path)

        # create dataset
        root = args.root
        if root is not None:
            root = os.path.join(root, args.lerobot_id)
        dataset = LeRobotDataset.create(
            repo_id=args.lerobot_id,
            fps=args.fps,
            features=features,
            root=root,
            robot_type=args.robot_type,
        )

        task = args.task if args.task is not None else args.lerobot_id

        for demo_key in tqdm(demo_keys, desc="demos"):
            demo = data_group[demo_key]
            n = demo["actions"].shape[0]

            for t in range(n):
                frame: dict = {}
                for lerobot_key, hdf5_path in feature_map.items():
                    sample = demo[hdf5_path][t]
                    dtype_str = features[lerobot_key]["dtype"]
                    if dtype_str not in ("video", "image"):
                        sample = sample.astype(dtype_str)
                    frame[lerobot_key] = sample
                frame["task"] = task
                dataset.add_frame(frame)

            dataset.save_episode()

    print("Done. Finalizing dataset...")
    dataset.finalize()


if __name__ == "__main__":
    main()

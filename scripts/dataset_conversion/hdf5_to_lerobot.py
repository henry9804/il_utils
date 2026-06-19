import argparse
import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Script for exporting ACT hdf5 file as lerobot dataset")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--lerobot-id", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    hdf5_list = [fname for fname in os.listdir(dataset_dir) if fname.endswith('hdf5')]


    # load one frame to set features
    features_tmp = {}
    feature_map = {}
    with h5py.File(os.path.join(dataset_dir, hdf5_list[0]), 'r') as data:
        for key in data.keys():
            if key == 'observations':
                continue
            features_tmp[key] = {
                "dtype": str(data[key][0].dtype),
                "shape": data[key][0].shape,
                "names": None
            }
            feature_map[key] = key
        for key in data['observations'].keys():
            if key == 'images':
                continue
            features_tmp[f'observation.{key}'] = {
                "dtype": str(data[f'observations/{key}'][0].dtype),
                "shape": data[f'observations/{key}'][0].shape,
                "names": None
            }
            feature_map[f'observation.{key}'] = f'observations/{key}'
        for key in data['observations/images'].keys():
            features_tmp[f'observation.images.{key}'] = {
                "dtype": "video",
                "shape": data[f'observations/images/{key}'][0].shape,
                "names": ["height", "width", "channels"]
            }
            feature_map[f'observation.images.{key}'] = f'observations/images/{key}'

    # Save features_tmp to JSON for user review/editing
    features_json_path = "features.json"
    with open(features_json_path, "w") as f:
        json.dump(features_tmp, f, indent=4)
    print(f"features_tmp saved to '{features_json_path}'. Edit it, then press Enter to continue...")
    input()

    with open(features_json_path, "r") as f:
        features_loaded = json.load(f)
    features = {k: {**v, "shape": tuple(v["shape"])} for k, v in features_loaded.items()}
    os.remove(features_json_path)


    # create lerobot dataset
    lerobot_id = args.lerobot_id
    root = args.root
    if root is not None:    # if None, dataset will be saved in $HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot
        root = os.path.join(root, lerobot_id)
    lerobot_dataset = LeRobotDataset.create(
        repo_id=lerobot_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type=args.robot_type
    )


    # save frames in hdf5 for each episode
    for fname in tqdm(hdf5_list):
        with h5py.File(os.path.join(dataset_dir, fname), 'r') as data:
            episode_len = data['action'].shape[0]
            for t_ in range(episode_len):
                frame = {}
                for lerobot_key, hdf5_key in feature_map.items():
                    feature_data = data[hdf5_key][t_]
                    dtype = features[lerobot_key]['dtype']
                    if dtype not in ["video", "image"]:
                        feature_data = feature_data.astype(dtype)
                    frame[lerobot_key] = feature_data
                frame['task'] = args.task if args.task is not None else args.lerobot_id
                lerobot_dataset.add_frame(frame)
        lerobot_dataset.save_episode()


if __name__ == "__main__":
    main()
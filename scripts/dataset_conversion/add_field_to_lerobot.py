"""
Add a new data field in-place to an existing LeRobot dataset.

The script directly modifies the parquet files under data/ and updates
meta/info.json — no new dataset is created.

The new field value is computed by a user-defined function
`compute_new_field(row, episode_index, frame_index)` that you edit below.

Usage
-----
python add_field_to_lerobot.py \
    --repo-id  <repo_id of the dataset> \
    [--root    <local root containing the dataset>]
"""

import os
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ─── Define your new field here ───────────────────────────────────────────────
'''
NEW_FIELD_NAME  = ["observation.state.task", "action.task"]
NEW_FIELD_DTYPE = ["float32", "float32"]    # numpy dtype string  (not "image" or "video")
NEW_FIELD_SHAPE = [     (4,),      (4,)]    # tuple — must match what compute_new_field returns

def compute_new_field(row: dict) -> list:
    """
    Return the value of the new field for one frame.

    Parameters
    ----------
    row : dict
        All existing scalar/vector fields for this timestep as numpy arrays.
        (image/video fields are NOT included — they live in separate mp4 files)

    Returns
    -------
    list  (list of np.ndarray, each shape must match NEW_FIELD_SHAPE, dtype NEW_FIELD_DTYPE)

    Example: scalar L2 norm of the action vector.
    """
    ee_state_pos = row["observation.panda_hand.pos"]
    gripper_state = row["observation.state"][-1:]
    ee_action_pos = row["action.panda_hand.pos"]
    gripper_action = row["action"][-1:]

    new_state = np.concatenate([ee_state_pos, gripper_state]).astype(NEW_FIELD_DTYPE[0])
    new_action = np.concatenate([ee_action_pos, gripper_action]).astype(NEW_FIELD_DTYPE[1])

    return [new_state, new_action]

NEW_FIELD_NAME  = ["observation.state"]
NEW_FIELD_DTYPE = ["float32"]
NEW_FIELD_SHAPE = [    (10,)]

from il_utils.transform import TF_mat

def compute_new_field(row: dict) -> list:
    """
    Return the value of the new field for one frame.

    Parameters
    ----------
    row : dict
        All existing scalar/vector fields for this timestep as numpy arrays.
        (image/video fields are NOT included — they live in separate mp4 files)

    Returns
    -------
    list  (list of np.ndarray, each shape must match NEW_FIELD_SHAPE, dtype NEW_FIELD_DTYPE)

    Example: scalar L2 norm of the action vector.
    """
    ee_state_pos = row["observation.robot0_eef_pos"]
    ee_state_quat = row["observation.robot0_eef_quat"]
    gripper_state = row["observation.robot0_gripper_qpos"][0:1]

    ee_state_mat = TF_mat.from_vectors(ee_state_pos, ee_state_quat)
    ee_state_rot6d = ee_state_mat.get_rot6d()
    new_state = np.concatenate([ee_state_pos, ee_state_rot6d, gripper_state]).astype(NEW_FIELD_DTYPE[0])

    return [new_state]
'''

NEW_FIELD_NAME  = ["action.absolute"]
NEW_FIELD_DTYPE = ["float32"]
NEW_FIELD_SHAPE = [    (10,)]

def compute_new_field(rows: dict, n_rows: int) -> list:
    """
    Return the value of the new field for one frame.

    Parameters
    ----------
    rows : dict
        All existing scalar/vector fields for entire dataset as numpy arrays.
        (image/video fields are NOT included — they live in separate mp4 files)
    n_rows: int
        number of rows

    Returns
    -------
    list  (list of new field values in np.ndarray. Each shape must match NEW_FIELD_SHAPE, dtype NEW_FIELD_DTYPE)

    Example: scalar L2 norm of the action vector.
    """
    abs_action = []
    for i in range(n_rows-1):
        if rows["frame_index"][i+1] == 0:
            abs_action.append(np.concatenate([rows["observation.state"][i,:-1], rows["action"][i,-1:]]).astype(NEW_FIELD_DTYPE[0]))
        else:
            abs_action.append(np.concatenate([rows["observation.state"][i+1,:-1], rows["action"][i,-1:]]).astype(NEW_FIELD_DTYPE[0]))
    abs_action.append(abs_action[-1])

    return [np.array(abs_action)]

# ──────────────────────────────────────────────────────────────────────────────


INFO_PATH = "meta/info.json"


def load_info(root: Path) -> dict:
    with open(root / INFO_PATH) as f:
        return json.load(f)


def save_info(info: dict, root: Path) -> None:
    with open(root / INFO_PATH, "w") as f:
        json.dump(info, f, indent=4)


def process_parquet(path: Path, field_names: list) -> tuple[dict, np.ndarray]:
    """Read one parquet file, compute new columns, write it back.

    Returns
    -------
    values : dict[field_name -> np.ndarray (N, *shape)]
        Computed values for every new field — used for global stats.
    episode_indices : np.ndarray (N,)
        episode_index of each row — used to group values for per-episode stats.
    """
    table = pq.read_table(path)
    columns = {name: table.column(name) for name in table.schema.names}

    n_rows = table.num_rows
    new_field_values = {name: [] for name in field_names}

    rows = {}
    for col_name, col_data in columns.items():
        rows[col_name] = []
        for i in range(n_rows):
            val = col_data[i].as_py()
            if isinstance(val, list):
                val = np.array(val, dtype=np.float32)
            rows[col_name].append(val)
        rows[col_name] = np.array(rows[col_name])

    new_values = compute_new_field(rows, n_rows)
    for name, value in zip(field_names, new_values):
        new_field_values[name] = value

    '''
    for i in range(n_rows):
        row = {}
        for col_name, col_data in columns.items():
            val = col_data[i].as_py()
            if isinstance(val, list):
                val = np.array(val, dtype=np.float32)
            row[col_name] = val

        new_values = compute_new_field(row)
        for name, value in zip(field_names, new_values):
            new_field_values[name].append(value)
    '''

    # Append each new column to the table (carry updated table through the loop)
    new_table = table
    for name in field_names:
        new_column = new_field_values[name]  # (N, *shape)
        # new_column = np.stack(new_field_values[name])  # (N, *shape)
        if new_column.ndim == 1:
            pa_col = pa.array(new_column.tolist(), type=pa.float32())
        else:
            pa_col = pa.array(new_column.tolist())
        new_table = new_table.append_column(pa.field(name, pa_col.type), pa_col)

    pq.write_table(new_table, path)

    episode_indices = np.array(columns["episode_index"].to_pylist(), dtype=np.int64)
    return {name: np.stack(new_field_values[name]) for name in field_names}, episode_indices


def update_episodes_parquets(root: Path, field_names: list,
                              per_episode: dict[int, dict[str, np.ndarray]]) -> None:
    """Append per-episode stats columns to every meta/episodes parquet file.

    For each new field, adds columns named ``stats/<field>/<stat>`` (one row per
    episode), mirroring the format already used for existing fields.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    field_names : list[str]
        Names of the newly added fields.
    per_episode : dict[episode_index -> dict[field_name -> np.ndarray (T, *shape)]]
        All frame values for each episode, keyed by episode index.
    """
    stat_keys = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        print("  [WARN] No episode parquet files found — skipping episode stats update.")
        return

    for ep_path in tqdm(episode_files, desc="Episode parquet files"):
        shutil.copy2(ep_path, str(ep_path) + ".bak")
        table = pq.read_table(ep_path)
        ep_indices = table.column("episode_index").to_pylist()

        new_table = table
        for field_name in field_names:
            # Compute per-episode stats for every stat key
            ep_stat_cols: dict[str, list] = {k: [] for k in stat_keys}

            for ep_idx in ep_indices:
                vals = per_episode.get(ep_idx, {}).get(field_name)
                if vals is None or len(vals) == 0:
                    raise KeyError(f"No values found for episode {ep_idx}, field '{field_name}'")
                s = compute_stats(vals)
                for k in stat_keys:
                    ep_stat_cols[k].append(s[k])

            # Append one column per stat key
            for stat_key in stat_keys:
                col_name = f"stats/{field_name}/{stat_key}"
                pa_col = pa.array(ep_stat_cols[stat_key])
                new_table = new_table.append_column(pa.field(col_name, pa_col.type), pa_col)

        pq.write_table(new_table, ep_path)


def compute_stats(all_values: np.ndarray) -> dict:
    """Compute LeRobot-style stats for a (N, D) or (N,) array.

    Returns a dict with keys: min, max, mean, std, count, q01, q10, q50, q90, q99.
    Each value is a Python list (one element per feature dimension).
    """
    if all_values.ndim == 1:
        all_values = all_values[:, None]   # (N, 1)

    count = all_values.shape[0]
    return {
        "min":   all_values.min(axis=0).tolist(),
        "max":   all_values.max(axis=0).tolist(),
        "mean":  all_values.mean(axis=0).tolist(),
        "std":   all_values.std(axis=0).tolist(),
        "count": [count],
        "q01":   np.percentile(all_values,  1, axis=0).tolist(),
        "q10":   np.percentile(all_values, 10, axis=0).tolist(),
        "q50":   np.percentile(all_values, 50, axis=0).tolist(),
        "q90":   np.percentile(all_values, 90, axis=0).tolist(),
        "q99":   np.percentile(all_values, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add a new data field in-place to a LeRobot dataset."
    )
    parser.add_argument("--repo-id", required=True, help="repo_id of the dataset")
    parser.add_argument("--root",    default=None,  help="local root containing the dataset")
    args = parser.parse_args()

    # ── Resolve dataset root ───────────────────────────────────────────────────
    if args.root is not None:
        root = Path(args.root) / args.repo_id
    else:
        home = os.path.expanduser("~")
        hf_home = Path(os.environ.get("HF_LEROBOT_HOME", f"{home}/.cache/huggingface/lerobot"))
        root = hf_home / args.repo_id

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # ── Load and check info.json ───────────────────────────────────────────────
    info = load_info(root)
    features = info.get("features", {})

    for name in NEW_FIELD_NAME:
        if name in features:
            raise ValueError(f"Field '{name}' already exists in the dataset.")

    print(f"Dataset : {root}")
    print(f"Features: {list(features.keys())}")
    for name, dtype, shape in zip(NEW_FIELD_NAME, NEW_FIELD_DTYPE, NEW_FIELD_SHAPE):
        print(f"Adding  : '{name}'  dtype={dtype}  shape={shape}")

    # ── Find all data parquet files ────────────────────────────────────────────
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    print(f"\nParquet files to update: {len(parquet_files)}")

    # ── Back up info.json and stats.json before modifying anything ───────────
    info_path = root / INFO_PATH
    shutil.copy2(info_path, str(info_path) + ".bak")

    stats_path = root / "meta/stats.json"
    if stats_path.exists():
        shutil.copy2(stats_path, str(stats_path) + ".bak")

    # ── Process each parquet file, accumulate global + per-episode values ────
    accumulated  = {name: [] for name in NEW_FIELD_NAME}  # global: list of (N, *shape)
    per_episode: dict[int, dict[str, list]] = {}           # ep_idx -> field -> list of rows

    for pq_path in tqdm(parquet_files, desc="Parquet files"):
        chunk_values, ep_indices = process_parquet(pq_path, NEW_FIELD_NAME)
        for name, arr in chunk_values.items():
            accumulated[name].append(arr)
            for i, ep_idx in enumerate(ep_indices):
                ep_idx = int(ep_idx)
                per_episode.setdefault(ep_idx, {name: [] for name in NEW_FIELD_NAME})
                per_episode[ep_idx].setdefault(name, [])
                per_episode[ep_idx][name].append(arr[i])

    # Stack per-episode lists into arrays
    for ep_idx in per_episode:
        for name in NEW_FIELD_NAME:
            per_episode[ep_idx][name] = np.stack(per_episode[ep_idx][name])

    # ── Update info.json with the new features ────────────────────────────────
    for name, dtype, shape in zip(NEW_FIELD_NAME, NEW_FIELD_DTYPE, NEW_FIELD_SHAPE):
        features[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "names": None,
        }
    info["features"] = features
    save_info(info, root)

    # ── Compute and write stats ───────────────────────────────────────────────
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}

    for name in NEW_FIELD_NAME:
        all_values = np.concatenate(accumulated[name], axis=0)  # (N_total, *shape)
        stats[name] = compute_stats(all_values)
        print(f"  {name}: mean={stats[name]['mean']}, std={stats[name]['std']}")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    # ── Update meta/episodes parquet files with per-episode stats ─────────────
    print("\nUpdating episode parquet files...")
    update_episodes_parquets(root, NEW_FIELD_NAME, per_episode)

    print(f"\nDone. '{NEW_FIELD_NAME}' added in-place to {root}")
    print(f"(backups saved as *.bak)")


if __name__ == "__main__":
    main()

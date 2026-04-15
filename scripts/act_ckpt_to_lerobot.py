"""
Convert an original ACT checkpoint (policy_val_best.ckpt + dataset_stats.pkl)
to the lerobot safetensors format.

Outputs
-------
<output_dir>/
    config.json            - ACTConfig (lerobot format)
    model.safetensors      - remapped model weights
    preprocessor.json      - NormalizerProcessorStep pipeline config
    preprocessor_step_0_normalizer_processor.safetensors  - input norm stats
    postprocessor.json     - UnnormalizerProcessorStep pipeline config
    postprocessor_step_0_normalizer_processor.safetensors - output unnorm stats

Usage
-----
python act_ckpt_to_lerobot.py

Adjust the constants at the top of the file as needed.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

# ─── Configuration ────────────────────────────────────────────────────────────

CKPT_PATH   = "/media/lyh/SSD2TB/act/ckpt/real_tocabi_pick_n_place/ee_rel_small"
OUTPUT_DIR  = "/home/lyh/Data/ckpt/lerobot/real_tocabi_pick_n_place/act_ee_rel_small"
KEY_MAP_FILE = "act.json"

# ImageNet mean/std (used when VISUAL features have MEAN_STD normalization).
# Original ACT does not normalise images in the policy; these are applied by
# the backbone.  We register them so lerobot's preprocessor is consistent.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

POLICY_CONFIG = ACTConfig(
    chunk_size=30,
    n_obs_steps=1,
    n_action_steps=30,
    device="cuda",

    input_features={
        "observation.images.stereo": PolicyFeature(FeatureType.VISUAL, [3, 480, 1280]),
        "observation.state":         PolicyFeature(FeatureType.STATE,  [10]),
    },
    output_features={
        "action": PolicyFeature(FeatureType.ACTION, [10]),
    },

    normalization_mapping={
        "VISUAL": "MEAN_STD",
        "STATE":  "MEAN_STD",
        "ACTION": "MEAN_STD",
    },

    dim_model=512,
    dim_feedforward=3200,
    n_heads=8,
    n_encoder_layers=4,
    n_decoder_layers=7,
    latent_dim=32,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def remap_state_dict(ckpt_state_dict: dict, key_map: dict) -> dict:
    """Apply key_map to rename ACT keys → lerobot keys."""
    mapped = {}
    unmapped = []
    for act_key, lerobot_key in key_map.items():
        if act_key in ckpt_state_dict:
            mapped[lerobot_key] = ckpt_state_dict[act_key]
        else:
            unmapped.append(act_key)
    if unmapped:
        print(f"[WARN] {len(unmapped)} keys in act.json were not found in the checkpoint:")
        for k in unmapped[:10]:
            print(f"       {k}")
        if len(unmapped) > 10:
            print(f"       ... and {len(unmapped) - 10} more")
    return mapped


def convert_stats(raw_stats: dict, config: ACTConfig) -> dict:
    """
    Convert original ACT dataset_stats.pkl to lerobot normalization stats format.

    Original ACT format (typical):
        {
            'qpos_mean':   ndarray,
            'qpos_std':    ndarray
            'action_mean': ndarray, 
            'action_std':  ndarray,
            ...
        }

    Lerobot format expected by NormalizerProcessorStep.state_dict():
        {
            'observation.state.mean': Tensor,
            'observation.state.std':  Tensor,
            'action.mean':            Tensor,
            'action.std':             Tensor,
            'observation.images.stereo.mean': Tensor,   # ImageNet values
            'observation.images.stereo.std':  Tensor,
            ...
        }
    """
    lerobot_stats: dict[str, dict[str, torch.Tensor]] = {}

    # Map state / action stats from pkl
    lerobot_stats["observation.state"] = {
        "mean": torch.from_numpy(raw_stats["qpos_mean"][3:].astype(np.float32)),
        "std":  torch.from_numpy(raw_stats["qpos_std"][3:].astype(np.float32))
    }
    lerobot_stats["action"] = {
        "mean": torch.from_numpy(raw_stats["action_mean"][3:].astype(np.float32)),
        "std":  torch.from_numpy(raw_stats["action_std"][3:].astype(np.float32))
    }

    # Add ImageNet stats for every VISUAL input feature
    for feat_key, feat in config.input_features.items():
        if feat.type == FeatureType.VISUAL:
            lerobot_stats[feat_key] = {
                "mean": torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(3, 1, 1),
                "std":  torch.tensor(IMAGENET_STD,  dtype=torch.float32).reshape(3, 1, 1),
            }

    return lerobot_stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load original checkpoint
    ckpt_file = os.path.join(CKPT_PATH, "policy_val_best.ckpt")
    print(f"Loading ACT checkpoint: {ckpt_file}")
    raw_state_dict = torch.load(ckpt_file, map_location="cpu")

    # 2. Load key map and remap keys
    with open(KEY_MAP_FILE) as f:
        key_map = json.load(f)
    print(f"Remapping {len(key_map)} keys via {KEY_MAP_FILE} ...")
    mapped_state_dict = remap_state_dict(raw_state_dict, key_map)
    print(f"  {len(mapped_state_dict)} keys mapped successfully")

    # 3. Build policy and load weights
    print("Building ACTPolicy and loading weights ...")
    policy = ACTPolicy(POLICY_CONFIG)
    missing, unexpected = policy.load_state_dict(mapped_state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print("  Weights loaded.")

    # 4. Save policy (config.json + model.safetensors)
    print(f"Saving policy to: {output_dir}")
    policy.save_pretrained(output_dir)
    print("  Saved config.json + model.safetensors")

    # 5. Load and convert normalization stats
    stats_file = os.path.join(CKPT_PATH, "dataset_stats.pkl")
    print(f"Loading dataset stats: {stats_file}")
    with open(stats_file, "rb") as f:
        raw_stats = pickle.load(f)
    print(f"  Raw stats keys: {list(raw_stats.keys())}")

    lerobot_stats = convert_stats(raw_stats, POLICY_CONFIG)
    print(f"  Converted stats keys: {list(lerobot_stats.keys())}")

    # 6. Save pre/postprocessors (normalises inputs and outputs of policy)
    print("Saving pre/postprocessors ...")
    preprocessor, postprocessor = make_act_pre_post_processors(POLICY_CONFIG, lerobot_stats)
    preprocessor.save_pretrained(output_dir, config_filename='policy_preprocessor.json')
    postprocessor.save_pretrained(output_dir, config_filename='policy_postprocessor.json')

    print(f"\nConversion complete. Output directory: {output_dir}")
    print("Files:")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

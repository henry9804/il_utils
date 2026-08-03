#!/usr/bin/env python

"""
Flexible LeRobot training script.

Supports multiple policy types (act, diffusion, pi0, smolvla, ...) with
user-configurable delta_timestamps and input/output feature key names.

Key constraints (lerobot internals):
  - OBS_STATE = "observation.state"   policies look for this key in the batch
  - ACTION    = "action"              policies look for this key in the batch
  - image keys must start with "observation." (batch-to-transition converter)

Custom state/action keys are remapped:
  - Observation keys (state, images): RenameObservationsProcessorStep (built into
    every policy preprocessor) is configured with the rename map after creation.
  - Action key: renamed manually in the batch before each preprocessor call.
  - Dataset stats are also renamed so the normalizer finds the right entries.

Example usage:
    # ACT with defaults (delta_timestamps inferred from policy)
    python lerobot_train.py \
        --dataset-id user/my_dataset \
        --policy act \
        --state-key observation.state \
        --image-keys top wrist \
        --action-key action \
        --output-dir outputs/train/my_run \
        --steps 100000 \
        --batch-size 64

    # Custom dataset keys (internally remapped to standard names)
    python lerobot_train.py \
        --dataset-id user/my_dataset \
        --policy act \
        --state-key observation.robot_arm_state \
        --image-keys cam_top cam_wrist \
        --action-key my_action \
        --output-dir outputs/train/my_run

    # Custom delta_indices (overrides policy defaults, converted to seconds via fps):
    python lerobot_train.py \
        --dataset-id user/my_dataset \
        --policy diffusion \
        --state-key observation.state \
        --image-keys observation.images.top \
        --policy-kwargs '{"n_obs_steps": 2, "horizon": 16}' \
        --obs-delta-indices -1 0 \
        --action-delta-indices -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
"""

import argparse
import logging
import time
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms.v2 as T

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.utils.utils import cycle
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.processor.rename_processor import RenameObservationsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.logging_utils import AverageMeter


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Feature helpers
# ------------------------------------------------------------------ #

def build_input_features(
    ds_meta: LeRobotDatasetMetadata,
    state_key: str | None,
    image_keys: list[str],
) -> dict[str, PolicyFeature]:
    """
    Build input_features for the policy config, deriving shapes from ds_meta.

    State is registered under OBS_STATE so every policy's robot_state_feature
    property finds it.  Image shape is taken from the dataset (converted to
    C,H,W via dataset_to_policy_features).  Image keys must start with
    "observation." for the batch-to-transition converter.
    output_features is left to make_policy to infer from ds_meta.
    """
    policy_features = dataset_to_policy_features(ds_meta.features)
    input_features: dict[str, PolicyFeature] = {}

    if state_key is not None:
        if not state_key.startswith("observation."):
            raise ValueError(
                f"State key '{state_key}' must start with 'observation.' "
                "(required by lerobot's batch-to-transition converter)."
            )
        state_shape = ds_meta.shapes[state_key]
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=state_shape)

    for key in image_keys:
        if not key.startswith("observation."):
            raise ValueError(
                f"Image key '{key}' must start with 'observation.' "
                "(required by lerobot's batch-to-transition converter)."
            )
        image_shape = policy_features[key].shape  # already C,H,W
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)

    return input_features


def rename_stats(stats: dict, rename_map: dict[str, str]) -> dict:
    """Return a copy of stats with top-level keys renamed according to rename_map."""
    return {rename_map.get(k, k): v for k, v in stats.items()}


def _update_last_symlink(output_dir: Path, step_name: str) -> None:
    """Point checkpoints/last -> step_name (relative symlink, replaces any existing)."""
    link = output_dir / "checkpoints" / "last"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(step_name)


# ------------------------------------------------------------------ #
# delta_timestamps helper
# ------------------------------------------------------------------ #

def build_delta_timestamps(
    policy_cfg,
    fps: float,
    state_key: str | None,
    image_keys: list[str],
    action_key: str,
    obs_delta_indices: list[int] | None,
    action_delta_indices: list[int] | None,
) -> dict[str, list[float]]:
    """
    Build delta_timestamps dict for LeRobotDataset.

    Keys here are the DATASET column names (user-provided), not the
    standardized policy names, because this is what LeRobotDataset uses
    to index into its parquet files.

    If obs/action delta_indices are provided by the user they are converted
    to seconds via fps; otherwise the policy config's own indices are used.
    """
    if obs_delta_indices is not None:
        obs_ts = [i / fps for i in obs_delta_indices]
    else:
        obs_indices = policy_cfg.observation_delta_indices
        obs_ts = [0.0] if obs_indices is None else [i / fps for i in obs_indices]

    if action_delta_indices is not None:
        act_ts = [i / fps for i in action_delta_indices]
    else:
        act_indices = policy_cfg.action_delta_indices
        act_ts = [0.0] if act_indices is None else [i / fps for i in act_indices]

    delta_timestamps: dict[str, list[float]] = {}
    if state_key is not None:
        delta_timestamps[state_key] = obs_ts
    for key in image_keys:
        delta_timestamps[key] = obs_ts
    delta_timestamps[action_key] = act_ts

    return delta_timestamps


# ------------------------------------------------------------------ #
# Training loop
# ------------------------------------------------------------------ #

def train(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_key: str | None = args.state_key if args.state_key else None
    image_keys = [f"observation.images.{cam}" for cam in args.image_keys]
    action_key: str | None = args.action_key if args.action_key else None

    # ---- dataset metadata (needed for shapes before building features) ----
    log.info(f"Loading dataset metadata: {args.dataset_id}")
    ds_meta = LeRobotDatasetMetadata(args.dataset_id)

    # ---- rename maps ----
    # obs_rename_map: fed to RenameObservationsProcessorStep inside the preprocessor.
    obs_rename_map = {state_key: OBS_STATE} if (state_key and state_key != OBS_STATE) else {}
    # action_rename_map: applied manually before each preprocessor call 
    # (action is not an observation key so RenameObservationsProcessorStep won't touch it).
    if action_key and action_key != ACTION:
        action_rename_map = {action_key: ACTION}
        ds_meta.features[ACTION] = ds_meta.features[action_key]
    else:
        action_rename_map = {}
    stats_rename_map = {**obs_rename_map, **action_rename_map}
    if stats_rename_map:
        log.info(f"Key renames: {stats_rename_map}")

    # ---- input features (shapes derived from ds_meta; pre-set so make_policy keeps them) ----
    input_features = build_input_features(
        ds_meta=ds_meta,
        state_key=state_key,
        image_keys=image_keys,
    )

    # ---- image size: resize-to-fit + center crop ----
    resize_crop_tf = T.Identity()
    if args.img_size is not None and image_keys:
        tgt_h, tgt_w = args.img_size
        orig_c, orig_h, orig_w = input_features[image_keys[0]].shape
        scale = min(orig_h / tgt_h, orig_w / tgt_w)
        crop_h, crop_w = round(tgt_h * scale), round(tgt_w * scale)
        log.info(f"Image crop: ({orig_h},{orig_w}) -> ({crop_h},{crop_w}) -> resize ({tgt_h},{tgt_w})")
        resize_crop_tf = T.Compose([
            T.CenterCrop((crop_h, crop_w)),
            T.Resize((tgt_h, tgt_w), antialias=True),
        ])
        for key in image_keys:
            c = input_features[key].shape[0]
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, tgt_h, tgt_w))

    log.info(f"Input features: { {k: v.shape for k, v in input_features.items()} }")

    # ---- policy config ----
    policy_kwargs: dict = dict(input_features=input_features)
    if args.policy_kwargs:
        import json
        policy_kwargs.update(json.loads(args.policy_kwargs))
    if args.obs_delta_indices is not None:
        policy_kwargs['horizon'] = len(args.obs_delta_indices)
    if args.action_delta_indices is not None:
        policy_kwargs['n_obs_steps'] = len(args.action_delta_indices)

    policy_cfg = make_policy_config(args.policy, **policy_kwargs)
    policy_cfg.device = args.device
    policy_cfg.repo_id = args.dataset_id

    delta_timestamps = build_delta_timestamps(
        policy_cfg=policy_cfg,
        fps=ds_meta.fps,
        state_key=state_key,
        image_keys=image_keys,
        action_key=args.action_key,
        obs_delta_indices=args.obs_delta_indices,
        action_delta_indices=args.action_delta_indices,
    )
    log.info(f"delta_timestamps: {delta_timestamps}")

    image_transforms = ImageTransforms(ImageTransformsConfig(enable=args.image_transforms))
    log.info(f"Image transforms: {'enabled' if args.image_transforms else 'disabled'}")

    if args.num_episodes is not None:
        episodes = np.arange(args.num_episodes)
    else:
        episodes = None

    log.info("Loading dataset...")
    dataset = LeRobotDataset(
        args.dataset_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=T.Compose([resize_crop_tf, image_transforms]),
    )
    log.info(f"  frames: {dataset.num_frames}  episodes: {dataset.num_episodes}")

    # ---- policy (make_policy infers output_features from ds_meta) ----
    log.info(f"Creating policy: {args.policy}")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta, rename_map=obs_rename_map or None)
    policy.train()

    # ---- pre/post processors ----
    # Stats renamed so the normalizer (after RenameObservationsProcessorStep) finds
    # entries under the standardized key names.
    stats = rename_stats(ds_meta.stats, stats_rename_map)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=stats)

    # Wire obs rename_map into the existing RenameObservationsProcessorStep.
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = obs_rename_map
            break

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_cosine:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # ---- dataloader ----
    device = torch.device(args.device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # ---- training loop ----
    loss_meter = AverageMeter("loss", ":.4f")
    step_time_meter = AverageMeter("step time", ":.3f")
    log.info(f"Starting training for {args.steps} steps")

    for step in range(args.steps):
        t0 = time.perf_counter()

        batch = next(dl_iter)

        # Rename action key (and its pad mask) — obs keys handled by RenameObservationsProcessorStep
        if action_key and action_key != ACTION:
            if action_key in batch:
                batch[ACTION] = batch.pop(action_key)
            else:
                log.error(f"{action_key} is not in dataset!")
                return
            if f"{action_key}_is_pad" in batch:
                batch[f"{ACTION}_is_pad"] = batch.pop(f"{action_key}_is_pad")

        batch = preprocessor(batch)

        loss, _ = policy.forward(batch)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        step_time = time.perf_counter() - t0

        loss_meter.update(loss.item())
        step_time_meter.update(step_time)

        if step % args.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            log.info(
                f"step {step}/{args.steps}  loss={loss_meter.avg:.4f}  lr={lr:.2e}  time={step_time_meter.avg:.3f}s"
            )
            loss_meter.reset()
            step_time_meter.reset()

        if args.save_freq > 0 and step % args.save_freq == 0:
            ckpt_dir = output_dir / f"checkpoints/{step:07d}/pretrained_model"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            _update_last_symlink(output_dir, f"{step:07d}")
            log.info(f"Checkpoint saved to {ckpt_dir}")

    # final checkpoint
    ckpt_dir = output_dir / f"checkpoints/{args.steps:07d}/pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    _update_last_symlink(output_dir, f"{args.steps:07d}")
    log.info(f"Training complete. Final model saved to {ckpt_dir}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flexible LeRobot training script supporting multiple policy types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument("--dataset-id", required=True,
                   help="LeRobot dataset repo-id (e.g. 'user/my_dataset')")
    p.add_argument("--num-episodes", type=int, default=None,
                   help="Number of episodes used for training")

    # Policy
    p.add_argument("--policy", required=True,
                   help="Policy type to train (e.g. act, diffusion, pi0, smolvla, vqbet, tdmpc)")
    p.add_argument("--policy-kwargs", default=None,
                   help="JSON string of extra policy config kwargs, "
                        "e.g. '{\"chunk_size\": 50, \"n_obs_steps\": 1}'")

    # Feature keys (dataset column names)
    p.add_argument("--state-key", default="observation.state",
                   help="Dataset column for robot state. Pass empty string to disable state input.")
    p.add_argument("--image-keys", nargs="*", default=[],
                   help="Dataset columns for image observations (must start with 'observation.')")
    p.add_argument("--action-key", default="action",
                   help="Dataset column for actions")

    # Image transforms
    p.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="Resize images to fit HxW (maintains aspect ratio) then center-crop. "
                        "Applied per-batch before the preprocessor. "
                        "Also updates input_features so the policy sees the correct shape.")
    p.add_argument("--image-transforms", action="store_true",
                   help="Enable default image augmentations (brightness, contrast, saturation, "
                        "hue, sharpness, affine) via lerobot's ImageTransforms")

    # delta_indices (optional overrides)
    p.add_argument("--obs-delta-indices", type=int, nargs="+", default=None,
                   help="Observation delta indices (converted to seconds via fps, "
                        "overrides policy defaults). E.g. --obs-delta-indices -1 0")
    p.add_argument("--action-delta-indices", type=int, nargs="+", default=None,
                   help="Action delta indices (converted to seconds via fps, "
                        "overrides policy defaults). E.g. --action-delta-indices 0 1 2 3 4")

    # Training
    p.add_argument("--steps", type=int, default=50000,
                   help="Total number of gradient update steps")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=10.0,
                   help="Gradient clipping norm (0 = disabled)")
    p.add_argument("--lr-cosine", action="store_true",
                   help="Use cosine annealing LR scheduler")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers")

    # Logging / checkpointing
    p.add_argument("--output-dir", default="outputs/train/run",
                   help="Directory for checkpoints and final model")
    p.add_argument("--log-freq", type=int, default=100,
                   help="Log every N steps")
    p.add_argument("--save-freq", type=int, default=5000,
                   help="Save checkpoint every N steps (0 = only at end)")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to train on")

    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"Policy:    {args.policy}")
    log.info(f"Dataset:   {args.dataset_id}")
    log.info(f"Device:    {args.device}")
    log.info(f"Steps:     {args.steps}  BatchSize: {args.batch_size}  LR: {args.lr}")
    time_start = time.perf_counter()
    train(args)
    print(f"Training finished: {time.perf_counter() - time_start}s")


if __name__ == "__main__":
    main()

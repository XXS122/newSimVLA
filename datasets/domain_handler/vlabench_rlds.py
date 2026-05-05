from __future__ import annotations

import io
import os
import random
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image

from .base import DomainHandler


class VLABenchRLDSHandler(DomainHandler):
    """
    VLABench RLDS/TFRecord data handler.

    Each shard file is treated as one trajectory (traj_idx → shard file).
    Internally iterates all episodes within the shard.
    Uses 4 views: front, wrist, image_0, image_1.
    proprio: 7-dim ee_state (xyz+euler+gripper), no euler→axis_angle conversion.
    No 180° image rotation.
    """
    dataset_name = "vlabench_rlds"

    def __init__(self, meta: dict, num_views: int = 4) -> None:
        super().__init__(meta, num_views)
        self.data_dir = meta.get("data_dir", "")
        self.shard_files: List[str] = meta.get("datalist", [])

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "vlabench_joint",
        lang_aug_map: dict | None = None,
        **kwargs,
    ) -> Iterable[dict]:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        import tensorflow as tf  # lazy import

        shard_name = self.shard_files[traj_idx]
        shard_path = (
            shard_name if os.path.isabs(shard_name)
            else os.path.join(self.data_dir, shard_name)
        )

        dataset = tf.data.TFRecordDataset(shard_path)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            yield from self._iter_example(
                example,
                num_actions=num_actions,
                training=training,
                image_aug=image_aug,
                lang_aug_map=lang_aug_map,
            )

    def _iter_example(
        self,
        example,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
    ) -> Iterable[dict]:
        feat = example.features.feature

        # actions [T*7] → [T, 7]
        actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
        T = len(actions_flat) // 7
        if T == 0:
            return
        actions = actions_flat.reshape(T, 7)

        # proprio [T*7] → [T, 7]
        ee_flat = np.array(feat["steps/observation/ee_state"].float_list.value, dtype=np.float32)
        proprio = ee_flat.reshape(T, 7)

        # language instruction — take first frame
        lang_bytes = feat["steps/language_instruction"].bytes_list.value
        language = lang_bytes[0].decode("utf-8") if lang_bytes else ""
        if training and lang_aug_map and language in lang_aug_map:
            language = random.choice(lang_aug_map[language])

        # image bytes lists
        front_bytes  = list(feat["steps/observation/front"].bytes_list.value)
        wrist_bytes  = list(feat["steps/observation/wrist"].bytes_list.value)
        image0_bytes = list(feat["steps/observation/image_0"].bytes_list.value)
        image1_bytes = list(feat["steps/observation/image_1"].bytes_list.value)

        image_mask = torch.ones(self.num_views, dtype=torch.bool)

        indices = list(range(max(0, T - num_actions)))
        if training:
            random.shuffle(indices)

        for idx in indices:
            action_chunk = self._get_action_chunk(actions, idx, num_actions)

            imgs = []
            for img_bytes_list in [front_bytes, wrist_bytes, image0_bytes, image1_bytes]:
                img = Image.open(io.BytesIO(img_bytes_list[idx])).convert("RGB")
                if image_aug:
                    img = image_aug(img)
                imgs.append(img)

            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            image_input = torch.stack(imgs[: self.num_views], dim=0)

            yield {
                "language_instruction": language,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),
                "abs_trajectory": torch.tensor(action_chunk, dtype=torch.float32),
            }

    def _get_action_chunk(
        self, actions: np.ndarray, start_idx: int, num_actions: int
    ) -> np.ndarray:
        T, action_dim = actions.shape
        chunk = np.zeros((num_actions + 1, action_dim), dtype=np.float32)
        for i in range(num_actions + 1):
            chunk[i] = actions[min(start_idx + i, T - 1)]
        return chunk

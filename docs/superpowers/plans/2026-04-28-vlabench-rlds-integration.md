# VLABench RLDS 数据集接入实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 VLABench RLDS/TFRecord 数据集，并将训练脚本路径改为从环境变量读取。

**Architecture:** 新建 `VLABenchRLDSHandler` 遵循现有 handler 模式；新增 `vlabench_joint` action space；工具脚本生成元数据和归一化统计；训练脚本读取 `paths.env` 环境变量。

**Tech Stack:** Python, TensorFlow (tf.data.TFRecordDataset), PyTorch, PIL, numpy

---

## 文件清单

| 操作 | 文件 |
|------|------|
| 新建 | `datasets/domain_handler/vlabench_rlds.py` |
| 新建 | `create_vlabench_meta.py` |
| 新建 | `compute_vlabench_norm_stats.py` |
| 修改 | `datasets/domain_handler/registry.py` |
| 修改 | `datasets/domain_config.py` |
| 修改 | `models/action_hub.py` |
| 修改 | `train_smolvlm_small.sh` |
| 修改 | `train_smolvlm_large.sh` |

---

### Task 1: 新建 VLABenchRLDSHandler

**Files:**
- Create: `datasets/domain_handler/vlabench_rlds.py`

- [ ] **Step 1: 创建 handler 文件**

```python
# datasets/domain_handler/vlabench_rlds.py
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
        import tensorflow as tf  # lazy import — TF GPU disabled by worker_init_fn

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

    # ------------------------------------------------------------------
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
```

- [ ] **Step 2: Commit**

```bash
git add datasets/domain_handler/vlabench_rlds.py
git commit -m "feat: add VLABenchRLDSHandler for TFRecord data"
```

---

### Task 2: 注册 handler，更新 domain_config

**Files:**
- Modify: `datasets/domain_handler/registry.py`
- Modify: `datasets/domain_config.py`

- [ ] **Step 1: 在 registry.py 中注册**

在 `registry.py` 的 import 区域添加：
```python
from .vlabench_rlds import VLABenchRLDSHandler
```

在 `_REGISTRY` 字典中添加：
```python
    "vlabench_rlds": VLABenchRLDSHandler,
```

- [ ] **Step 2: 在 domain_config.py 中添加权重**

在 `DATA_WEIGHTS` 中添加：
```python
    "vlabench_rlds": 1.0,
```

在 `DATA_DOMAIN_ID` 中添加：
```python
    "vlabench_rlds": 1,
```

- [ ] **Step 3: Commit**

```bash
git add datasets/domain_handler/registry.py datasets/domain_config.py
git commit -m "feat: register vlabench_rlds handler and domain config"
```

---

### Task 3: 新增 vlabench_joint action space

**Files:**
- Modify: `models/action_hub.py`

- [ ] **Step 1: 在 action_hub.py 末尾（`__all__` 之前）添加**

```python
# =============================================================================
# VLABench Action Space
# =============================================================================
@register_action("vlabench_joint")
class VLABenchJointActionSpace(BaseActionSpace):
    """
    VLABench joint action space.

    Data layout:
      - state (proprio): 7-dim [xyz(3), euler(3), gripper(1)]
      - actions: 7-dim [xyz(3), euler(3), gripper(1)]
    """

    dim_action = 7
    dim_proprio = 7
    gripper_idx = (6,)

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
    ):
        super().__init__()
        self.use_quantile_norm = use_quantile_norm
        self.state_norm_stats: Optional[NormStats] = None
        self.action_norm_stats: Optional[NormStats] = None

        if norm_stats_path:
            self._load(norm_stats_path)

    def _load(self, path: str):
        stats_dict = load_norm_stats(path)
        if "state" in stats_dict:
            self.state_norm_stats = stats_dict["state"]
        if "actions" in stats_dict:
            self.action_norm_stats = stats_dict["actions"]

    def to(self, device):
        if self.state_norm_stats is not None:
            self.state_norm_stats.to(device)
        if self.action_norm_stats is not None:
            self.action_norm_stats.to(device)
        return super().to(device)

    def _norm(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        if stats.mean.device != x.device:
            stats.to(x.device)
        D = x.shape[-1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01, q99 = stats.q01[..., :D], stats.q99[..., :D]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return (x - stats.mean[..., :D]) / (stats.std[..., :D] + 1e-6)

    def _unnorm(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        if stats.mean.device != x.device:
            stats.to(x.device)
        D = x.shape[-1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01, q99 = stats.q01[..., :D], stats.q99[..., :D]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return x * (stats.std[..., :D] + 1e-6) + stats.mean[..., :D]

    def compute_loss(self, pred, target):
        return {"velocity_loss": torch.mean(torch.square(pred - target))}

    def preprocess(self, proprio, action, mode="train"):
        if self.state_norm_stats is not None:
            proprio = self._norm(proprio, self.state_norm_stats)
        if self.action_norm_stats is not None:
            action = self._norm(action, self.action_norm_stats)
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_norm_stats is not None:
            return self._unnorm(action, self.action_norm_stats)
        return action
```

更新 `__all__` 列表，添加 `"VLABenchJointActionSpace"`。

- [ ] **Step 2: Commit**

```bash
git add models/action_hub.py
git commit -m "feat: add vlabench_joint action space"
```

---

### Task 4: 新建 create_vlabench_meta.py

**Files:**
- Create: `create_vlabench_meta.py`

- [ ] **Step 1: 创建脚本**

```python
#!/usr/bin/env python3
"""
生成 VLABench RLDS 数据集元数据 JSON。

用法:
    python create_vlabench_meta.py --output ./datasets/metas/vlabench_train.json
    python create_vlabench_meta.py --data_dir /custom/path --output ./datasets/metas/vlabench_train.json
"""
import argparse
import glob
import json
import os


def create_vlabench_meta(data_dir: str, output_path: str | None = None) -> dict:
    pattern = os.path.join(data_dir, "primitive-train.tfrecord-*-of-*")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(f"No TFRecord shards found in {data_dir}")

    datalist = [os.path.basename(f) for f in shard_files]

    meta = {
        "dataset_name": "vlabench_rlds",
        "data_dir": data_dir,
        "datalist": datalist,
        "num_files": len(datalist),
        "action_dim": 7,
        "proprio_dim": 7,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta to {output_path} ({len(datalist)} shards)")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SIMVLA_VLABENCH_DATA", "./datasets/vlabench/data/1.0.0"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    meta = create_vlabench_meta(args.data_dir, args.output)
    print(f"Found {meta['num_files']} shard files")
```

- [ ] **Step 2: Commit**

```bash
git add create_vlabench_meta.py
git commit -m "feat: add create_vlabench_meta.py script"
```

---

### Task 5: 新建 compute_vlabench_norm_stats.py

**Files:**
- Create: `compute_vlabench_norm_stats.py`

- [ ] **Step 1: 创建脚本**

```python
#!/usr/bin/env python3
"""
计算 VLABench RLDS 数据集的归一化统计（均值/标准差/分位数）。

用法:
    python compute_vlabench_norm_stats.py --output ./norm_stats/vlabench_norm.json
    python compute_vlabench_norm_stats.py --max_files 50 --output ./norm_stats/vlabench_norm.json
"""
import argparse
import glob
import json
import os

import numpy as np


def compute_norm_stats(data_dir: str, max_files: int | None = None) -> dict:
    import tensorflow as tf
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    pattern = os.path.join(data_dir, "primitive-train.tfrecord-*-of-*")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No TFRecord shards found in {data_dir}")

    if max_files is not None:
        shard_files = shard_files[:max_files]

    print(f"Processing {len(shard_files)} shard files...")

    all_actions = []
    all_states = []

    for i, shard_path in enumerate(shard_files):
        if i % 50 == 0:
            print(f"  {i}/{len(shard_files)}")
        dataset = tf.data.TFRecordDataset(shard_path)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feat = example.features.feature

            actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
            T = len(actions_flat) // 7
            if T == 0:
                continue
            all_actions.append(actions_flat.reshape(T, 7))

            ee_flat = np.array(feat["steps/observation/ee_state"].float_list.value, dtype=np.float32)
            all_states.append(ee_flat.reshape(T, 7))

    actions_arr = np.concatenate(all_actions, axis=0)  # [N, 7]
    states_arr  = np.concatenate(all_states,  axis=0)  # [N, 7]

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "q01":  np.percentile(arr, 1, axis=0).tolist(),
            "q99":  np.percentile(arr, 99, axis=0).tolist(),
        }

    return {
        "state":   _stats(states_arr),
        "actions": _stats(actions_arr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SIMVLA_VLABENCH_DATA", "./datasets/vlabench/data/1.0.0"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_files", type=int, default=None,
                        help="采样加速：只处理前 N 个 shard")
    args = parser.parse_args()

    stats = compute_norm_stats(args.data_dir, args.max_files)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved norm stats to {args.output}")
```

- [ ] **Step 2: Commit**

```bash
git add compute_vlabench_norm_stats.py
git commit -m "feat: add compute_vlabench_norm_stats.py script"
```

---

### Task 6: 训练脚本路径变量化

**Files:**
- Modify: `train_smolvlm_small.sh`
- Modify: `train_smolvlm_large.sh`

- [ ] **Step 1: 修改 train_smolvlm_small.sh 路径配置段**

将：
```bash
LIBERO_DATA_DIR="./datasets/metas"
NORM_STATS_PATH="./norm_stats/libero_norm.json"
TRAIN_METAS_PATH="./datasets/metas/libero_train.json"

# SmolVLM backbone (can be local path or HuggingFace repo)
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
```

替换为：
```bash
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
LIBERO_DATA_DIR="./datasets/metas"
NORM_STATS_PATH="./norm_stats/libero_norm.json"
TRAIN_METAS_PATH="./datasets/metas/libero_train.json"
```

- [ ] **Step 2: 修改 train_smolvlm_large.sh 路径配置段**（同上）

- [ ] **Step 3: Commit**

```bash
git add train_smolvlm_small.sh train_smolvlm_large.sh
git commit -m "feat: read SMOLVLM_MODEL from env var in training scripts"
```

---

### Task 7: 更新 __init__.py 导出

**Files:**
- Modify: `datasets/domain_handler/__init__.py`

- [ ] **Step 1: 添加 VLABenchRLDSHandler 导出**

在 `__init__.py` 中添加：
```python
from .vlabench_rlds import VLABenchRLDSHandler
```

并在 `__all__`（如有）中添加 `"VLABenchRLDSHandler"`。

- [ ] **Step 2: Commit**

```bash
git add datasets/domain_handler/__init__.py
git commit -m "chore: export VLABenchRLDSHandler from domain_handler"
```

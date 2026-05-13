"""
VLABench RLDS/TFRecord 数据处理器

核心功能：
  - 读取 VLABench TFRecord 格式的机器人操控数据
  - 每个 shard 文件对应一条轨迹（trajectory）
  - 支持 4 个视角：front, wrist, image_0, image_1
  - 输出：图像、语言指令、动作序列、本体感知、前一帧wrist图像（用于MotionCNN帧差分）

数据格式：
  - 动作/本体感知：7维 [xyz(3), euler(3), gripper(1)]
  - 图像：4个视角的字节序列
  - 语言指令：取首帧的文本描述
"""

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
    VLABench RLDS/TFRecord 数据处理器

    每个 shard 文件对应一条轨迹（traj_idx → shard file），
    内部迭代该 shard 中的所有 episode（片段）。

    关键设计：
      - 4 个视角：front, wrist, image_0, image_1
      - 本体感知：7维 ee_state (xyz+euler+gripper)，不做欧拉角→轴角转换
      - 图像不做 180° 旋转
    """

    dataset_name = "vlabench_rlds"

    def __init__(self, meta: dict, num_views: int = 4) -> None:
        """
        参数
        ----
        meta : dict
            包含数据目录和 shard 文件列表的元数据
        num_views : int
            视角数量，默认 4（front, wrist, image_0, image_1）
        """
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
        """
        迭代一条轨迹中的所有片段（episode）

        参数
        ----
        traj_idx : int
            轨迹索引，对应 shard_files 列表中的文件
        num_actions : int
            动作块长度（预测未来多少步）
        training : bool
            是否为训练模式（训练时会随机打乱片段顺序）
        image_aug : callable
            图像增强函数（可选）
        action_mode : str
            动作空间模式，如 "vlabench_joint"
        lang_aug_map : dict
            语言增强映射表（可选）

        返回
        ----
        Iterable[dict]
            每个元素是一个样本字典，包含：
            - language_instruction: 语言指令
            - image_input: 4视角图像 [V, C, H, W]
            - image_mask: 视角掩码 [V]（bool）
            - proprio: 本体感知 [7]
            - abs_trajectory: 动作序列 [T+1, 7]
            - wrist_prev_pixels: 前一帧wrist图像（用于MotionCNN）
        """
        # 抑制 TensorFlow 日志
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        import tensorflow as tf  # lazy import，避免启动时加载

        # 获取 shard 文件路径
        shard_name = self.shard_files[traj_idx]
        shard_path = (
            shard_name if os.path.isabs(shard_name)
            else os.path.join(self.data_dir, shard_name)
        )

        # 读取 TFRecord 文件并迭代
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
        """
        解析单个 TFRecord example，生成训练样本

        关键步骤：
          1. 解析动作序列 [T*7] → [T, 7]
          2. 解析本体感知 [T*7] → [T, 7]
          3. 解析语言指令（取首帧）
          4. 解析4个视角的图像字节序列
          5. 遍历轨迹中的每个时间步，生成样本
        """
        feat = example.features.feature

        # 解析动作序列：[T*7] → [T, 7]
        # 7维：xyz(3) + euler(3) + gripper(1)
        actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
        T = len(actions_flat) // 7
        if T == 0:
            return  # 空轨迹，跳过
        actions = actions_flat.reshape(T, 7)

        # 解析本体感知：[T*7] → [T, 7]
        # 与动作维度相同（xyz + euler + gripper）
        ee_flat = np.array(feat["steps/observation/ee_state"].float_list.value, dtype=np.float32)
        proprio = ee_flat.reshape(T, 7)

        # 语言指令：取首帧的文本描述
        lang_bytes = feat["steps/language_instruction"].bytes_list.value
        language = lang_bytes[0].decode("utf-8") if lang_bytes else ""
        # 训练时可选的语言增强
        if training and lang_aug_map and language in lang_aug_map:
            language = random.choice(lang_aug_map[language])

        # 4个视角的图像字节序列
        front_bytes  = list(feat["steps/observation/front"].bytes_list.value)
        wrist_bytes  = list(feat["steps/observation/wrist"].bytes_list.value)
        image0_bytes = list(feat["steps/observation/image_0"].bytes_list.value)
        image1_bytes = list(feat["steps/observation/image_1"].bytes_list.value)

        # 视角掩码：所有视角都有效（全1）
        image_mask = torch.ones(self.num_views, dtype=torch.bool)

        # 遍历轨迹中的每个时间步
        # 训练时随机打乱顺序，避免按时间顺序学习
        indices = list(range(max(0, T - num_actions)))
        if training:
            random.shuffle(indices)

        for idx in indices:
            # 获取动作块：从 idx 开始的 num_actions+1 步
            action_chunk = self._get_action_chunk(actions, idx, num_actions)

            # 读取4个视角的当前帧图像
            imgs = []
            for img_bytes_list in [front_bytes, wrist_bytes, image0_bytes, image1_bytes]:
                img = Image.open(io.BytesIO(img_bytes_list[idx])).convert("RGB")
                if image_aug:
                    img = image_aug(img)
                imgs.append(img)

            # 如果视角数不足，用零填充
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            # 堆叠为 [V, C, H, W] 的张量
            image_input = torch.stack(imgs[: self.num_views], dim=0)

            # 前一帧 wrist 图像（用于运动引导注意力的帧差分）
            # idx=0 时用当前帧重复填充，差分图全零，不影响正常 attention
            # 这是 MotionCNN 的输入：当前帧 - 前一帧 = 运动区域
            prev_idx = max(0, idx - 1)
            wrist_prev = Image.open(io.BytesIO(wrist_bytes[prev_idx])).convert("RGB")
            if image_aug:
                wrist_prev = image_aug(wrist_prev)

            yield {
                "language_instruction": language,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),
                "abs_trajectory": torch.tensor(action_chunk, dtype=torch.float32),
                "wrist_prev_pixels": wrist_prev,  # PIL Image，由 dataset 统一转 tensor
            }

    def _get_action_chunk(
        self, actions: np.ndarray, start_idx: int, num_actions: int
    ) -> np.ndarray:
        """
        获取动作块：从 start_idx 开始的 num_actions+1 步动作

        参数
        ----
        actions : np.ndarray
            完整动作序列 [T, 7]
        start_idx : int
            起始时间步
        num_actions : int
            动作块长度

        返回
        ----
        np.ndarray
            动作块 [num_actions+1, 7]（包含当前帧）
        """
        T, action_dim = actions.shape
        chunk = np.zeros((num_actions + 1, action_dim), dtype=np.float32)
        for i in range(num_actions + 1):
            # 边界处理：超出轨迹长度时重复最后一帧
            chunk[i] = actions[min(start_idx + i, T - 1)]
        return chunk

#!/usr/bin/env python3
"""
离线 Action MSE 评估脚本

不需要模拟器，直接在 VLABench TFRecord 测试数据上评估模型的动作预测误差。
用于快速判断代码改动方向是否有效。

用法:
    python eval_action_mse.py \
        --checkpoint ./runs/simvla_vlabench_debug/ckpt-10000 \
        --norm_stats ./norm_stats/vlabench_norm.json \
        --data_dir /path/to/vlabench/data/1.0.0 \
        --num_shards 10 \
        --num_samples 200
"""

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def load_model(checkpoint_path, norm_stats_path, device):
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    print(f"Loading model from {checkpoint_path} ...")
    model = SmolVLMVLA.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)

    if norm_stats_path and os.path.exists(norm_stats_path):
        model.action_space.load_norm_stats(norm_stats_path)
        model.action_space.to(device)

    processor = SmolVLMVLAProcessor.from_pretrained(checkpoint_path)
    return model, processor


def load_samples_from_shards(data_dir, num_shards, num_samples, num_actions=10):
    """从 TFRecord shard 中随机采样，返回样本列表。"""
    import glob
    import tensorflow as tf

    pattern = os.path.join(data_dir, "primitive-train.tfrecord-*-of-*")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No TFRecord shards found in {data_dir}")

    shard_files = shard_files[:num_shards]
    print(f"Loading from {len(shard_files)} shards ...")

    samples = []
    for shard_path in shard_files:
        dataset = tf.data.TFRecordDataset(shard_path)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feat = example.features.feature

            actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
            T = len(actions_flat) // 7
            if T <= num_actions:
                continue
            actions = actions_flat.reshape(T, 7)

            ee_flat = np.array(feat["steps/observation/ee_state"].float_list.value, dtype=np.float32)
            proprio = ee_flat.reshape(T, 7)

            lang_bytes = feat["steps/language_instruction"].bytes_list.value
            language = lang_bytes[0].decode("utf-8") if lang_bytes else ""

            front_bytes  = list(feat["steps/observation/front"].bytes_list.value)
            wrist_bytes  = list(feat["steps/observation/wrist"].bytes_list.value)
            image0_bytes = list(feat["steps/observation/image_0"].bytes_list.value)
            image1_bytes = list(feat["steps/observation/image_1"].bytes_list.value)

            # 随机采样一个时间步
            idx = random.randint(0, T - num_actions - 1)
            action_chunk = actions[idx: idx + num_actions]  # [num_actions, 7]

            imgs = []
            for img_bytes_list in [front_bytes, wrist_bytes, image0_bytes, image1_bytes]:
                img = Image.open(io.BytesIO(img_bytes_list[idx])).convert("RGB")
                imgs.append(img)

            samples.append({
                "language": language,
                "images": imgs,           # list of 4 PIL images
                "proprio": proprio[idx],  # [7]
                "action": action_chunk,   # [num_actions, 7]
            })

            if len(samples) >= num_samples:
                return samples

    return samples


@torch.no_grad()
def evaluate(model, processor, samples, device, batch_size=8, num_actions=10):
    """计算 Action MSE。"""
    all_mse = []
    all_per_dim_mse = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]

        # 编码语言
        languages = [s["language"] for s in batch]
        lang = processor.encode_language(languages)
        input_ids = lang["input_ids"].to(device)

        # 编码图像
        # images: list of [4 PIL images] per sample → [B, 4, C, H, W]
        img_tensors = []
        for s in batch:
            imgs_t = processor.encode_image(s["images"])  # [4, C, H, W]
            img_tensors.append(imgs_t)
        image_input = torch.stack(img_tensors, dim=0).to(device)  # [B, 4, C, H, W]
        image_mask = torch.ones(len(batch), 4, dtype=torch.bool, device=device)

        # proprio
        proprio = torch.tensor(
            np.stack([s["proprio"] for s in batch]), dtype=torch.float32, device=device
        )

        # 推理
        pred_actions = model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio,
            steps=10,
        )  # [B, num_actions, 7]

        # 真实动作
        gt_actions = torch.tensor(
            np.stack([s["action"] for s in batch]), dtype=torch.float32, device=device
        )  # [B, num_actions, 7]

        # MSE
        mse = torch.mean((pred_actions - gt_actions) ** 2, dim=(1, 2))  # [B]
        per_dim = torch.mean((pred_actions - gt_actions) ** 2, dim=(0, 1))  # [7]

        all_mse.extend(mse.cpu().numpy().tolist())
        all_per_dim_mse.append(per_dim.cpu().numpy())

        if (i // batch_size) % 5 == 0:
            print(f"  [{i}/{len(samples)}] running MSE: {np.mean(all_mse):.4f}")

    mean_mse = np.mean(all_mse)
    per_dim_mse = np.mean(all_per_dim_mse, axis=0)
    return mean_mse, per_dim_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--norm_stats", default=None, help="归一化统计 JSON 路径")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SIMVLA_VLABENCH_DATA", "./datasets/vlabench/data/1.0.0"),
    )
    parser.add_argument("--num_shards", type=int, default=10, help="用于评估的 shard 数量")
    parser.add_argument("--num_samples", type=int, default=200, help="采样的样本数量")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载模型
    model, processor = load_model(args.checkpoint, args.norm_stats, args.device)

    # 加载样本
    print(f"Loading {args.num_samples} samples from {args.num_shards} shards ...")
    samples = load_samples_from_shards(
        args.data_dir, args.num_shards, args.num_samples, args.num_actions
    )
    print(f"Loaded {len(samples)} samples")

    # 评估
    print("Evaluating ...")
    mean_mse, per_dim_mse = evaluate(
        model, processor, samples, args.device, args.batch_size, args.num_actions
    )

    print("\n" + "=" * 50)
    print(f"Action MSE (mean): {mean_mse:.4f}")
    print(f"Per-dim MSE: xyz={per_dim_mse[:3].mean():.4f}  "
          f"euler={per_dim_mse[3:6].mean():.4f}  "
          f"gripper={per_dim_mse[6]:.4f}")
    print(f"Per-dim detail: {[f'{v:.4f}' for v in per_dim_mse]}")
    print("=" * 50)


if __name__ == "__main__":
    main()

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

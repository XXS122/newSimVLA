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


def create_vlabench_meta(data_dir: str, output_path: str | None = None, max_files: int | None = None) -> dict:
    pattern = os.path.join(data_dir, "primitive-train.tfrecord-*-of-*")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(f"No TFRecord shards found in {data_dir}")

    if max_files is not None:
        shard_files = shard_files[:max_files]

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
    parser.add_argument("--max_files", type=int, default=None,
                        help="只使用前 N 个 shard（用于生成调试子集）")
    args = parser.parse_args()

    meta = create_vlabench_meta(args.data_dir, args.output, args.max_files)
    print(f"Found {meta['num_files']} shard files")

#!/usr/bin/env python3
"""
SimVLA VLABench Policy Server (WebSocket)

在 simvla 环境中运行，接收 VLABench 客户端的观测，返回动作。

输入（来自 simvla_policy.py）：
  - observation/images: list of 4 numpy arrays [H, W, 3]（front/wrist/image_0/image_1）
  - observation/state: [7] float（xyz + euler + gripper）
  - prompt: str

输出：
  - actions: [[7] x action_horizon]（绝对位置，xyz + euler + gripper_cmd）

用法：
    CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_vlabench.py \
        --checkpoint ../../runs/simvla_vlabench_debug/ckpt-10000 \
        --norm_stats ../../norm_stats/vlabench_norm.json \
        --port 8200
"""

import argparse
import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# 禁止 HuggingFace 联网，强制使用本地缓存/路径
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Auto-load paths.env if present
_repo_root = Path(__file__).parent.parent.parent
_paths_env = _repo_root / "paths.env"
if _paths_env.exists():
    for _line in _paths_env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import websockets

try:
    import msgpack
    import msgpack_numpy
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack_numpy not installed")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
    "num_views": 4,
}


def load_model(checkpoint_path: str, norm_stats_path: str = None):
    global model, processor
    logger.info(f"Loading SimVLA from {checkpoint_path} ...")
    model = SmolVLMVLA.from_pretrained(checkpoint_path)
    model.to(device).eval()

    # 优先从 checkpoint 加载 processor，fallback 到 config 里的本地路径
    smolvlm_path = checkpoint_path
    try:
        processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)
    except Exception:
        smolvlm_path = model.config.smolvlm_model_path
        logger.info(f"Falling back to smolvlm_model_path: {smolvlm_path}")
        processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)

    if norm_stats_path and os.path.exists(norm_stats_path):
        logger.info(f"Loading norm stats: {norm_stats_path}")
        if hasattr(model.action_space, 'load_norm_stats'):
            model.action_space.load_norm_stats(norm_stats_path)
        elif hasattr(model.action_space, '_load'):
            model.action_space._load(norm_stats_path)
        model.action_space.to(device)

    logger.info(f"Model ready on {device}")


def preprocess_images(images_np):
    """
    images_np: list of 4 numpy arrays [H, W, 3] uint8
    returns: [1, 4, C, H, W], [1, 4] bool mask
    """
    size = CONFIG["image_size"]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tensors = [transform(Image.fromarray(img.astype(np.uint8))) for img in images_np]
    image_input = torch.stack(tensors, dim=0).unsqueeze(0)   # [1, 4, C, H, W]
    image_mask = torch.ones(1, 4, dtype=torch.bool)
    return image_input, image_mask


def infer(observation: Dict[str, Any]) -> Dict[str, Any]:
    try:
        images = observation["observation/images"]   # list of 4 np arrays
        state  = np.array(observation["observation/state"], dtype=np.float32)  # [7]
        prompt = observation.get("prompt", "")

        image_input, image_mask = preprocess_images(images)
        image_input = image_input.to(device)
        image_mask  = image_mask.to(device)

        lang = processor.encode_language([prompt])
        lang = {k: v.to(device) for k, v in lang.items()}

        proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            actions = model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=image_input,
                image_mask=image_mask,
                proprio=proprio,
                steps=CONFIG["action_horizon"],
            )

        return {"actions": actions.cpu().numpy()[0].tolist()}

    except Exception as e:
        logger.error(f"Inference error: {e}")
        traceback.print_exc()
        return {"actions": [[0.0] * CONFIG["action_dim"]] * CONFIG["action_horizon"]}


async def handle_connection(websocket, path=None):
    logger.info(f"Client connected: {websocket.remote_address}")
    try:
        metadata = {
            "model": "SimVLA-VLABench",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "image_size": CONFIG["image_size"],
            "num_views": CONFIG["num_views"],
        }
        if HAS_MSGPACK:
            await websocket.send(msgpack_numpy.packb(metadata, use_bin_type=True))
        else:
            import json
            await websocket.send(json.dumps(metadata))

        async for message in websocket:
            if HAS_MSGPACK and isinstance(message, bytes):
                request = msgpack_numpy.unpackb(message, raw=False)
            else:
                import json
                request = json.loads(message)

            result = infer(request)

            if HAS_MSGPACK:
                response = msgpack.packb(result, use_bin_type=True)
            else:
                import json
                response = json.dumps(result)
            await websocket.send(response)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info(f"Client disconnected: {websocket.remote_address}")


async def serve(host: str, port: int):
    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA VLABench server listening on {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    load_model(args.checkpoint, args.norm_stats)
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()

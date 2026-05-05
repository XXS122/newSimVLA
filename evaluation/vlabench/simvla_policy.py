"""
SimVLA Policy for VLABench evaluation.

在 vlabench conda 环境中使用，放到 VLABench/VLABench/evaluation/model/policy/ 目录下。

用法（在 VLABench 目录）：
    python scripts/evaluate_policy.py \
        --eval-track track_debug_simple \
        --policy simvla \
        --host localhost \
        --port 8200 \
        --n-episode 10 \
        --save-dir logs/simvla_debug
"""

import collections
import time
from typing import Tuple

import msgpack
import msgpack_numpy
import numpy as np
import websockets.sync.client

from VLABench.utils.utils import quaternion_to_euler


# 坐标系偏移（与 openpi 保持一致）
_COORD_OFFSET = np.array([0.0, -0.4, 0.78])


class SimVLAPolicy:
    """
    SimVLA policy client for VLABench evaluation.
    通过 WebSocket 连接 serve_smolvlm_vlabench.py 服务器。
    """

    def __init__(self, host: str = "localhost", port: int = 8200, replan_steps: int = 4):
        self._uri = f"ws://{host}:{port}"
        self._replan_steps = replan_steps
        self._action_plan = collections.deque(maxlen=replan_steps)
        self._timestep = 0
        self.name = "simvla"
        self.control_mode = "ee"

        print(f"Connecting to SimVLA server at {self._uri} ...")
        self._ws, self._metadata = self._wait_for_server()
        print(f"Connected. Server metadata: {self._metadata}")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._timestep = 0
        self._action_plan = collections.deque(maxlen=self._replan_steps)

    def predict(self, observation: dict, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 (target_pos [3], target_euler [3], gripper_state [2])
        """
        if self._timestep % self._replan_steps == 0:
            # 提取 4 个视角图像：right, left, front, wrist（VLABench 顺序）
            right, left, front, wrist = observation["rgb"]

            # 解析 ee_state: [x, y, z, qx, qy, qz, qw, gripper]
            ee_state = observation["ee_state"]
            pos   = ee_state[:3].copy()
            quat  = ee_state[3:7]
            gripper = float(ee_state[7]) if len(ee_state) > 7 else 0.0

            euler = quaternion_to_euler(quat)

            # 转换到模型坐标系
            pos_model = pos - _COORD_OFFSET
            state_model = np.concatenate([pos_model, euler, [gripper]]).astype(np.float32)

            policy_input = {
                "observation/images": [front, wrist, right, left],  # front/wrist/image_0/image_1
                "observation/state": state_model,
                "prompt": observation["instruction"],
            }

            action_chunk = self._infer(policy_input)["actions"]  # list of [7]
            self._action_plan.extend(action_chunk[: self._replan_steps])

        self._timestep += 1
        raw_action = self._action_plan.popleft()
        raw_action = np.array(raw_action, dtype=np.float32)

        target_pos   = raw_action[:3] + _COORD_OFFSET   # 转回世界坐标
        target_euler = raw_action[3:6]
        gripper_cmd  = float(raw_action[6])

        if gripper_cmd >= 0.1:
            gripper_state = np.ones(2) * 0.04   # 打开
        else:
            gripper_state = np.zeros(2)          # 关闭

        return target_pos, target_euler, gripper_state

    # ------------------------------------------------------------------
    def _infer(self, obs: dict) -> dict:
        data = msgpack_numpy.packb(obs, use_bin_type=True)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error: {response}")
        return msgpack.unpackb(response, raw=False)

    def _wait_for_server(self):
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                metadata = msgpack_numpy.unpackb(conn.recv(), raw=False)
                return conn, metadata
            except Exception:
                print("Waiting for SimVLA server ...")
                time.sleep(5)

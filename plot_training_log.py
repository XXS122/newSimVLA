"""
训练日志可视化脚本

从 train_smolvlm.log 中解析 loss、lr 等指标，绘制训练曲线。

用法：
    python plot_training_log.py --log_path /path/to/train_smolvlm.log
    python plot_training_log.py --log_path /path/to/train_smolvlm.log --smooth 50
"""

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    """解析训练日志，提取 step/loss/lr 数据。"""
    pattern = re.compile(
        r"\[(\d+)/\d+\]\s+"
        r"loss=([\d.]+)\s+"
        r"lr_core=([\d.e+-]+)\s+"
        r"lr_action=([\d.e+-]+)\s+"
        r"lr_vlm=([\d.e+-]+)"
    )

    steps, losses, lr_cores, lr_actions, lr_vlms = [], [], [], [], []

    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lr_cores.append(float(m.group(3)))
                lr_actions.append(float(m.group(4)))
                lr_vlms.append(float(m.group(5)))

    return {
        "step": np.array(steps),
        "loss": np.array(losses),
        "lr_core": np.array(lr_cores),
        "lr_action": np.array(lr_actions),
        "lr_vlm": np.array(lr_vlms),
    }


def find_last_run(data: dict):
    """找到最后一次连续训练的起点（step 回退说明重启了）。"""
    steps = data["step"]
    start_idx = 0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            start_idx = i
    return {k: v[start_idx:] for k, v in data.items()}


def smooth(values, window):
    """滑动平均平滑。"""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot(data: dict, smooth_window: int, output_path: str):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    steps = data["step"]
    loss = data["loss"]

    # Loss 曲线
    ax = axes[0]
    ax.plot(steps, loss, alpha=0.3, color="steelblue", linewidth=0.5, label="raw")
    if smooth_window > 1 and len(loss) > smooth_window:
        loss_smooth = smooth(loss, smooth_window)
        steps_smooth = steps[smooth_window - 1:]
        ax.plot(steps_smooth, loss_smooth, color="darkblue", linewidth=1.5,
                label=f"smooth (window={smooth_window})")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Learning Rate 曲线
    ax = axes[1]
    ax.plot(steps, data["lr_core"], label="lr_core (transformer)", linewidth=1.5)
    ax.plot(steps, data["lr_action"], label="lr_action (action heads)", linewidth=1.5, linestyle="--")
    ax.plot(steps, data["lr_vlm"], label="lr_vlm (SmolVLM backbone)", linewidth=1.5, linestyle=":")
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title("Learning Rate Schedule")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser("训练日志可视化")
    parser.add_argument("--log_path", type=str, required=True, help="train_smolvlm.log 路径")
    parser.add_argument("--smooth", type=int, default=20, help="Loss 平滑窗口大小（默认 20）")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径（默认保存在 log 同目录）")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"文件不存在: {log_path}")
        return

    data = parse_log(str(log_path))
    print(f"共解析 {len(data['step'])} 条记录")

    data = find_last_run(data)
    print(f"最后一次训练: step {data['step'][0]} → {data['step'][-1]}（共 {len(data['step'])} 条）")
    print(f"Loss: {data['loss'][0]:.4f} → {data['loss'][-1]:.4f}")

    output = args.output or str(log_path.parent / "training_curves.png")
    plot(data, args.smooth, output)


if __name__ == "__main__":
    main()

"""
训练日志可视化脚本
====================

从 SimVLA 训练日志（train_smolvlm.log）中解析每步的 loss，绘制训练曲线。

日志格式（来自 train_smolvlm.py 中的 logger 输出）：
    HH:MM:SS | INFO | __main__ | [step/total] loss=X.XXXX lr_core=... lr_action=... lr_vlm=... (X.XXs/it)

用法示例：
    # 默认平滑窗口 20，图片保存到当前工作目录下的 training_curves.png
    python tests/plot_training_log.py --log_path /path/to/train_smolvlm.log

    # 自定义平滑窗口（值越大曲线越平滑）
    python tests/plot_training_log.py --log_path /path/to/train_smolvlm.log --smooth 50

    # 指定输出路径
    python tests/plot_training_log.py --log_path /path/to/train_smolvlm.log --output my_curve.png

可视化约定：
    - 浅蓝色半透明曲线：原始 loss（每个 step 的真实值，抖动较大）
    - 深蓝色实线：滑动平均平滑后的 loss（反映整体趋势）
    - 两者叠加：既能看到抖动幅度，又能看清收敛趋势（TensorBoard / W&B 风格）
"""

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    """从训练日志中解析每一步的 step 和 loss。

    用正则匹配形如 `[1234/200000] loss=0.1234` 的字段。
    其他行（Args 打印、模型初始化信息等）会被自动跳过。

    参数
    ----
    log_path : str
        train_smolvlm.log 的绝对路径

    返回
    ----
    dict
        {"step": np.ndarray[int], "loss": np.ndarray[float]}
    """
    # 匹配 [step/total] loss=值 的格式
    # \[(\d+)/\d+\]   捕获 step 数字（忽略总步数）
    # loss=([\d.]+)    捕获 loss 数值
    pattern = re.compile(r"\[(\d+)/\d+\]\s+loss=([\d.]+)")

    steps, losses = [], []
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))

    return {"step": np.array(steps), "loss": np.array(losses)}


def find_last_run(data: dict):
    """裁掉历史重启数据，只保留最后一次连续训练的曲线。

    训练过程中可能多次中断重启（每次重启都会从 step 0 或 resume 的 step 开始），
    日志会累积所有次的输出。如果 step[i] <= step[i-1]，说明发生了重启，把起点更新到 i。

    例子：
        日志 step 序列: 0, 20, 40, 60, 0, 20, 40, 60, 80, ...
        重启点索引:                    ^
        返回结果只保留从第二个 0 开始的数据

    参数
    ----
    data : dict
        parse_log 的返回值，含 "step" 和 "loss" 两个数组

    返回
    ----
    dict
        同结构，但只包含最后一次连续训练的片段
    """
    steps = data["step"]
    start_idx = 0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            # 发现 step 回退 / 重复，说明进入了新一次训练
            start_idx = i
    return {k: v[start_idx:] for k, v in data.items()}


def smooth(values, window):
    """对 loss 序列做滑动平均平滑。

    用 np.convolve 实现等权重滑动平均。mode="valid" 表示只保留完全覆盖窗口的位置，
    因此输出长度 = len(values) - window + 1，前 window-1 个位置没有平滑值。

    参数
    ----
    values : np.ndarray
        原始 loss 序列
    window : int
        滑动窗口大小（值越大曲线越平滑，但会滞后）

    返回
    ----
    np.ndarray
        平滑后的序列（长度 = len(values) - window + 1）
    """
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot(data: dict, smooth_window: int, output_path: str):
    """绘制训练 loss 曲线并保存为 PNG 图片。

    绘制策略（参考 TensorBoard / W&B 的默认样式）：
        1. 原始 loss 用半透明细线（alpha=0.3，lw=0.5）画在背景
           —— 保留抖动信息，但不抢视觉焦点
        2. 平滑 loss 用深蓝实线（lw=1.5）覆盖在上面
           —— 展示整体下降趋势
        3. y 轴从 0 起步（而非 log 尺度或自动范围）
           —— 避免视觉误导，真实反映 loss 绝对值

    参数
    ----
    data : dict
        含 "step" 和 "loss" 两个 np.ndarray
    smooth_window : int
        平滑窗口大小，<=1 时跳过平滑
    output_path : str
        输出图片路径（通常是 training_curves.png）
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    steps = data["step"]
    loss = data["loss"]

    # 原始 loss 曲线：半透明薄线作为"背景纹理"
    ax.plot(steps, loss, alpha=0.3, color="steelblue", linewidth=0.5, label="raw")

    # 平滑 loss 曲线：需要至少 smooth_window 个点才能算出一个平滑值
    if smooth_window > 1 and len(loss) > smooth_window:
        loss_smooth = smooth(loss, smooth_window)
        # 平滑后前 smooth_window-1 个位置没有值，对应的 x 轴也要跳过相同长度
        steps_smooth = steps[smooth_window - 1:]
        ax.plot(steps_smooth, loss_smooth, color="darkblue", linewidth=1.5,
                label=f"smooth (window={smooth_window})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.set_ylim(bottom=0)          # y 轴从 0 起步
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser("训练日志可视化")
    parser.add_argument("--log_path", type=str, required=True,
                        help="train_smolvlm.log 的路径")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Loss 滑动平均窗口大小，默认 20（值越大越平滑）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径，默认保存在当前工作目录下的 training_curves.png")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"文件不存在: {log_path}")
        return

    # 1. 读取全部 step/loss 记录（可能包含多次重启的历史）
    data = parse_log(str(log_path))
    print(f"共解析 {len(data['step'])} 条记录")

    # 2. 只保留最后一次连续训练的数据
    data = find_last_run(data)
    print(f"最后一次训练: step {data['step'][0]} → {data['step'][-1]}（共 {len(data['step'])} 条）")
    print(f"Loss: {data['loss'][0]:.4f} → {data['loss'][-1]:.4f}")

    # 3. 绘图保存（默认到当前工作目录）
    output = args.output or str(Path.cwd() / "training_curves.png")
    plot(data, args.smooth, output)


if __name__ == "__main__":
    main()

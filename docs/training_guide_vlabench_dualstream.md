# VLABench 双流融合训练指南

## 你的配置

- 数据集：VLABench
- 融合方式：concat_linear
- 模型：SmolVLM-500M
- GPU：1 张卡
- 训练步数：10W（100,000）

## 前置准备

### 1. 确保数据准备好

```bash
cd /home/sapi/hyj/code/newSimVLA

# 检查数据是否存在
ls /home/sapi/hyj/vlabench/data/1.0.0/primitive-train.tfrecord-00000-of-00512

# 检查元数据和归一化统计
ls datasets/metas/vlabench_train.json
ls norm_stats/vlabench_norm.json
```

如果元数据或归一化统计不存在，脚本会自动创建。

### 2. 检查 paths.env 配置

```bash
cat paths.env
```

确保以下变量已设置：
- `SIMVLA_SMOLVLM_MODEL`：SmolVLM 模型路径
- `SIMVLA_VLABENCH_DATA`：VLABench 数据目录
- `SIMVLA_OUTPUT_DIR`：输出目录
- `SIMVLA_CUDA_DEVICES`：GPU 设备号（你的情况是 0）
- `SIMVLA_NUM_GPUS`：GPU 数量（你的情况是 1）

## 训练命令

### 方式 1：使用现成的脚本（推荐）

```bash
cd /home/sapi/hyj/code/newSimVLA

# 参数说明：
# $1 = batch_size（默认 32）
# $2 = learning_coef（默认 0.1）
# $3 = resume_ckpt（默认空，从头训练）
# $4 = fusion_type（默认 cross_attn，改成 concat_linear）

bash train_vlabench_dualstream.sh 32 0.1 "" concat_linear
```

**参数调整建议**（1 张卡）：
- batch_size：32（如果 OOM 改成 16）
- learning_coef：0.1（VLM 学习率系数）
- fusion_type：concat_linear

### 方式 2：直接调用 Python 脚本

如果脚本有问题，可以直接跑：

```bash
cd /home/sapi/hyj/code/newSimVLA

conda run -n simvla python train_smolvlm.py \
    --output_dir ./runs/vlabench_dualstream_concat_linear \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --smolvlm_model_path /home/sapi/hyj/models/smolvlm/SmolVLM-500M-Instruct \
    --action_mode vlabench_joint \
    --num_views 4 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --num_actions 10 \
    --iters 100000 \
    --freeze_steps 1000 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --image_size 384 \
    --use_dual_stream \
    --dual_stream_fusion concat_linear
```

## 训练过程中的监控

### 查看日志

```bash
# 实时查看日志
tail -f ./runs/vlabench_dualstream_concat_linear_*/train.log

# 或者查看最新的输出目录
ls -lt ./runs/ | head -5
```

### 查看 checkpoint

```bash
# 查看保存的 checkpoint
ls -la ./runs/vlabench_dualstream_concat_linear_*/ckpt-*/
```

## 恢复训练

如果训练中断，可以从 checkpoint 恢复：

```bash
bash train_vlabench_dualstream.sh 32 0.1 "./runs/vlabench_dualstream_concat_linear_20260509_10/ckpt-50000" concat_linear
```

## 预期结果

- **训练时间**：10W 步，1 张 GPU，batch_size=32，约 20-30 小时
- **显存占用**：约 20-25 GB（如果 OOM 降低 batch_size）
- **输出**：
  - checkpoint 每 10K 步保存一次
  - 日志每 20 步打印一次
  - 最终模型在 `./runs/vlabench_dualstream_concat_linear_*/ckpt-100000/`

## 常见问题

### Q1：OOM（显存不足）

降低 batch_size：
```bash
bash train_vlabench_dualstream.sh 16 0.1 "" concat_linear
```

### Q2：训练很慢

- 检查 GPU 利用率：`nvidia-smi`
- 如果 GPU 利用率低，可能是 I/O 瓶颈，增加 num_workers（但 VLABench 通常设 0）

### Q3：想改变训练步数

修改脚本中的 `ITERS` 变量，或直接在 Python 命令中加 `--iters 50000`

---

现在可以开始训练了！

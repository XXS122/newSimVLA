# OFP 自蒸馏推理加速（Self-Distillation Fast Inference）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Flow Matching 训练损失上增加自洽性损失（Self-Consistency Loss），训练模型在 1-2 步内生成高质量动作，推理速度提升 5-10 倍，无需额外教师模型。

**Architecture:** 在 `forward()` 中，对同一样本分别从 t=1 和 t=0.5 出发做 Euler 积分，要求两条路径在 t=0 处收敛到相同动作（自洽性约束）。总损失 = Flow Matching MSE + β × 自洽性 MSE。推理时只需将 `steps` 从 10 改为 1-2，无需重新训练。

**Tech Stack:** PyTorch, 现有 SmolVLMVLA Flow Matching 架构

---

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `models/configuration_smolvlm_vla.py` | 新增 `consistency_loss_weight` 配置项 |
| 修改 | `models/modeling_smolvlm_vla.py` | `forward()` 增加自洽性损失计算 |
| 修改 | `train_smolvlm.py` | 新增 `--consistency_loss_weight` 参数 |
| 修改 | `train_vlabench_debug.sh` | 新增自蒸馏开关 |
| 修改 | `evaluation/vlabench/serve_smolvlm_vlabench.py` | 新增 `--inference_steps` 参数 |

---

### Task 1: 新增配置项

**Files:**
- Modify: `models/configuration_smolvlm_vla.py:29-80`

- [ ] **Step 1: 添加自蒸馏配置字段**

在 `moe_aux_loss_weight: float = 0.01,` 之后添加：

```python
        # === Self-Distillation Fast Inference ===
        consistency_loss_weight: float = 0.0,  # 0 = 禁用，推荐 0.1
```

在对应的 `self.moe_aux_loss_weight = moe_aux_loss_weight` 之后添加：

```python
        self.consistency_loss_weight = consistency_loss_weight
```

- [ ] **Step 2: Commit**

```bash
git add models/configuration_smolvlm_vla.py
git commit -m "feat: add consistency_loss_weight config field"
```

---

### Task 2: 在 forward() 中实现自洽性损失

**Files:**
- Modify: `models/modeling_smolvlm_vla.py:324-384`

- [ ] **Step 1: 实现自洽性损失计算**

在 `forward()` 方法中，在现有 Flow Matching 损失计算之后（`velocity_loss = torch.mean(torch.square(v_t - u_t))` 之后），添加：

```python
        loss_dict = {"velocity_loss": velocity_loss}

        # 自洽性损失（Self-Consistency Loss）
        if self.config.consistency_loss_weight > 0:
            with torch.no_grad():
                # 多步参考：从 t=1 出发，10 步 Euler 积分到 t=0
                # 使用 stop_gradient，作为蒸馏目标
                x_ref = torch.randn_like(action_norm)
                dt_ref = -1.0 / 10
                t_ref = 1.0
                while t_ref > -dt_ref / 2:
                    t_ref_tensor = torch.full((B,), t_ref, device=device, dtype=action_norm.dtype)
                    v_ref = self.transformer(
                        vlm_features=enc["vlm_features"],
                        action_with_noise=x_ref,
                        t=t_ref_tensor,
                        proprio=proprio_norm,
                    )
                    x_ref = x_ref + dt_ref * v_ref
                    t_ref = t_ref + dt_ref
                x0_ref = x_ref  # [B, T_action, D] 多步参考终点

            # 单步快速：从 t=0.5 出发，1 步 Euler 积分到 t=0
            noise_half = torch.randn_like(action_norm)
            x_half = 0.5 * noise_half + 0.5 * action_norm  # t=0.5 插值点
            t_half = torch.full((B,), 0.5, device=device, dtype=action_norm.dtype)
            v_half = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_half,
                t=t_half,
                proprio=proprio_norm,
            )
            x0_fast = x_half + (-0.5) * v_half  # 单步到 t=0

            consistency_loss = torch.mean(torch.square(x0_fast - x0_ref))
            loss_dict["consistency_loss"] = self.config.consistency_loss_weight * consistency_loss
```

将原来的 `return {"velocity_loss": velocity_loss}` 替换为 `return loss_dict`。

**注意：** 如果 Task 5 of MoE 计划已实现（`loss_dict` 已存在），则只需在其基础上添加 `consistency_loss` 键，不需要重复创建 `loss_dict`。

- [ ] **Step 2: Commit**

```bash
git add models/modeling_smolvlm_vla.py
git commit -m "feat: add self-consistency loss to Flow Matching training"
```

---

### Task 3: 修改训练脚本

**Files:**
- Modify: `train_smolvlm.py`

- [ ] **Step 1: 添加 CLI 参数**

在 `--moe_aux_loss_weight` 之后添加：

```python
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0,
                        help="自洽性损失权重，0=禁用，推荐 0.1")
```

- [ ] **Step 2: 将参数传入 config**

```python
        consistency_loss_weight=args.consistency_loss_weight,
```

- [ ] **Step 3: Commit**

```bash
git add train_smolvlm.py
git commit -m "feat: add --consistency_loss_weight CLI arg"
```

---

### Task 4: 更新 debug 训练脚本

**Files:**
- Modify: `train_vlabench_debug.sh`

- [ ] **Step 1: 添加自蒸馏参数**

在 `DUAL_STREAM_FUSION=cross_attn` 之后添加：

```bash
CONSISTENCY_LOSS_WEIGHT=0.1   # 0.0 = 禁用自蒸馏
```

在 ARGS 构建中添加：

```bash
if [ "$(echo "$CONSISTENCY_LOSS_WEIGHT > 0" | bc -l)" = "1" ]; then
    ARGS="${ARGS} --consistency_loss_weight ${CONSISTENCY_LOSS_WEIGHT}"
fi
```

- [ ] **Step 2: Commit**

```bash
git add train_vlabench_debug.sh
git commit -m "feat: add consistency loss weight to debug training script"
```

---

### Task 5: 修改推理服务器支持可变步数

**Files:**
- Modify: `evaluation/vlabench/serve_smolvlm_vlabench.py`

- [ ] **Step 1: 在 CONFIG 中添加 inference_steps**

将：
```python
CONFIG = {
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
    "num_views": 4,
}
```

改为：
```python
CONFIG = {
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
    "num_views": 4,
    "inference_steps": 10,  # 可通过 --inference_steps 修改
}
```

- [ ] **Step 2: 在 infer() 中使用 CONFIG["inference_steps"]**

将：
```python
            actions = model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=image_input,
                image_mask=image_mask,
                proprio=proprio,
                steps=CONFIG["action_horizon"],
            )
```

改为：
```python
            actions = model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=image_input,
                image_mask=image_mask,
                proprio=proprio,
                steps=CONFIG["inference_steps"],
            )
```

- [ ] **Step 3: 在 main() 中添加 --inference_steps 参数**

```python
    parser.add_argument("--inference_steps", type=int, default=10,
                        help="Flow Matching 推理步数，自蒸馏训练后可设为 1-2")
```

在 `load_model(args.checkpoint, args.norm_stats)` 之前添加：

```python
    CONFIG["inference_steps"] = args.inference_steps
```

- [ ] **Step 4: Commit**

```bash
git add evaluation/vlabench/serve_smolvlm_vlabench.py
git commit -m "feat: add --inference_steps to vlabench server"
```

---

### Task 6: 验证

- [ ] **Step 1: 验证自洽性损失正常计算**

```bash
CUDA_VISIBLE_DEVICES=2 python train_smolvlm.py \
    --consistency_loss_weight 0.1 \
    --train_metas_path ./datasets/metas/vlabench_debug_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --action_mode vlabench_joint --num_views 4 \
    --iters 50 --batch_size 2 --output_dir /tmp/test_consistency \
    --log_interval 1
```

预期：日志中出现 `velocity_loss` 和 `consistency_loss` 两个损失项，`consistency_loss` 初始值通常在 0.5-2.0 之间，随训练下降。

- [ ] **Step 2: 对比不同推理步数的 Action MSE**

训练完 debug checkpoint 后（带 `--consistency_loss_weight 0.1`），运行：

```bash
for STEPS in 1 2 5 10; do
    echo "=== steps=$STEPS ==="
    python eval_action_mse.py \
        --checkpoint ./runs/simvla_vlabench_debug/ckpt-10000 \
        --norm_stats ./norm_stats/vlabench_norm.json \
        --num_shards 5 --num_samples 100
done
```

预期：steps=1 的 MSE 比无自蒸馏训练的 steps=1 低，steps=2 接近 steps=10 的效果。

- [ ] **Step 3: 验证推理服务器 --inference_steps 参数生效**

```bash
# 启动服务器（1步推理）
CUDA_VISIBLE_DEVICES=2 python evaluation/vlabench/serve_smolvlm_vlabench.py \
    --checkpoint ./runs/simvla_vlabench_debug/ckpt-10000 \
    --norm_stats ./norm_stats/vlabench_norm.json \
    --inference_steps 1 \
    --port 8201 &

# 等待启动后 Ctrl+C 停止，确认日志无报错
```

预期：服务器正常启动，日志显示 `SimVLA VLABench server listening on 0.0.0.0:8201`。

- [ ] **Step 4: Commit**

```bash
git commit -m "test: verify self-distillation training and variable inference steps"
```

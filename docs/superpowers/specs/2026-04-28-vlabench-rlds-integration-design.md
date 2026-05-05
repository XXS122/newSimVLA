# VLABench RLDS 数据集接入设计

**日期：** 2026-04-28  
**状态：** 已批准

---

## 背景

当前代码只支持 LIBERO HDF5 格式。需要接入 VLABench 数据集（RLDS/TFRecord 格式），同时将训练脚本中的硬编码路径改为从 `paths.env` 环境变量读取。

---

## 数据格式（已确认）

数据位于 `/datasets/vlabench/data/1.0.0`，共 512 个 shard 文件（`primitive-train.tfrecord-XXXXX-of-00512`）。

每个 TFRecord episode 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `steps/action` | float [T×7] | xyz(3) + euler(3) + gripper(1) |
| `steps/observation/ee_state` | float [T×7] | proprio：xyz(3) + euler(3) + gripper(1) |
| `steps/observation/front` | JPEG bytes [T] | 正面相机，224×224 |
| `steps/observation/wrist` | JPEG bytes [T] | 腕部相机，224×224 |
| `steps/observation/image_0` | JPEG bytes [T] | 额外视角，224×224 |
| `steps/observation/image_1` | JPEG bytes [T] | 额外视角，224×224 |
| `steps/language_instruction` | bytes [T] | 语言指令（每步重复，取第一帧） |
| `steps/is_first / is_last` | int [T] | episode 边界 |

---

## 架构

遵循现有 handler 模式，最小侵入。

### 新建文件

- `datasets/domain_handler/vlabench_rlds.py` — TFRecord handler
- `create_vlabench_meta.py` — 生成元数据 JSON
- `compute_vlabench_norm_stats.py` — 计算归一化统计

### 修改文件

- `datasets/domain_handler/registry.py` — 注册 `vlabench_rlds`
- `datasets/domain_config.py` — 添加采样权重
- `models/action_hub.py` — 新增 `vlabench_joint` action space
- `train_smolvlm_small.sh` / `train_smolvlm_large.sh` — 路径读环境变量

### 不改动

`dataset_smolvlm.py`、`modeling_smolvlm_vla.py`、所有 LIBERO 相关代码。

---

## VLABenchRLDSHandler 数据流

```
iter_episode(traj_idx)
  → 打开 datalist[traj_idx] 对应的 tfrecord 文件
  → tf.data.TFRecordDataset 遍历每个 episode
  → 解析 tf.train.Example：
      actions  = float_list[T*7] → reshape [T, 7]
      ee_state = float_list[T*7] → reshape [T, 7]
      front / wrist / image_0 / image_1 = bytes_list[T] → PIL.Image
      language = bytes_list[T] → 取第一帧字符串
  → 构建 action chunk（num_actions+1 帧，末尾 padding）
  → image_mask = [True, True, True, True]（4个视角全用）
  → yield 样本字典
```

**关键决策：**
- 每个 shard 文件作为一个 trajectory（方案 A），handler 内部遍历文件里的所有 episode
- 使用全部 4 个视角：`front`、`wrist`、`image_0`、`image_1`，训练时 `--num_views 4`
- proprio 直接使用 7-dim ee_state，不做 euler→axis_angle 转换
- 图像不做 180° 旋转（VLABench 图像方向正常）
- TensorFlow GPU 禁用由现有 `worker_init_fn` 处理

---

## vlabench_joint Action Space

```python
@register_action("vlabench_joint")
class VLABenchJointActionSpace(BaseActionSpace):
    dim_action = 7   # xyz(3) + euler(3) + gripper(1)
    dim_proprio = 7  # xyz(3) + euler(3) + gripper(1)
```

---

## 工具脚本

### create_vlabench_meta.py

扫描 `$SIMVLA_VLABENCH_DATA`（fallback `./datasets/vlabench/data/1.0.0`），输出：

```json
{
  "dataset_name": "vlabench_rlds",
  "data_dir": "/datasets/vlabench/data/1.0.0",
  "datalist": ["primitive-train.tfrecord-00000-of-00512", ...],
  "num_files": 512,
  "action_dim": 7,
  "proprio_dim": 7
}
```

### compute_vlabench_norm_stats.py

遍历 tfrecord 文件，计算 action 和 proprio 的均值/标准差/分位数。支持 `--max_files N` 采样加速。输出格式与 `libero_norm.json` 一致。

### 训练脚本路径变量化

```bash
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
DATA_DIR="${SIMVLA_VLABENCH_DATA:-/datasets/vlabench/data/1.0.0}"
# 输出目录和 norm stats 保持相对路径 ./runs/ ./norm_stats/
```

---

## 使用流程

```bash
source paths.env

# 一次性准备
python create_vlabench_meta.py --output ./datasets/metas/vlabench_train.json
python compute_vlabench_norm_stats.py --output ./norm_stats/vlabench_norm.json

# 训练
bash train_smolvlm_small.sh
```

---

## 约束

- VLABench（`--num_views 4`）与 LIBERO（`--num_views 3`）不能混合训练，维度不同
- 仅支持 `vlabench_joint` action mode，不影响现有 LIBERO 训练

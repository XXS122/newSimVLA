# VLABench RLDS 数据集接入方案

## 一、背景

当前代码只支持 LIBERO HDF5 格式。需要接入 VLABench 数据集（RLDS/TFRecord 格式，存放于 `/datasets/vlabench/data/1.0.0`），同时将代码中的硬编码路径改为从 `paths.env` 读取，方便迁移到其他服务器。

---

## 二、VLABench 数据格式（已通过读取一帧确认）

| 字段 | 类型 | 说明 |
|------|------|------|
| `steps/action` | float [T×7] | 动作：xyz(3) + euler(3) + gripper(1) |
| `steps/observation/ee_state` | float [T×7] | 本体感知：xyz(3) + euler(3) + gripper(1) |
| `steps/observation/front` | JPEG bytes [T] | 正面相机，224×224 |
| `steps/observation/wrist` | JPEG bytes [T] | 腕部相机，224×224 |
| `steps/observation/image_0/1` | JPEG bytes [T] | 额外视角（暂不使用） |
| `steps/language_instruction` | bytes [T] | 语言指令（每步重复） |
| `steps/is_first / is_last` | int [T] | episode 边界标记 |
| `episode_metadata/file_path` | bytes | 原始 HDF5 路径 |

**与 LIBERO 的关键差异：**
- 格式：TFRecord（不是 HDF5）
- proprio 是 **7-dim**（LIBERO 是 8-dim，多一个 gripper 状态）
- 图像分辨率：224×224（LIBERO 是 128×128）
- 每个 `.tfrecord` 文件包含多个 episode

---

## 三、需要改动的文件

```
新建：
  datasets/domain_handler/vlabench_rlds.py   ← TFRecord 数据读取器
  create_vlabench_meta.py                    ← 生成训练元数据 JSON
  compute_vlabench_norm_stats.py             ← 计算归一化统计

修改：
  datasets/domain_handler/registry.py        ← 注册 vlabench_rlds
  datasets/domain_config.py                  ← 添加采样权重
  models/action_hub.py                       ← 新增 vlabench_joint action space
  train_smolvlm_small.sh                     ← 路径改为读取环境变量
  train_smolvlm_large.sh                     ← 路径改为读取环境变量
  paths.env                                  ← 补充 SIMVLA_OUTPUT_DIR 变量
```

---

## 四、各文件改动说明

### 1. `datasets/domain_handler/vlabench_rlds.py`（新建）

核心数据读取器，逻辑如下：

```
读取 tfrecord 文件
  → 解析每个 episode 的所有步骤
  → 提取 front + wrist 图像（2个视角，与 LIBERO 一致）
  → 提取 ee_state 作为 proprio（7-dim，直接使用）
  → 提取 action（7-dim）
  → 构建 action chunk（num_actions 帧）
  → yield 样本字典
```

输出格式与 LIBERO handler 完全一致：
```python
{
  "language_instruction": str,
  "image_input": Tensor[3, C, 384, 384],   # 3个视角槽，前2个有效
  "image_mask": BoolTensor[3],              # [True, True, False]
  "proprio": Tensor[7],
  "abs_trajectory": Tensor[num_actions+1, 7],
}
```

### 2. `models/action_hub.py`（修改）

新增 `vlabench_joint` action space，proprio 是 7-dim：

```python
@register_action("vlabench_joint")
class VLABenchJointActionSpace(BaseActionSpace):
    dim_action = 7   # xyz(3) + euler(3) + gripper(1)
    dim_proprio = 7  # xyz(3) + euler(3) + gripper(1)
```

### 3. `train_smolvlm_small.sh` / `train_smolvlm_large.sh`（修改）

将硬编码路径替换为读取环境变量（带默认值兜底）：

```bash
# 从 paths.env 读取（source paths.env 后生效）
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
DATA_DIR="${SIMVLA_VLABENCH_DATA:-/datasets/vlabench/data/1.0.0}"
OUTPUT_DIR="${SIMVLA_OUTPUT_DIR:-./runs}/simvla_vlabench_small"
```

### 4. `paths.env`（修改）

补充 `SIMVLA_OUTPUT_DIR` 变量：

```bash
SIMVLA_OUTPUT_DIR="/datasets/simvla_output"
```

### 5. `create_vlabench_meta.py`（新建）

扫描 tfrecord 文件，生成元数据 JSON，路径从 `$SIMVLA_VLABENCH_DATA` 读取：

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

### 6. `compute_vlabench_norm_stats.py`（新建）

遍历 tfrecord 文件，计算 action 和 proprio 的均值/标准差/分位数，输出格式与 `libero_norm.json` 完全一致。

---

## 五、使用流程（改动后）

```bash
# 第一步：配置路径（只需改这一个文件）
vim paths.env
source paths.env

# 第二步：生成元数据（一次性）
python create_vlabench_meta.py \
    --output ./datasets/metas/vlabench_train.json

# 第三步：计算归一化统计（一次性）
python compute_vlabench_norm_stats.py \
    --output ./norm_stats/vlabench_norm.json

# 第四步：训练
bash train_smolvlm_small.sh
# 或
bash train_smolvlm_large.sh
```

迁移到新服务器时，只需修改 `paths.env` 中的路径，其他不变。

---

## 六、不改动的部分

- `datasets/dataset_smolvlm.py` — 无需改动，已支持任意 handler
- `models/modeling_smolvlm_vla.py` — 无需改动，action space 通过 `action_mode` 参数切换
- LIBERO 相关代码 — 完全保留，两个数据集可以混合训练

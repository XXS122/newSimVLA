# dual_stream.py 代码逐行讲解

> 从 `__main__` 开始，按代码实际执行顺序讲解。面向 Python 基础、不懂 PyTorch 的初学者。

---

## 背景：这个文件在干什么？

想象一个机器人，它有 4 个摄像头：
- **front**（正前方）：看到桌子上有什么东西
- **image_0、image_1**（左右两侧）：看到周围环境
- **wrist**（手腕上）：看到手正在抓什么

前三个摄像头看的是"大场景"（**静态流**），手腕摄像头看的是"手在干嘛"（**动态流**）。

**这个文件的目的**：把"手在干嘛"的信息融合到"大场景"里去，让模型同时理解"场景里有什么"和"手正在做什么动作"。

---

## 第一块：文件开头的导入

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
```

**解释：**

- `torch`：PyTorch 库，做深度学习用的。你可以把它理解为"高级版 numpy"，能在 GPU 上跑，能自动求导。
- `torch.nn`：里面装着各种"积木块"，比如线性层（矩阵乘法）、归一化层等。你用这些积木搭建神经网络。
- `torch.nn.functional`：里面装着各种"函数"，比如 softmax、attention 计算等。和 nn 的区别是：nn 里的是"有参数的模块"，F 里的是"无状态的纯函数"。

---

## 第二块：配置参数

```python
DATA_DIR = "/home/sapi/hyj/vlabench/data/1.0.0"
SHARD    = "primitive-train.tfrecord-00000-of-00512"
SHARD_PATH = os.path.join(DATA_DIR, SHARD)

VLM_HIDDEN   = 960    
NUM_PATCHES  = 64     
NUM_VIEWS    = 4      
TEXT_LEN     = 20     
BATCH_SIZE   = 2      
```

**解释：**

- `SHARD_PATH`：一个真实的数据文件路径，TFRecord 格式（TensorFlow 的数据存储格式）
- `VLM_HIDDEN = 960`：VLM 模型输出的特征维度。意思是模型用 960 个数字来描述一个 token。
- `NUM_PATCHES = 64`：每个视角的图片被切成 64 个小方块（patch）。真实训练时是 1024，这里为了调试省内存用 64。
- `NUM_VIEWS = 4`：4 个摄像头视角（front、wrist、image_0、image_1）
- `TEXT_LEN = 20`：文本指令（比如"把杯子放桌上"）编码后有 20 个 token
- `BATCH_SIZE = 2`：一次处理 2 个样本

### TEXT_LEN = 20 的含义

这里是调试用的模拟值，随便定的。真实运行时，文本长度取决于指令的实际字数。比如"pick up the red cup"经过分词器（tokenizer）切分后可能变成 8 个 token，"put the blue bottle on the top shelf"可能变成 12 个 token。代码里设了 `language_max_length = 50` 作为上限，不够的补零，超过的截断。

这里写 20 纯粹是为了调试时模拟一个合理的文本长度，换成 10 或 30 都行，不影响测试逻辑。

### BATCH_SIZE 高和低的影响

batch_size 就是"一次喂给模型多少个样本"。

- **设高了**（比如 64、128）：
  - 优点：训练更稳定，因为每次算梯度用的样本多，方向更准确
  - 优点：GPU 利用率高，并行计算效率好
  - 缺点：显存占用大，可能 OOM（内存不够）
  - 缺点：有研究表明太大的 batch 可能导致模型泛化能力变差

- **设低了**（比如 1、2、4）：
  - 优点：省显存
  - 优点：梯度有噪声，有时反而帮助模型跳出局部最优
  - 缺点：训练不稳定，loss 曲线抖动大
  - 缺点：GPU 利用率低，训练慢

这里 `BATCH_SIZE = 2` 纯粹是调试用，只是为了验证代码逻辑能跑通，不是训练。

---

## 第三块：读取真实数据

```python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
```

- 设置环境变量让 TensorFlow 别打印一堆警告信息，然后导入 TensorFlow（因为数据是 TFRecord 格式，需要用 TF 来读）。

```python
frames = []
dataset = tf.data.TFRecordDataset(SHARD_PATH)
```

- `frames`：一个空列表，用来存读出来的图片
- `TFRecordDataset`：打开那个数据文件，可以一条一条遍历里面的记录

```python
for raw_record in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    feat = example.features.feature
```

- 每条 `raw_record` 是一段二进制数据，`ParseFromString` 把它解析成结构化的格式
- `feat` 就是解析后的字典，可以用 key 取出里面的图片、动作等数据

```python
    front_bytes  = list(feat["steps/observation/front"].bytes_list.value)
    wrist_bytes  = list(feat["steps/observation/wrist"].bytes_list.value)
    image0_bytes = list(feat["steps/observation/image_0"].bytes_list.value)
    image1_bytes = list(feat["steps/observation/image_1"].bytes_list.value)
    T = len(front_bytes)
```

- 从数据里取出 4 个视角的图片字节流。每个是一个列表，列表长度 = 这条轨迹有多少个时间步（T）。
- 比如机器人执行了 50 步动作，那 `front_bytes` 就有 50 张图片的字节数据。

### 关键概念：一帧 = 一个时间步

在这个数据集里，一帧 = 一个时间步（timestep）。

每个时间步同时记录了：
- 4 个摄像头的图片（front、wrist、image_0、image_1）
- 机器人的动作（7 维：xyz 位移 + 欧拉角 + 夹爪）
- 机器人的本体感知状态（7 维：当前位姿）

所以 T=62 意味着这条轨迹里机器人从开始到完成任务，总共走了 62 步，每一步都拍了照片、记录了动作。

### 不同轨迹的长度不同

不同轨迹长度差别很大：
- 第 0 个文件：62 步
- 第 5 个文件：159 步
- 第 10 个文件：148 步
- 第 50 个文件：72 步
- 第 100 个文件：78 步

这很正常——不同任务难度不同，简单的任务机器人几十步就完成了，复杂的可能要 150+ 步。这是数据采集时固定的：控制器每执行一步动作，就同时保存一次所有传感器的数据（4 张图片 + 动作 + 状态）。

---

## 第四块：逐行拆解读取循环

```python
for t in range(min(T, BATCH_SIZE - len(frames))):
```

- `T` = 这条轨迹总共有多少个时间步（比如机器人执行了 50 步，T=50）
- `BATCH_SIZE - len(frames)` = 我还需要几帧（BATCH_SIZE=2，如果已经读了 0 帧，还需要 2 帧）
- `min(T, ...)` = 取两者中较小的那个。意思是：要么轨迹里的帧不够了，要么我已经够了，哪个先到就停

所以这个循环就是：**从这条轨迹里一帧一帧地取，取到我够 2 帧为止**。

```python
    imgs = []
    for buf in [front_bytes[t], wrist_bytes[t], image0_bytes[t], image1_bytes[t]]:
        imgs.append(Image.open(io.BytesIO(buf)).convert("RGB"))
```

- `front_bytes[t]` = 第 t 个时间步的 front 相机图片的原始字节（一堆二进制数据，比如 JPEG 编码）
- `io.BytesIO(buf)` = 把字节数据包装成一个"文件对象"（因为 `Image.open` 需要接收文件或文件对象）
- `Image.open(...).convert("RGB")` = 解码成 PIL 图片，转成 RGB 格式（3 通道彩色）
- 循环 4 次，得到 `imgs = [front图片, wrist图片, image_0图片, image_1图片]`

```python
    frames.append(imgs)
```

把这一帧的 4 张图片作为一个整体加入 `frames` 列表。

```python
    if len(frames) >= BATCH_SIZE:
        break
```

内层循环的提前退出：如果已经凑够 2 帧了，不用继续读了。

```python
if len(frames) >= BATCH_SIZE:
    break
```

外层循环的提前退出：外层循环是遍历 TFRecord 里的多条记录（多条轨迹）。如果第一条轨迹就够 2 帧了，就不用读第二条轨迹了。

### 总结

这段代码的目的就是从数据文件里读出 2 帧，每帧包含 4 个视角的图片。最终 `frames` 长这样：

```
frames = [
    [front_0, wrist_0, image0_0, image1_0],   # 第 0 帧
    [front_1, wrist_1, image0_1, image1_1],   # 第 1 帧
]
```

---

## 第五块：模拟 VLM 输出

```python
device = torch.device("cpu")
print(f"[dual_stream debug] 使用设备: {device}（调试模式强制 CPU）")
```

- `device` 就是指定计算在哪里运行：CPU 还是 GPU
- 这里强制用 CPU，因为 GPU 显存被其他进程占满了

```python
T_enc = NUM_VIEWS * NUM_PATCHES + TEXT_LEN
```

- `NUM_VIEWS = 4`（4 个视角）
- `NUM_PATCHES = 64`（每个视角 64 个 patch token）
- `TEXT_LEN = 20`（文本 20 个 token）
- `T_enc = 4 * 64 + 20 = 276`

意思是：VLM 模型的输出会有 276 个 token。前 256 个是图像 token（4 视角 × 64），后 20 个是文本 token。

```python
vlm_features = torch.randn(BATCH_SIZE, T_enc, VLM_HIDDEN, device=device)
```

- `torch.randn(...)` 生成一个随机张量（从标准正态分布采样）
- 形状 `[2, 276, 960]`：2 个样本，每个 276 个 token，每个 token 960 维
- 这是**模拟** VLM 的输出。真实训练时，这个张量是 VLM 模型计算出来的，这里为了调试直接用随机数代替

```python
num_valid_views = torch.tensor([4, 3], device=device)
```

- 第 1 个样本有 4 个有效视角
- 第 2 个样本有 3 个有效视角（比如某个视角坏了或者没有）

---

## 第六块：测试三种融合方式

```python
for fusion_type in ["add", "concat_linear", "cross_attn"]:
    model = DualStreamFusion(
        hidden_size=VLM_HIDDEN,
        fusion_type=fusion_type,
        num_patches_per_view=NUM_PATCHES,
        static_view_indices=[0, 2, 3],   # front / image_0 / image_1
        dynamic_view_indices=[1],         # wrist
    ).to(device)
```

这里创建了 3 个不同的 `DualStreamFusion` 模块，分别用三种融合方式。

- `static_view_indices=[0, 2, 3]`：视角 0、2、3 是静态流（front、image_0、image_1）
- `dynamic_view_indices=[1]`：视角 1 是动态流（wrist）
- `.to(device)`：把模块放到 CPU 上

```python
    out = model(vlm_features, num_valid_views, num_patches_per_view=NUM_PATCHES)
```

调用模块的 `forward` 方法，输入是：
- `vlm_features`：[2, 276, 960] 的张量
- `num_valid_views`：[4, 3] 表示两个样本的有效视角数
- `num_patches_per_view`：64

输出 `out` 也是 [2, 276, 960]，形状和输入一样。

---

## 补充：两帧图像的文本指令

从真实数据读出的两帧文本：

```
帧 0 的文本指令：Please select the painting of style ukiyo-e.
帧 1 的文本指令：Please select the painting of style ukiyo-e.
```

这是机器人要执行的任务指令。在训练时，这段文本会被分词器（tokenizer）切分成多个 token，然后和图像 token 一起输入到 VLM 模型。

---

## 补充：num_valid_views 的含义详解

```python
num_valid_views = torch.tensor([4, 3], device=device)
```

这是**模拟**的，不是真实数据。含义是：
- 第 1 个样本有 4 个有效视角（所有视角都正常）
- 第 2 个样本有 3 个有效视角（某个视角缺失，比如摄像头坏了或没装）

真实情况下，VLABench 数据集所有样本通常都有 4 个有效视角。这里设 [4, 3] 是为了**测试代码能否处理"某个样本缺少一个视角"的情况**。

### 这个模拟有什么意义？

这个模拟的意义是**测试代码的鲁棒性**。

虽然真实情况下，VLABench 数据集所有样本都有 4 个完整视角，不会缺失。但在实际部署时可能遇到：
- 某个摄像头坏了
- 某个摄像头没装
- 某个视角的图片损坏无法读取

如果代码只在"所有视角都有"的情况下测试，就无法发现这些边界情况的 bug。设 [4, 3] 就是在模拟"第二个样本少了一个视角"，验证代码能否正确处理。

如果代码写得不好，可能会：
- 数组越界
- 维度不匹配
- 融合结果错误

所以这个模拟是**防御性编程**——提前测试异常情况，确保代码足够健壮。

DualStreamFusion 的 `_get_stream_indices` 方法就是为了处理这种情况——它会根据 `num_valid_views` 过滤掉无效的视角索引。

---

## 第七块：DualStreamFusion 的 forward 方法（第 1-2 步）

现在我们进入 DualStreamFusion 的核心逻辑。forward 方法分 6 步，先讲前 2 步。

### 第 1 步：分离图像 token 和文本 token

```python
B, T_enc, D = vlm_features.shape
n = num_patches_per_view if num_patches_per_view is not None else self.num_patches_per_view

max_img_tokens = int(max(num_valid_views).item()) * n
img_tokens = vlm_features[:, :max_img_tokens, :]
text_tokens = vlm_features[:, max_img_tokens:, :]
```

**数值示例**：
- `B=2, T_enc=276, D=960`
- `n=64`（每个视角 64 个 patch token）
- `max(num_valid_views) = max([4, 3]) = 4`
- `max_img_tokens = 4 * 64 = 256`
- `img_tokens = vlm_features[:, :256, :]`：前 256 个 token 是图像
- `text_tokens = vlm_features[:, 256:, :]`：后 20 个 token 是文本

**直觉**：把输入的 token 序列分成两部分。图像部分要融合，文本部分保持不变。

### 第 2 步：按视角索引拆分为静态流和动态流

```python
static_parts = []
dynamic_parts = []

for b in range(B):  # 遍历 batch 中的 2 个样本
    n_valid = int(num_valid_views[b].item())  # 第 b 个样本的有效视角数
    s_idx, d_idx = self._get_stream_indices(n_valid)  # 获取静态/动态索引
    
    s_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in s_idx], dim=0)
    d_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in d_idx], dim=0)
    static_parts.append(s_tokens)
    dynamic_parts.append(d_tokens)
```

**举例**（第 1 个样本，b=0）：
- `n_valid = 4`（4 个有效视角）
- `s_idx = [0, 2, 3]`（静态流：front、image_0、image_1）
- `d_idx = [1]`（动态流：wrist）
- `img_tokens[0]` 的布局：`[view0_64 | view1_64 | view2_64 | view3_64]`
- `s_tokens = cat([view0_64, view2_64, view3_64]) = [192个token]`
- `d_tokens = cat([view1_64]) = [64个token]`

**举例**（第 2 个样本，b=1）：
- `n_valid = 3`（只有 3 个有效视角）
- `s_idx = [0, 2]`（过滤掉了索引 3，因为只有 3 个视角）
- `d_idx = [1]`
- `s_tokens = [128个token]`
- `d_tokens = [64个token]`

**直觉**：从图像 token 中按视角索引切片，把属于静态流的视角拼在一起，把属于动态流的视角拼在一起。

---

## 补充问题解答

### 问题 1：你是怎么确定前 256 个就是图像的，后面的是文本的？

这不是我"确定"的，而是**代码设计的约定**。

在 `modeling_smolvlm_vla.py` 的 `forward_vlm_efficient` 方法里，VLM 的输出是这样构造的：

```python
for b in range(B):
    num_valid = valid_per_sample[b].item()
    sample_image_feats = full_image_features[b, :num_valid]  # 图像特征
    sample_text_embeds = text_embeds[b]                       # 文本特征
    combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # 拼接
    batch_inputs_embeds.append(combined)
```

**关键一行**：`torch.cat([sample_image_feats, sample_text_embeds], dim=0)`

这说明 VLM 输出的 token 序列是这样排列的：
```
[所有图像 token | 所有文本 token]
```

所以：
- 前面 `num_views * num_patches_per_view` 个 token 是图像
- 后面的 token 是文本

这是一个**硬编码的约定**，整个代码都依赖这个约定。如果改变这个顺序，代码就会出错。

### 问题 2：你只是用了数据的格式，没有真正读取到数据啊

你说得对。我在 `__main__` 里读取了真实的图片数据（frames），但后面的 `vlm_features` 是**随机生成的**，不是真实的 VLM 输出。

为什么这样做？因为：
1. 加载完整的 VLM 模型（SmolVLM-500M）需要很多显存和时间
2. 这个调试脚本的目的是**验证 DualStreamFusion 的逻辑是否正确**，不是验证 VLM 的输出
3. 用随机张量就足以测试融合模块的形状变换、索引操作等逻辑

**真实的数据流**应该是这样的：
```
真实图片 → VLM 模型 → vlm_features [B, T_enc, D] → DualStreamFusion → 融合后的特征
```

但在这个调试脚本里，我们跳过了 VLM 模型，直接用随机张量代替 vlm_features。这样可以快速验证 DualStreamFusion 的代码逻辑。

如果要用真实的 VLM 输出，需要：
1. 加载 SmolVLM 模型
2. 把读取的图片和文本指令输入到 VLM
3. 得到真实的 vlm_features
4. 再传给 DualStreamFusion

但这样会很慢，不适合快速调试。

---

## 第八块：DualStreamFusion 的 forward 方法（第 3 步）

### 先理清概念：什么是"样本"？

在这个代码里，**样本（sample）= 一帧视频**。

- `BATCH_SIZE = 2` 意思是一次处理 2 帧视频
- 第 1 个样本 = 第 1 帧视频
- 第 2 个样本 = 第 2 帧视频

每一帧视频都有 4 个摄像头的图片（front、wrist、image_0、image_1）。

### 为什么 192 和 128 不同？

回顾一下：
- `T_enc = NUM_VIEWS * NUM_PATCHES + TEXT_LEN = 4 * 64 + 20 = 276`
- 前 256 个 token 是图像（4 视角 × 64 patch）
- 后 20 个 token 是文本

**但是**，`num_valid_views = [4, 3]` 意思是：
- 第 1 个样本有 4 个有效视角
- 第 2 个样本只有 3 个有效视角（某个摄像头坏了）

所以在第 2 步拆分时：
- 第 1 个样本的静态流：视角 [0, 2, 3] = 3 个视角 × 64 patch = **192 个 token**
- 第 2 个样本的静态流：视角 [0, 2] = 2 个视角 × 64 patch = **128 个 token**（因为只有 3 个有效视角，索引 3 被过滤掉了）

### 第 3 步：Padding 对齐

```python
max_s = max(t.shape[0] for t in static_parts)
max_d = max(t.shape[0] for t in dynamic_parts)

static_padded = torch.zeros(B, max_s, D, device=vlm_features.device, dtype=vlm_features.dtype)
dynamic_padded = torch.zeros(B, max_d, D, device=vlm_features.device, dtype=vlm_features.dtype)
for b in range(B):
    static_padded[b, :static_parts[b].shape[0]] = static_parts[b]
    dynamic_padded[b, :dynamic_parts[b].shape[0]] = dynamic_parts[b]
```

**问题**：第 1 个样本的静态流有 192 个 token，第 2 个样本的静态流有 128 个 token。它们长度不一样，怎么放在一个 batch 里？

**解决方案**：Padding（补零）。

**数值示例**：
- `max_s = max(192, 128) = 192`
- `max_d = max(64, 64) = 64`
- `static_padded` 形状：`[2, 192, 960]`
- `dynamic_padded` 形状：`[2, 64, 960]`

**具体操作**：
- 第 1 个样本：`static_padded[0, :192] = static_parts[0]`（192 个 token，满）
- 第 2 个样本：`static_padded[1, :128] = static_parts[1]`，后面 64 个位置补零

**直觉**：把不同长度的序列补齐到同一长度，这样才能放在一个 batch 里一起计算。

---

## 第九块：DualStreamFusion 的 forward 方法（第 4 步 - Add 融合方式）

### 第 4 步：执行融合 - 方式 1：Add（相加）

```python
if self.fusion_type == "add":
    dynamic_means = torch.zeros(B, 1, D, device=vlm_features.device, dtype=vlm_features.dtype)
    for b in range(B):
        n_d = dynamic_parts[b].shape[0]
        dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)
    fused_static = static_padded + dynamic_means
```

### 问题：dynamic_means 不是全零张量吗？均值是怎么算进去的？

对，`torch.zeros(...)` 创建的确实是全零张量。但紧接着下面的循环**把每个位置的值覆盖掉了**：

```python
dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)
```

这行是**赋值操作**，不是累加。逐步拆解：

**第 1 步：`dynamic_padded[b, :n_d]`**

取第 b 个样本的动态流有效 token，形状 `[64, 960]`（64 个 token，每个 960 维）。

**第 2 步：`.mean(dim=0)`**

沿着第 0 维（token 维度）求平均：
```
token_0:  [0.1, 0.3, ..., 0.5]   ← 960 个数
token_1:  [0.2, 0.1, ..., 0.3]
...
token_63: [0.4, 0.2, ..., 0.1]
─────────────────────────────────
均值:     [0.25, 0.2, ..., 0.3]  ← 960 个数的平均
```

结果是一个 `[960]` 的向量，代表 64 个 wrist patch token 的平均特征。

**第 3 步：`dynamic_means[b, 0] = ...`**

把这个 `[960]` 的向量写入 `dynamic_means` 的第 b 个样本、第 0 个位置。

所以 `dynamic_means` 最终不是全零，而是：
```
dynamic_means[0, 0] = 第 1 个样本的 wrist 均值  ← [960]
dynamic_means[1, 0] = 第 2 个样本的 wrist 均值  ← [960]
```

形状 `[2, 1, 960]`。

**为什么先建零张量再赋值，而不是直接计算？**

因为 batch 里每个样本的有效 token 数可能不同（n_d 不同），没法直接用一行矩阵运算搞定，只能用循环逐个计算，然后写入预先分配好的张量。

---

## 补充：为什么静态流做 Q，动态流做 K/V？

这是 Cross-Attention 的核心设计决策，用一个生活比喻来解释：

**比喻：你（静态流）在查字典（动态流）**

- **Q（Query）= 你提出的问题**：静态流的每个 patch 说"我想知道手腕附近发生了什么"
- **K（Key）= 字典的目录**：动态流的每个 patch 说"我这里有这些信息"
- **V（Value）= 字典的内容**：动态流的每个 patch 实际包含的特征值

**计算过程**：
1. 静态流的每个 patch（Q）和动态流的所有 patch（K）计算相似度
2. 相似度高的动态 patch 贡献更多（权重大）
3. 最终静态 patch 获得动态流的加权信息（V）

**为什么不反过来？**

如果动态流做 Q，静态流做 K/V，那就变成"手腕视角去查询场景信息"，输出的是动态流的更新结果。但我们的目的是**更新静态流**（让场景理解获得运动信息补充），所以必须静态流做 Q。

**输出形状也说明了这一点**：

Cross-Attention 的输出形状和 Q 一致。
- Q 是静态流 `[B, T_s, D]`，所以输出也是 `[B, T_s, D]`
- 输出的是**更新后的静态流特征**，动态流只是提供信息，自身不变

---

## 第十块：DualStreamFusion 的 forward 方法（第 4 步 - Concat+Linear 融合方式）

```python
elif self.fusion_type == "concat_linear":
    dynamic_means = torch.zeros(B, 1, D, device=vlm_features.device, dtype=vlm_features.dtype)
    for b in range(B):
        n_d = dynamic_parts[b].shape[0]
        dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)
    dynamic_gap = dynamic_means.expand_as(static_padded)
    fused_static = self.norm(self.fusion_linear(
        torch.cat([static_padded, dynamic_gap], dim=-1)
    ))
```

**第 1 步：计算动态流均值（和 Add 方式一样）**

`dynamic_means` 形状 `[2, 1, 960]`，每个样本一个 wrist 均值向量。

**第 2 步：expand_as**

```python
dynamic_gap = dynamic_means.expand_as(static_padded)
```

- `dynamic_means` 形状：`[2, 1, 960]`
- `static_padded` 形状：`[2, 192, 960]`
- `expand_as` 把 `[2, 1, 960]` 扩展成 `[2, 192, 960]`

扩展的方式是**复制**，不是新建数据：
```
dynamic_means[0, 0] = [0.25, 0.2, ..., 0.3]  ← 1 个向量
扩展后：
dynamic_gap[0, 0]   = [0.25, 0.2, ..., 0.3]  ← 复制 192 次
dynamic_gap[0, 1]   = [0.25, 0.2, ..., 0.3]
...
dynamic_gap[0, 191] = [0.25, 0.2, ..., 0.3]
```

**第 3 步：torch.cat 拼接**

```python
torch.cat([static_padded, dynamic_gap], dim=-1)
```

- `static_padded` 形状：`[2, 192, 960]`
- `dynamic_gap` 形状：`[2, 192, 960]`
- `dim=-1` 表示在最后一维（特征维度）拼接
- 结果形状：`[2, 192, 1920]`

每个 token 从 960 维变成 1920 维，前 960 维是场景信息，后 960 维是运动信息。

**第 4 步：fusion_linear 压缩**

```python
self.fusion_linear(...)  # nn.Linear(1920, 960)
```

- 输入：`[2, 192, 1920]`
- 输出：`[2, 192, 960]`

线性层把 1920 维压缩回 960 维，**学习如何混合场景信息和运动信息**。

**第 5 步：LayerNorm**

```python
self.norm(...)
```

对输出做层归一化，稳定数值分布。

**和 Add 方式的区别**：
- Add：直接相加，没有可学习参数，场景和运动信息各占一半权重
- Concat+Linear：线性层可以学习"场景信息和运动信息各应该占多少权重"，更灵活

---

## 第十一块：DualStreamFusion 的 forward 方法（第 5-6 步）

### 第 5 步：将融合后的静态流 token 写回原始位置

```python
fused_img_tokens = img_tokens.clone()
for b in range(B):
    n_valid = int(num_valid_views[b].item())
    s_idx, _ = self._get_stream_indices(n_valid)
    for j, i in enumerate(s_idx):
        src_start = j * n
        src_end = src_start + n
        if src_end <= static_parts[b].shape[0]:
            fused_img_tokens[b, i*n:(i+1)*n] = fused_static[b, src_start:src_end]
```

**为什么需要这一步？**

回顾一下数据流：
- 原始 `img_tokens` 布局：`[view0 | view1 | view2 | view3]`（按视角顺序排列）
- 融合后的 `fused_static` 布局：`[view0 | view2 | view3]`（只有静态流，顺序也变了）

所以需要把 `fused_static` 里的内容**写回到正确的位置**。

**举例（第 1 个样本，b=0）**：
- `s_idx = [0, 2, 3]`
- `j=0, i=0`：`fused_static[0, 0:64]` → 写回 `fused_img_tokens[0, 0:64]`（view0）
- `j=1, i=2`：`fused_static[0, 64:128]` → 写回 `fused_img_tokens[0, 128:192]`（view2）
- `j=2, i=3`：`fused_static[0, 128:192]` → 写回 `fused_img_tokens[0, 192:256]`（view3）
- view1（wrist）的位置不动，保持原始值

**`img_tokens.clone()` 的作用**：

先复制一份 `img_tokens`，再在复制上修改。这样 wrist（动态流）的 token 保持原始值不变，只有静态流的 token 被替换成融合后的值。

### 第 6 步：拼接图像 token 和文本 token，恢复原始形状

```python
return torch.cat([fused_img_tokens, text_tokens], dim=1)
```

- `fused_img_tokens` 形状：`[2, 256, 960]`（图像部分，静态流已融合）
- `text_tokens` 形状：`[2, 20, 960]`（文本部分，全程未动）
- 拼接结果：`[2, 276, 960]`

**和输入形状完全一样**，这就是为什么调试时能看到输出 shape 和输入 shape 相同。

---

## 总结：forward 方法的完整数据流

```
输入 vlm_features [2, 276, 960]
        ↓
第1步：分离图像和文本
  img_tokens  [2, 256, 960]
  text_tokens [2,  20, 960]
        ↓
第2步：按视角拆分
  static_parts: [样本0: 192个token, 样本1: 128个token]
  dynamic_parts: [样本0: 64个token, 样本1: 64个token]
        ↓
第3步：padding 对齐
  static_padded  [2, 192, 960]
  dynamic_padded [2,  64, 960]
        ↓
第4步：融合（concat_linear）
  fused_static [2, 192, 960]
        ↓
第5步：写回原始位置
  fused_img_tokens [2, 256, 960]
        ↓
第6步：拼接文本
输出 [2, 276, 960]  ← 和输入形状完全一样
```

---

## 待续...

下一块：CrossAttentionFusion 的 forward 方法详解


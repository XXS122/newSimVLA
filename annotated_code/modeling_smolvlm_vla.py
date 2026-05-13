"""
SmolVLM-VLA 主模型

核心功能：
  - 基于 SmolVLM-500M-Instruct 的视觉-语言-动作（VLA）策略模型
  - 将图像和语言编码为 VLM 特征
  - 通过 Flow Matching 生成机器人动作序列

关键组件：
  1. SmolVLM 视觉-语言骨干网络（500M 参数）
  2. SmolVLMActionTransformer（流匹配动作头）
  3. 动作空间（前/后处理 + 损失计算）

创新点：
  - 双流多视角融合（DualStreamFusion）
  - 运动引导跨视角注意力（MotionCNN）
  - ActionVAE 隐式扩散策略
  - AdaLN + Concat 混合条件注入
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer
from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig


class SmolVLMVLA(PreTrainedModel):
    """
    SmolVLM-VLA: 基于 SmolVLM-500M-Instruct 的视觉-语言-动作策略模型

    核心组件：
      - SmolVLM-500M-Instruct 骨干网络（视觉-语言融合）
      - SmolVLMActionTransformer（流匹配动作头）
      - 动作空间（前/后处理 + 损失计算）

    与 FlorenceVLA 的区别：
      - 所有视角一起输入 VLM（无 aux_visual_inputs）
      - 512x512 图像分辨率（SmolVLM-500M 使用 512x512 patches）
      - 高效的 500M 参数模型
    """

    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # ========== 核心设置 ==========
        self.num_actions: int = config.num_actions  # 动作块长度（预测未来多少步）
        self.use_proprio: bool = config.use_proprio  # 是否使用本体感知
        self.action_mode: str = config.action_mode.lower()  # 动作空间模式
        self.image_size: int = config.image_size  # 图像尺寸（384 或 512）
        self.num_views: int = config.num_views  # 视角数量（VLABench 为 4）

        # ========== 动作空间 ==========
        # 根据动作模式构建动作空间（如 vlabench_joint, galaxea_joint 等）
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action  # 动作维度（VLABench 为 7）
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)  # 本体感知维度

        # ========== SmolVLM 骨干网络 ==========
        # 加载预训练的 SmolVLM-500M-Instruct 模型
        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.smolvlm_model_path,
            torch_dtype=torch.float32,  # 使用 float32 保证训练稳定性
            trust_remote_code=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.smolvlm_model_path,
            trust_remote_code=True,
        )

        # 获取 SmolVLM 的隐藏层维度
        # SmolVLM-500M 的 text_config.hidden_size = 576
        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        logging.info(f"SmolVLM hidden size: {vlm_hidden_size}")

        # ========== DiT/AdaLN 模式设置 ==========
        self.use_adaln = getattr(config, 'use_adaln', False)

        # ========== ActionVAE 隐式扩散策略（RoLD arxiv:2403.07312）==========
        # 可选模块：将动作块编码至低维隐空间做 Flow Matching
        self.use_action_vae = getattr(config, "use_action_vae", False)
        self.action_vae = None
        self.latent_flow_net = None

        if self.use_action_vae:
            # 启用 ActionVAE 模式
            from .action_vae import ActionVAE, LatentFlowNet
            latent_dim = getattr(config, "latent_dim", 32)  # 隐变量维度（默认 32）

            # ActionVAE：动作序列 [B,T,7] ↔ 隐变量 z [B,32]
            self.action_vae = ActionVAE(
                dim_action=dim_action,
                seq_len=config.num_actions,
                vlm_hidden=vlm_hidden_size,
                latent_dim=latent_dim,
            )

            # LatentFlowNet：隐空间速度场 MLP（4层残差）
            self.latent_flow_net = LatentFlowNet(
                latent_dim=latent_dim,
                vlm_hidden=vlm_hidden_size,
                dim_proprio=dim_proprio,
            )

            # VAE 损失权重
            self.vae_beta = getattr(config, "vae_beta", 0.001)  # KL 散度权重
            self.vae_recon_weight = getattr(config, "vae_recon_weight", 1.0)  # 重建损失权重
            self.transformer = None  # 不使用标准 Transformer

            logging.info(
                f"✓ ActionVAE Latent Diffusion: latent_dim={latent_dim}, "
                f"β={self.vae_beta}, recon_w={self.vae_recon_weight}"
            )
        else:
            # 标准 Flow Matching 模式（动作序列空间）
            self.transformer = SmolVLMActionTransformer(
                hidden_size=config.hidden_size,
                vlm_hidden_size=vlm_hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dim_action=dim_action,
                dim_propio=dim_proprio,
                dim_time=config.dim_time,
                max_len_seq=config.max_len_seq,
                use_adaln=self.use_adaln,
                proprio_history_len=getattr(config, "proprio_history_len", 1),
                use_adaln_hybrid=getattr(config, "use_adaln_hybrid", False),
            )

            if self.use_adaln:
                logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
            else:
                logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        # ========== 双流融合（可选）==========
        # 静态视角（front/image_0/image_1）作 Query，动态视角（wrist）作 Key/Value
        self.dual_stream_fusion = None
        if config.use_dual_stream:
            from .dual_stream import DualStreamFusion
            # VLM 特征维度 = text_model 的 hidden_size（576）
            vlm_hidden = self.vlm.config.text_config.hidden_size
            self.dual_stream_fusion = DualStreamFusion(
                hidden_size=vlm_hidden,
                fusion_type=config.dual_stream_fusion,  # "add" | "concat_linear" | "cross_attn"
                use_missing_token=getattr(config, "use_missing_token", False),
            )
            logging.info(f"[SmolVLMVLA] Dual-stream fusion enabled: {config.dual_stream_fusion}, hidden_size={vlm_hidden}")

        # ========== 运动引导跨视角注意力（Motion-Guided Cross-Attention）==========
        # 帧差分图 → MotionCNN → 运动激活图 → 注入 cross-attention bias
        self.motion_cnn = None
        self.use_motion_guided_attn = getattr(config, "use_motion_guided_attn", False)
        if self.use_motion_guided_attn:
            from .dual_stream import MotionCNN

            # 通过 dummy forward 获取实际 num_patches（SigLIP + connector 决定）
            # 这是因为 SmolVLM 的视觉编码器输出 patch 数量取决于图像尺寸
            with torch.no_grad():
                _dummy = torch.zeros(1, 3, config.image_size, config.image_size,
                                     device="cpu", dtype=torch.float32)
                _vis = self.vlm.model.vision_model(pixel_values=_dummy).last_hidden_state
                if hasattr(self.vlm.model, "connector"):
                    _vis = self.vlm.model.connector(_vis)
                _num_patches = _vis.shape[1]

            # MotionCNN：帧差分 → 3层CNN → 运动激活图 [B, num_patches]
            self.motion_cnn = MotionCNN(num_patches=_num_patches, image_size=config.image_size)
            logging.info(
                f"[SmolVLMVLA] Motion-Guided Cross-Attention enabled: "
                f"num_patches={_num_patches}, image_size={config.image_size}"
            )

        # FastAPI 应用（用于推理服务）
        self.app: FastAPI | None = None

    # ============================= SmolVLM 编码器 =============================
    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - 多视角图像
        image_mask: torch.Tensor,           # [B, V] (bool 或 0/1)
        language_instruction: list[str] | None = None,  # 可选的文本提示
    ) -> Dict[str, torch.Tensor]:
        """
        通过 SmolVLM2 编码多视角图像

        所有视角一起处理，生成统一的特征表示。
        不需要 aux_visual_inputs - 所有视角都通过 VLM 处理。

        参数
        ----
        pixel_values : torch.FloatTensor
            多视角图像 [B, V, C, H, W]
        image_mask : torch.Tensor
            视角掩码 [B, V]（bool 或 0/1）
        language_instruction : list[str] | None
            可选的文本提示列表

        返回
        ----
        Dict[str, torch.Tensor]
            "vlm_features": [B, T_enc, D] - VLM 特征
        """
        # 处理输入维度
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, 0]

        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device

        # 为每个样本准备图像
        # SmolVLM 可以处理多张图像作为多图像推理
        batch_features = []

        for b in range(B):
            # 获取当前样本的有效图像
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]  # [num_valid, C, H, W]

            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")

            # 转换为 PIL 图像供 SmolVLM processor 使用
            pil_images = []
            for img_tensor in valid_images:
                # 反归一化并转换为 PIL
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # 假设使用 ImageNet 统计量归一化，反归一化
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))

            # 构建 SmolVLM 消息格式
            content = []
            for i, img in enumerate(pil_images):
                content.append({"type": "image", "image": img})

            # 添加文本提示
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})

            messages = [{"role": "user", "content": content}]

            # 使用 SmolVLM processor 处理
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # 获取编码器输出（隐藏状态）而不是生成文本
            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # 使用最后一层隐藏状态作为特征
            # 形状: [1, seq_len, hidden_size]
            hidden_states = outputs.hidden_states[-1]
            batch_features.append(hidden_states.squeeze(0))  # [seq_len, hidden_size]

        # 填充到相同长度并堆叠
        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]

        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat

        return {"vlm_features": padded_features}

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - 已预处理
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,  # [B, L] - 预分词的文本
    ) -> Dict[str, torch.Tensor]:
        """
        高效的 VLM 前向传播（训练用）- 使用完整 VLM 融合视觉和语言

        关键改进：使用完整的 VLM 前向传播（视觉编码器 + 语言模型）
        获取融合了视觉和语言信息的特征，而不是仅使用视觉编码器。

        流程：
          pixel_values → vision_encoder → image_features
                                               ↓
          input_ids → text_embeddings ─────────┤
                                               ↓
                                 [image_feats, text_embeds] (concat)
                                               ↓
                                 language_model forward
                                               ↓
                                 fused VLM features → return

        参数
        ----
        pixel_values : torch.FloatTensor
            多视角图像 [B, V, C, H, W]（已预处理）
        image_mask : torch.Tensor
            视角掩码 [B, V]
        input_ids : torch.LongTensor | None
            预分词的文本 [B, L]

        返回
        ----
        Dict[str, torch.Tensor]
            "vlm_features": [B, T_enc, D] - 融合了视觉和语言的特征
            "num_valid_views": [B] - 每个样本的有效视角数
            "num_patches_per_view": int - 每个视角的 patch 数量
        """
        # 处理输入维度
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype

        # ========== 步骤 1: 获取视觉特征 ==========
        # 展平图像: [B, V, C, H, W] -> [B*V, C, H, W]
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()

        # 获取有效图像
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]

        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        # 通过 SmolVLM 的视觉编码器（SigLIP）编码图像
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )

        # 获取图像特征并投影到语言模型空间
        image_features = vision_outputs.last_hidden_state  # [num_valid, num_patches, vision_hidden]

        # 使用 connector/projector 投影到语言模型空间
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)

        # ========== 步骤 2: 获取文本嵌入 ==========
        # Idefics3 (SmolVLM) 使用 'text_model' 而不是 'language_model'
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)  # [B, L, D]

        # ========== 步骤 3: 为每个样本构建组合序列 ==========
        # 对每个样本，拼接: [image_features_view1, ..., image_features_viewN, text_embeds]
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]

        # 重建具有 batch 结构的图像特征
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)

        # 统计每个样本的有效视角数（用于正确拼接）
        valid_per_sample = image_mask.sum(dim=1).int()  # [B]

        batch_inputs_embeds = []
        max_seq_len = 0

        for b in range(B):
            # 获取当前样本的有效图像特征
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid]  # [num_valid, num_patches, D]
            sample_image_feats = sample_image_feats.reshape(-1, hidden_size)  # [num_valid*num_patches, D]

            # 获取当前样本的文本嵌入
            sample_text_embeds = text_embeds[b]  # [L, D]

            # 拼接: [image_features, text_embeds]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # [T, D]
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])

        # ========== 步骤 4: 填充和堆叠 ==========
        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)

        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            attention_mask[b, :seq_len] = 1

        # ========== 步骤 5: 通过文本模型前向传播（Idefics3/SmolVLM）==========
        # 这会通过完整的 transformer 融合视觉和语言信息
        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 使用最后一层隐藏状态作为 VLM 特征
        # 这现在包含了融合了视觉-语言的表示
        vlm_features = lm_outputs.last_hidden_state  # [B, max_seq_len, D]

        return {
            "vlm_features": vlm_features,
            "num_valid_views": valid_per_sample,
            "num_patches_per_view": num_patches,
        }

    # ================================= 训练 =================================
    def forward(
        self,
        input_ids: torch.LongTensor,                    # [B, L] - 分词后的语言指令
        image_input: torch.FloatTensor,                 # [B, V, C, H, W]
        image_mask: torch.Tensor,                       # [B, V]
        proprio: torch.Tensor,                          # [B, dim_proprio] 或 [B, K, dim_proprio]
        action: torch.Tensor,                           # [B, T=num_actions, D=dim_action]
        wrist_prev_pixels: torch.FloatTensor | None = None,  # [B, C, H, W] 前一帧 wrist 图像
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching 训练

        时间采样策略（由 config.time_sampling 控制）：
          - "logit_normal": t = sigmoid(N(mean, std²))，来自 SD3 (arxiv:2403.03206)
          - "beta": t ~ Beta(1.5, 1) * 0.999 + 0.001（旧行为，兼容旧 checkpoint）

        流程：
          1) 时间采样：按上述策略采样 t
          2) 插值：x_t = t * noise + (1-t) * actions
          3) 目标：velocity u_t = noise - actions
          4) 模型预测 v_t，计算 MSE(v_t, u_t)

        参数
        ----
        input_ids : torch.LongTensor
            分词后的语言指令 [B, L]
        image_input : torch.FloatTensor
            多视角图像 [B, V, C, H, W]
        image_mask : torch.Tensor
            视角掩码 [B, V]
        proprio : torch.Tensor
            本体感知 [B, dim_proprio] 或 [B, K, dim_proprio]
        action : torch.Tensor
            动作序列 [B, T=num_actions, D=dim_action]
        wrist_prev_pixels : torch.FloatTensor | None
            前一帧 wrist 图像 [B, C, H, W]（用于 MotionCNN）

        返回
        ----
        Dict[str, torch.Tensor]
            "velocity_loss": 总损失
            "fm_loss": 流匹配损失（仅 ActionVAE 模式）
            "recon_loss": 重建损失（仅 ActionVAE 模式）
            "kl_loss": KL 散度损失（仅 ActionVAE 模式）
        """
        # 1. 通过 VLM 编码图像和语言
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        # 2. 计算运动激活图（帧差分 → MotionCNN）
        #    MotionCNN 输入：当前帧 - 前一帧 = 运动区域
        motion_map = None
        if self.motion_cnn is not None and wrist_prev_pixels is not None:
            # wrist 视角索引固定为 1（front=0, wrist=1, image_0=2, image_1=3）
            wrist_current = image_input[:, 1]   # [B, C, H, W]
            diff = wrist_current - wrist_prev_pixels  # 帧差分
            motion_map = self.motion_cnn(diff)        # [B, num_patches] 运动激活图

        # 3. 双流融合（可选）
        #    静态视角（front/image_0/image_1）作 Query，动态视角（wrist）作 Key/Value
        if self.dual_stream_fusion is not None:
            enc["vlm_features"] = self.dual_stream_fusion(
                enc["vlm_features"],
                enc["num_valid_views"],
                num_patches_per_view=enc.get("num_patches_per_view"),
                motion_map=motion_map,  # 注入运动偏置
            )

        B = input_ids.shape[0]
        device = input_ids.device

        # 4. 时间采样（SD3 logit-normal 默认，兼容旧 beta 模式）
        time_sampling = getattr(self.config, "time_sampling", "logit_normal")
        if time_sampling == "logit_normal":
            # Logit-Normal 采样：t = sigmoid(N(mean, std²))
            # 来自 SD3 (arxiv:2403.03206)，集中在 t=0.5 附近
            mean = getattr(self.config, "logit_normal_mean", 0.0)
            std  = getattr(self.config, "logit_normal_std",  1.0)
            z = torch.randn(B, device=device) * std + mean
            t = torch.sigmoid(z)
            # 夹到 [0.001, 0.999] 避免数值极端
            t = t.clamp(0.001, 0.999)
        else:
            # 旧 Beta(1.5, 1) 行为，兼容旧 checkpoint
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device),
                torch.tensor(1.0, device=device)
            )
            t = beta_dist.sample((B,)) * 0.999 + 0.001

        # 5. 归一化动作和本体感知
        if hasattr(self.action_space, 'normalize_action'):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, 'normalize'):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action

        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        # 6. 计算损失
        if self.use_action_vae:
            # ─── ActionVAE 隐式扩散策略（RoLD arxiv:2403.07312）───
            # 1. 将动作块编码至隐空间
            z, mu, log_var, recon = self.action_vae(action_norm, enc["vlm_features"])

            # 2. 重建损失（隐解码器监督）
            recon_loss = torch.mean(torch.square(recon - action_norm))

            # 3. KL 散度（β-VAE 正则）：-0.5 * E[1 + log σ² - μ² - σ²]
            kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

            # 4. 隐空间 Flow Matching：在 z [B,d_z] 上做 rectified flow
            noise_z = torch.randn_like(z)
            t_1d = t.unsqueeze(-1)                              # [B, 1]
            z_t = t_1d * noise_z + (1 - t_1d) * z             # [B, d_z] 插值
            u_z = noise_z - z                                   # 目标速度

            v_z = self.latent_flow_net(z_t, t, enc["vlm_features"], proprio_norm)
            fm_loss = torch.mean(torch.square(v_z - u_z))

            # 5. 总损失 = 流匹配损失 + 重建损失 + KL 散度
            total = (fm_loss
                     + self.vae_recon_weight * recon_loss
                     + self.vae_beta * kl_loss)
            return {
                "velocity_loss": total,
                "fm_loss": fm_loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
            }

        # ─── 标准 Flow Matching（动作序列空间）───
        # 1. 生成噪声
        noise = torch.randn_like(action_norm)

        # 2. 插值：x_t = t * noise + (1-t) * actions
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm

        # 3. 目标速度：u_t = noise - actions
        u_t = noise - action_norm

        # 4. 模型预测速度
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )

        # 5. 计算 MSE 损失
        return {"velocity_loss": torch.mean(torch.square(v_t - u_t))}

    # ================================= 推理 =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        wrist_prev_pixels: torch.FloatTensor | None = None,  # [B, C, H, W] 前一帧 wrist
    ) -> torch.Tensor:
        """
        Flow Matching 推理（Euler 积分）

        流程：
          1) 初始化 x_t = noise (t=1)
          2) 循环 t 从 1 到 0：
             - 模型预测速度 v_t
             - Euler 更新：x_t = x_t + dt * v_t
          3) 最终 x_0 ≈ 目标动作

        参数
        ----
        input_ids : torch.LongTensor
            分词后的语言指令 [B, L]
        image_input : torch.FloatTensor
            多视角图像 [B, V, C, H, W]
        image_mask : torch.Tensor
            视角掩码 [B, V]
        proprio : torch.Tensor
            本体感知 [B, dim_proprio]
        steps : int
            Euler 积分步数（默认 10）
        wrist_prev_pixels : torch.FloatTensor | None
            前一帧 wrist 图像 [B, C, H, W]（用于 MotionCNN）

        返回
        ----
        torch.Tensor
            预测的动作序列 [B, T, D]
        """
        self.eval()

        # 1. 通过 VLM 编码
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        # 2. 计算运动激活图
        motion_map = None
        if self.motion_cnn is not None and wrist_prev_pixels is not None:
            wrist_current = image_input[:, 1]           # [B, C, H, W]
            diff = wrist_current - wrist_prev_pixels    # 帧差分
            motion_map = self.motion_cnn(diff)          # [B, num_patches]

        # 3. 双流融合
        if self.dual_stream_fusion is not None:
            enc["vlm_features"] = self.dual_stream_fusion(
                enc["vlm_features"],
                enc["num_valid_views"],
                num_patches_per_view=enc.get("num_patches_per_view"),
                motion_map=motion_map,
            )

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        # 归一化本体感知
        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        steps = max(1, int(steps))
        dt = -1.0 / steps  # 负步长（从 t=1 到 t=0）

        if self.use_action_vae:
            # ─── ActionVAE 推理：隐空间 Euler 积分 → VAE 解码 ───
            latent_dim = self.config.latent_dim
            z_t = torch.randn(B, latent_dim, device=device, dtype=dtype)  # 初始化噪声
            t = 1.0
            while t > -dt / 2:
                t_tensor = torch.full((B,), t, device=device, dtype=dtype)
                v_z = self.latent_flow_net(z_t, t_tensor, enc["vlm_features"], proprio_norm)
                z_t = z_t + dt * v_z  # Euler 更新
                t = t + dt
            # 解码 z → 动作序列
            action = self.action_vae.decode(z_t, enc["vlm_features"])  # [B, T, D_a]
            return self.action_space.postprocess(action)

        # ─── 标准推理：动作序列空间 Euler 积分 ───
        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)  # 初始化噪声
        t = 1.0
        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
            x_t = x_t + dt * v_t  # Euler 更新
            t = t + dt

        return self.action_space.postprocess(x_t)

    # =============================== FastAPI 服务 =============================
    def _build_app(self, processor):
        """构建 FastAPI 推理服务应用"""
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # 解码图像
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload:
                        continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))

                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # 处理输入
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # 构建本体感知张量
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                # 对齐到模型设备/数据类型
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs["proprio"] = to_model(proprio.unsqueeze(0))

                # 推理
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """启动 FastAPI 推理服务"""
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)

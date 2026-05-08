"""
SmolVLM-VLA Configuration

Configuration class for SmolVLM-500M-Instruct based VLA model.
Uses SmolVLM as the vision-language backbone instead of Florence2.
"""

from transformers.configuration_utils import PretrainedConfig


class SmolVLMVLAConfig(PretrainedConfig):
    """
    Configuration class for the **SmolVLM-VLA (SmolVLM Vision-Language-Action)** model.

    This configuration defines all submodules of SmolVLM-VLA:
      - The visual-language backbone (SmolVLM-500M-Instruct)
      - The temporal/action transformer
      - The action/proprio setup
      
    Key differences from FlorenceVLA:
      - Uses SmolVLM (500M) instead of Florence2
      - Input image size: 512x512 (SmolVLM-500M uses 512x512 patches)
      - All views input to VLM directly, no aux_visual_inputs
      - Efficient model suitable for on-device applications
    """

    model_type = "smolvlm_vla"

    def __init__(
        self,
        # === SmolVLM backbone ===
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        
        # === Transformer head ===
        hidden_size: int = 768,  # Action transformer hidden size
        depth: int = 12,  # Number of transformer layers
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        max_len_seq: int = 512,  

        # === Action & proprio ===
        num_actions: int = 30,
        action_mode: str = "galaxea_joint",
        use_proprio: bool = True,
        
        # === DiT/AdaLN Mode ===
        use_adaln: bool = False,
        # 混合模式：AdaLN 注入 time+proprio，VLM token 保留在 Concat 序列里
        # 参考 DiT (arxiv:2212.09748) + π0 (arxiv:2410.24164)
        use_adaln_hybrid: bool = False,

        # === Dual-Stream Multi-View Fusion ===
        use_dual_stream: bool = False,
        dual_stream_fusion: str = "cross_attn",  # "add" | "concat_linear" | "cross_attn"

        # === Image settings ===
        image_size: int = 384,  # Can be 384 or 512
        num_views: int = 3,  # Number of camera views

        # === 辅助运动预测头（Joint Motion Image Diffusion, arxiv:2512.18007）===
        # 从 VLM 特征预测每视角全局光流向量，作为辅助监督（推理时不用）
        use_motion_head: bool = False,
        motion_out_dim: int = 6,         # num_views × 2（默认 3 视角×2方向=6）
        motion_loss_weight: float = 0.1, # 辅助损失权重 λ

        # === Flow Matching time sampling (SD3 arxiv:2403.03206) ===
        # "logit_normal": t = sigmoid(N(mean, std²))，集中在 t=0.5 附近
        # "beta": 原 Beta(1.5,1) 采样（兼容旧 checkpoint）
        time_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,

        # === View Dropout + Learned Missing Token ===
        # 训练时随机丢弃视角，让模型对缺失视角鲁棒
        use_view_dropout: bool = False,
        view_dropout_prob: float = 0.1,   # 每个视角被 dropout 的概率
        use_missing_token: bool = False,  # 用可学习 token 替代零填充

        # === Proprio 历史窗口（Diffusion Policy arxiv:2303.04137）===
        # K=1: 无历史（原行为）；K>1: 使用 GRU 编码 K 帧历史
        proprio_history_len: int = 1,

        **kwargs,
    ):
        # SmolVLM backbone path
        self.smolvlm_model_path = smolvlm_model_path
        
        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        
        # DiT/AdaLN settings
        self.use_adaln = use_adaln
        self.use_adaln_hybrid = use_adaln_hybrid

        # Dual-Stream Multi-View Fusion settings
        self.use_dual_stream = use_dual_stream
        self.dual_stream_fusion = dual_stream_fusion

        # Image settings
        self.image_size = image_size
        self.num_views = num_views

        # 辅助运动预测头
        self.use_motion_head = use_motion_head
        self.motion_out_dim = motion_out_dim
        self.motion_loss_weight = motion_loss_weight

        # Flow Matching time sampling
        self.time_sampling = time_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

        # View Dropout + Missing Token
        self.use_view_dropout = use_view_dropout
        self.view_dropout_prob = view_dropout_prob
        self.use_missing_token = use_missing_token

        # Proprio 历史窗口
        self.proprio_history_len = proprio_history_len

        # Initialize base HF config attributes
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Convert this configuration into a fully serializable dictionary.
        """
        output = super().to_dict()
        return output

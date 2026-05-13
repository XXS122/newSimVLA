"""
SmolVLM-VLA 配置

核心功能：
  - 定义 SmolVLM-VLA 模型的所有配置参数
  - 继承自 HuggingFace 的 PretrainedConfig
  - 包含所有子模块的开关和超参数

配置分类：
  1. SmolVLM 骨干网络配置
  2. Transformer 动作头配置
  3. 动作空间和本体感知配置
  4. DiT/AdaLN 模式配置
  5. 双流融合配置
  6. 图像设置配置
  7. Flow Matching 时间采样配置
  8. 视角 Dropout 配置
  9. 本体感知历史窗口配置
  10. ActionVAE 隐式扩散配置
  11. 运动引导注意力配置
"""

from transformers.configuration_utils import PretrainedConfig


class SmolVLMVLAConfig(PretrainedConfig):
    """
    SmolVLM-VLA (SmolVLM Vision-Language-Action) 模型配置

    此配置定义了 SmolVLM-VLA 的所有子模块：
      - 视觉-语言骨干网络（SmolVLM-500M-Instruct）
      - 时序/动作 Transformer
      - 动作/本体感知设置

    与 FlorenceVLA 的区别：
      - 使用 SmolVLM (500M) 而不是 Florence2
      - 输入图像尺寸：512x512（SmolVLM-500M 使用 512x512 patches）
      - 所有视角直接输入 VLM，无 aux_visual_inputs
      - 高效模型，适合设备端应用
    """

    model_type = "smolvlm_vla"

    def __init__(
        self,
        # === SmolVLM 骨干网络 ===
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",

        # === Transformer 动作头 ===
        hidden_size: int = 768,  # 动作 Transformer 隐藏层维度
        depth: int = 12,  # Transformer 层数
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        max_len_seq: int = 512,

        # === 动作和本体感知 ===
        num_actions: int = 30,
        action_mode: str = "galaxea_joint",
        use_proprio: bool = True,

        # === DiT/AdaLN 模式 ===
        use_adaln: bool = False,
        # 混合模式：AdaLN 注入 time+proprio，VLM token 保留在 Concat 序列里
        # 参考 DiT (arxiv:2212.09748) + π0 (arxiv:2410.24164)
        use_adaln_hybrid: bool = False,

        # === 双流多视角融合 ===
        use_dual_stream: bool = False,
        dual_stream_fusion: str = "cross_attn",  # "add" | "concat_linear" | "cross_attn"

        # === 图像设置 ===
        image_size: int = 384,  # 可以是 384 或 512
        num_views: int = 3,  # 视角数量

        # === Flow Matching 时间采样（SD3 arxiv:2403.03206）===
        # "logit_normal": t = sigmoid(N(mean, std²))，集中在 t=0.5 附近
        # "beta": 原 Beta(1.5,1) 采样（兼容旧 checkpoint）
        time_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,

        # === 视角 Dropout + 可学习缺失 Token ===
        # 训练时随机丢弃视角，让模型对缺失视角鲁棒
        use_view_dropout: bool = False,
        view_dropout_prob: float = 0.1,   # 每个视角被 dropout 的概率
        use_missing_token: bool = False,  # 用可学习 token 替代零填充

        # === 本体感知历史窗口（Diffusion Policy arxiv:2303.04137）===
        # K=1: 无历史（原行为）；K>1: 使用 GRU 编码 K 帧历史
        proprio_history_len: int = 1,

        # === ActionVAE 隐式扩散策略（RoLD arxiv:2403.07312）===
        # 将动作块 [B,T,D_a] 编码至紧凑隐空间 z [B,d_z]，在隐空间做 flow matching
        # 推理时：Euler 积分得到 z，再解码为动作序列
        use_action_vae: bool = False,
        latent_dim: int = 32,           # d_z：隐变量维度
        vae_beta: float = 0.001,        # β-VAE KL 散度权重
        vae_recon_weight: float = 1.0,  # 重建损失权重

        # === 运动引导跨视角注意力（Motion-Guided Cross-Attention）===
        # 帧差分图 → 轻量剪枝 CNN → 运动激活图 → 注入 cross-attention bias
        # 让静态视角自动聚焦到 wrist 图中正在运动的区域
        # 参考：DeltaCNN (CVPR 2022, arXiv:2203.03996)
        use_motion_guided_attn: bool = False,

        **kwargs,
    ):
        # SmolVLM 骨干网络路径
        self.smolvlm_model_path = smolvlm_model_path

        # Transformer 超参数
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq

        # 动作/本体感知设置
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio

        # DiT/AdaLN 设置
        self.use_adaln = use_adaln
        self.use_adaln_hybrid = use_adaln_hybrid

        # 双流多视角融合设置
        self.use_dual_stream = use_dual_stream
        self.dual_stream_fusion = dual_stream_fusion

        # 图像设置
        self.image_size = image_size
        self.num_views = num_views

        # Flow Matching 时间采样
        self.time_sampling = time_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

        # 视角 Dropout + 缺失 Token
        self.use_view_dropout = use_view_dropout
        self.view_dropout_prob = view_dropout_prob
        self.use_missing_token = use_missing_token

        # 本体感知历史窗口
        self.proprio_history_len = proprio_history_len

        # ActionVAE 隐式扩散策略
        self.use_action_vae = use_action_vae
        self.latent_dim = latent_dim
        self.vae_beta = vae_beta
        self.vae_recon_weight = vae_recon_weight

        # 运动引导跨视角注意力
        self.use_motion_guided_attn = use_motion_guided_attn

        # 初始化基础 HF 配置属性
        super().__init__(**kwargs)

    def to_dict(self):
        """
        将此配置转换为完全可序列化的字典
        """
        output = super().to_dict()
        return output

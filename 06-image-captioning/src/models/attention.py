import torch
from torch import Tensor, nn


class AdditiveAttention(nn.Module):
    """Additive attention in Show, Attend, and Tell."""

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int = 512):
        """Initialize AdditiveAttention.

        Args:
            feature_dim (int): Feature dimension.
            hidden_dim (int): Hidden dimension.
        """
        super().__init__()
        # ======== TODO: Init linear layers for attention ========
        # 1. 实现 Wh, Wf, 用于映射隐藏状态和特征
        # 2. 实现 v, 用于进行输出的线性映射
        self.Wh = nn.Linear(hidden_dim, attention_dim, bias=False)   # query 变换
        self.Wf = nn.Linear(feature_dim, attention_dim, bias=False)  # key 变换
        self.v = nn.Linear(attention_dim, 1, bias=False)             # 得分向量
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # 在像素维度上归一化
        # ========================================================

    def forward(self, hidden: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of AdditiveAttention.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            hidden (Tensor): Hidden state with shape (batch, hidden_dim).

        Returns:
            Tensor: Attention output with shape (batch, num_pixels, hidden_dim).
        """
        # ============ TODO: forward pass ============
        # e = v * tanh(Wf * features + Wh * hidden)
        # alpha = softmax(e)
        # context = sum(alpha * features)
        # raise NotImplementedError
        # 将 hidden 扩展为 (batch, 1, hidden_dim) 以匹配 features 的序列维度
        hidden_exp = hidden.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 计算注意力得分：e = v * tanh( Wf * features + Wh * hidden )
        energy = self.v(
            self.tanh(self.Wf(features) + self.Wh(hidden_exp))
        )  # (batch, num_pixels, 1)

        alpha = self.softmax(energy.squeeze(-1))  # (batch, num_pixels)

        # 上下文向量 = 加权求和
        context = torch.sum(features * alpha.unsqueeze(-1), dim=1)  # (batch, feature_dim)
        
        # ============================================
        return context, alpha

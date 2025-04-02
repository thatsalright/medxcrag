"""
门控特征融合模块，用于动态平衡全局信息与局部特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFeatureFusion(nn.Module):
    """
    门控特征融合模块，用于动态平衡全局信息与局部特征
    """
    def __init__(self, feature_dim):
        """
        初始化门控特征融合模块
        
        Args:
            feature_dim (int): 特征维度
        """
        super(GatedFeatureFusion, self).__init__()
        
        # 用于生成门控系数的线性层
        self.gate_generator = nn.Linear(feature_dim * 2, feature_dim)
        
        # 用于转换局部特征的MLP
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, global_feature, local_feature):
        """
        前向传播
        
        Args:
            global_feature (torch.Tensor): 全局特征，形状为[B, feature_dim]
            local_feature (torch.Tensor): 局部特征，形状为[B, feature_dim, H, W]或[B, seq_len, feature_dim]
            
        Returns:
            fused_feature (torch.Tensor): 融合后的特征，形状为[B, feature_dim]
        """
        # 检查局部特征的维度并进行相应处理
        if len(local_feature.shape) == 4:  # 图像特征 [B, C, H, W]
            # 对局部特征进行全局池化
            pooled_local = F.adaptive_max_pool2d(local_feature, 1).squeeze(-1).squeeze(-1)
        else:  # 文本特征 [B, seq_len, C]
            # 对局部特征进行全局池化
            pooled_local = torch.mean(local_feature, dim=1)
        
        # 拼接全局特征和池化后的局部特征
        concatenated = torch.cat([global_feature, pooled_local], dim=1)
        
        # 生成门控系数
        gate = torch.sigmoid(self.gate_generator(concatenated))
        
        # 应用MLP到局部特征
        if len(local_feature.shape) == 4:  # 图像特征
            processed_local = self.mlp(local_feature.view(local_feature.size(0), local_feature.size(1), -1).mean(dim=2))
        else:  # 文本特征
            processed_local = self.mlp(pooled_local)
        
        # 使用门控系数融合特征
        fused_feature = gate * global_feature + (1 - gate) * processed_local
        
        return fused_feature

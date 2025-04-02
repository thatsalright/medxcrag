"""
递归注意力金字塔模块，用于定位器官至病灶的渐进式视觉特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveAttentionPyramid(nn.Module):
    """
    递归注意力金字塔模块，用于定位器官至病灶的渐进式视觉特征
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3, num_time_steps=5):
        """
        初始化递归注意力金字塔模块
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): LSTM层数
            num_time_steps (int): 时间步数量
        """
        super(RecursiveAttentionPyramid, self).__init__()
        self.num_time_steps = num_time_steps
        
        # LSTM网络
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 注意力参数
        self.attention_weights = nn.Parameter(torch.randn(num_time_steps, hidden_dim))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图，形状为[B, C, H, W]
            
        Returns:
            final_feature (torch.Tensor): 加权融合后的特征图，形状为[B, hidden_dim, H, W]
            last_hidden (torch.Tensor): 最后一个时间步的隐藏状态，形状为[B, H*W, hidden_dim]
        """
        batch_size, channels, height, width = x.shape
        
        # 将特征图重塑为序列形式
        x_reshaped = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # 初始化隐藏状态
        h_t = None
        
        # 存储所有时间步的隐藏状态
        all_hidden_states = []
        
        # 递归处理多个时间步
        for t in range(self.num_time_steps):
            # 如果是第一个时间步，使用输入特征，否则使用上一时间步的输出
            if t == 0:
                input_t = x_reshaped
            else:
                input_t = all_hidden_states[-1].detach()
            
            # LSTM前向传播
            output, h_t = self.lstm(input_t, h_t)
            all_hidden_states.append(output)
        
        # 堆叠所有时间步的隐藏状态
        stacked_states = torch.stack(all_hidden_states, dim=1)  # [B, T, H*W, hidden_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(stacked_states, self.attention_weights.unsqueeze(-1))  # [B, T, H*W, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # 在时间维度上softmax
        
        # 加权融合特征
        weighted_sum = torch.sum(stacked_states * attention_weights, dim=1)  # [B, H*W, hidden_dim]
        
        # 重塑回特征图形式
        final_feature = weighted_sum.permute(0, 2, 1).view(batch_size, -1, height, width)
        
        return final_feature, weighted_sum

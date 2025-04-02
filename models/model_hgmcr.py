"""
层级门控多尺度跨模态检索框架（HGMCR）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .component.encoders import LesionAwareEncoder, TextEncoder

class HGMCR(nn.Module):
    """
    层级门控多尺度跨模态检索框架（HGMCR）
    """
    def __init__(self, config):
        """
        初始化HGMCR模型
        
        Args:
            config: 配置对象，包含模型参数
        """
        super(HGMCR, self).__init__()
        
        # 从配置中获取参数
        feature_dim = config.feature_dim
        hidden_dim = config.hidden_dim
        num_time_steps = config.num_time_steps
        temperature = config.temperature
        
        # 是否使用ViT作为视觉模型的基础
        use_vit = getattr(config, 'use_vit', True)
        
        # 病变感知编码器
        self.image_encoder = LesionAwareEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_time_steps=num_time_steps,
            use_vit=use_vit
        )
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_time_steps=num_time_steps,
            use_vit=use_vit  # 确保使用相同的MedCLIP模型基础
        )
        
        # 跨模态对齐引擎
        self.alignment_engine = CrossModalAlignmentEngine(
            feature_dim=feature_dim,
            temperature=temperature
        )
        
        self.temperature = temperature
        self.hard_negative_ratio = getattr(config, 'hard_negative_ratio', 0.3)
        
    def forward(self, images, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            images (torch.Tensor): 输入图像，形状为[B, 3, H, W]
            input_ids (torch.Tensor): 输入token ids，形状为[B, seq_len]
            attention_mask (torch.Tensor): 注意力mask，形状为[B, seq_len]
            
        Returns:
            joint_similarity (torch.Tensor): 联合相似度，形状为[B, B]
            global_similarity (torch.Tensor): 全局相似度，形状为[B, B]
            local_similarity (torch.Tensor): 局部相似度，形状为[B, B]
            attention_matrix (torch.Tensor): 注意力矩阵，形状为[B, H*W, seq_len]
        """
        batch_size = images.size(0)
        
        # 编码图像
        fused_image_features, global_image_features, local_image_features = self.image_encoder(images)
        
        # 编码文本
        fused_text_features, global_text_features, local_text_features = self.text_encoder(input_ids, attention_mask)
        
        # 保存每对图像-文本之间的相似度，确保维持梯度流
        joint_similarities = []
        global_similarities = []
        local_similarities = []
        diagonal_attentions = []
        
        # 改进: 归一化融合特征用于余弦相似度计算
        normalized_image_features = F.normalize(fused_image_features, p=2, dim=1)
        normalized_text_features = F.normalize(fused_text_features, p=2, dim=1)
        
        # 预计算全局相似度矩阵，提高效率
        global_sim_matrix = torch.mm(normalized_image_features, normalized_text_features.t())
        
        # 处理每个样本对的局部相似度和注意力
        for i in range(batch_size):
            row_joint = []
            row_local = []
            
            for j in range(batch_size):
                # 获取全局相似度
                global_sim = global_sim_matrix[i, j].unsqueeze(0)
                
                # 计算局部相似度和注意力矩阵
                # 重塑局部特征
                visual_local_reshaped = local_image_features[i].view(1, local_image_features.size(1), -1).transpose(1, 2)
                
                # 计算层级注意力和局部相似度
                local_sim, attention_matrix = self.alignment_engine.compute_hierarchical_attention(
                    visual_local_reshaped, 
                    local_text_features[j].unsqueeze(0)
                )
                
                # 计算联合相似度
                joint_sim = global_sim + 0.5 * local_sim
                
                row_joint.append(joint_sim)
                row_local.append(local_sim)
                
                # 只保存对角线上的注意力矩阵
                if i == j:
                    diagonal_attentions.append(attention_matrix)
            
            # 堆叠每行的相似度
            joint_similarities.append(torch.cat(row_joint))
            local_similarities.append(torch.cat(row_local))
        
        # 堆叠形成矩阵
        joint_sim_matrix = torch.stack(joint_similarities)
        local_sim_matrix = torch.stack(local_similarities)
        
        # 堆叠所有注意力矩阵
        attention_matrix = torch.cat(diagonal_attentions, dim=0) if diagonal_attentions else None
        
        return joint_sim_matrix, global_sim_matrix, local_sim_matrix, attention_matrix
    
    def generate_hard_negatives(self, batch_labels=None, similarity_matrix=None, ratio=None):
        """
        硬负例挖掘策略，基于相似度选择高混淆度负样本
        
        Args:
            batch_labels (torch.Tensor, optional): 批次中每个样本的疾病标签 [B, num_classes]
                                            如果为None，则直接使用相似度选择
            similarity_matrix (torch.Tensor): 相似度矩阵 [B, B]
            ratio (float, optional): 采样比例，默认使用self.hard_negative_ratio
            
        Returns:
            hard_negative_mask (torch.Tensor): 硬负例mask，形状为[B, B]
            count (int): 硬负例数量
        """
        if ratio is None:
            ratio = self.hard_negative_ratio
            
        batch_size = similarity_matrix.size(0)
        
        # 创建负样本mask（非对角线元素）
        negative_mask = 1 - torch.eye(batch_size, device=similarity_matrix.device)
        
        # 检查是否所有标签都是零，这表示没有标签信息
        no_labels = batch_labels is None or torch.sum(batch_labels) == 0
        
        # 训练初期使用简单的负样本策略
        curr_epoch = getattr(self, 'current_epoch', 0)
        warm_up_epochs = getattr(self, 'warm_up_epochs', 3)
        in_warm_up = curr_epoch < warm_up_epochs
        
        if no_labels or in_warm_up:
            # 无标签或训练初期，使用随机负样本而不是基于相似度的硬负例
            # if in_warm_up:
            #     print(f"在预热阶段 (epoch {curr_epoch}/{warm_up_epochs})，使用随机负样本")
                
            # 直接随机选择一部分负样本，避免使用相似度（可能不可靠）
            k = max(1, int(batch_size * ratio))  # 每个样本选择的负样本数量
            hard_negative_mask = torch.zeros_like(similarity_matrix)
            
            for i in range(batch_size):
                # 获取所有可能的负样本索引（非对角线）
                neg_indices = torch.nonzero(negative_mask[i]).squeeze(-1)
                
                # 如果负样本数量足够，随机选择k个
                if len(neg_indices) > k:
                    # 随机选择k个负样本
                    perm = torch.randperm(len(neg_indices))
                    selected_indices = neg_indices[perm[:k]]
                    hard_negative_mask[i, selected_indices] = 1
                else:
                    # 否则使用所有负样本
                    hard_negative_mask[i, neg_indices] = 1
        else:
            # 如果有疾病标签，使用疾病标签来选择共享特征的负样本
            batch_labels_bool = batch_labels > 0
            label_sim = torch.mm(batch_labels_bool.float(), batch_labels_bool.float().t()) > 0
            shared_disease_negatives = label_sim * negative_mask
            
            # 如果没有共享疾病的负样本，则使用所有负样本
            if shared_disease_negatives.sum() == 0:
                shared_disease_negatives = negative_mask
                
            # 根据相似度对负样本排序
            similarity_with_mask = similarity_matrix * shared_disease_negatives
            
            # 为每个样本选择前K个最相似的负样本
            k = max(1, int(batch_size * ratio))  # 确保至少选择1个负样本
            
            # 创建硬负例mask
            hard_negative_mask = torch.zeros_like(similarity_matrix)
            
            # 对每个样本，选择K个最高相似度的负样本
            for i in range(batch_size):
                # 获取当前样本与所有负样本的相似度
                sim_scores = similarity_with_mask[i]
                
                # 过滤掉正样本和无效样本
                valid_indices = torch.where(sim_scores > -float('inf'))[0]
                
                if len(valid_indices) > 0:
                    # 对有效负样本按相似度排序
                    valid_scores = sim_scores[valid_indices]
                    _, sorted_idx = torch.sort(valid_scores, descending=True)
                    
                    # 选择前K个
                    selected_idx = valid_indices[sorted_idx[:min(k, len(valid_indices))]]
                    hard_negative_mask[i, selected_idx] = 1
        
        # 计算硬负例总数
        count = hard_negative_mask.sum().item()
        
        return hard_negative_mask, count
    
    def encode_image(self, images):
        """
        编码图像，用于推理
        
        Args:
            images (torch.Tensor): 输入图像，形状为[B, 3, H, W]
            
        Returns:
            fused_features (torch.Tensor): 融合特征，形状为[B, feature_dim]
            global_features (torch.Tensor): 全局特征，形状为[B, feature_dim]
            local_features (torch.Tensor): 局部特征，形状为[B, feature_dim, H/16, W/16]
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            return self.image_encoder(images)
    
    def encode_text(self, input_ids, attention_mask):
        """
        编码文本，用于推理
        
        Args:
            input_ids (torch.Tensor): 输入token ids，形状为[B, seq_len]
            attention_mask (torch.Tensor): 注意力mask，形状为[B, seq_len]
            
        Returns:
            fused_features (torch.Tensor): 融合特征，形状为[B, feature_dim]
            global_features (torch.Tensor): 全局特征，形状为[B, feature_dim]
            local_features (torch.Tensor): 局部特征，形状为[B, seq_len, feature_dim]
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            return self.text_encoder(input_ids, attention_mask)


class CrossModalAlignmentEngine(nn.Module):
    """
    跨模态对齐引擎，负责建立视觉与文本特征间的语义关联
    """
    def __init__(self, feature_dim=512, temperature=0.05):
        """
        初始化跨模态对齐引擎
        
        Args:
            feature_dim (int): 特征维度
            temperature (float): 温度参数
        """
        super(CrossModalAlignmentEngine, self).__init__()
        
        self.temperature = temperature
        
        # 共享参数投影矩阵
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        
        # 门控融合参数
        self.gamma_generator = nn.Linear(feature_dim * 2, 1)
        
        # 初始化参数 - Xavier初始化
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.gamma_generator.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.zeros_(self.key_projection.bias)
        nn.init.zeros_(self.gamma_generator.bias)
        
    def compute_similarity(self, image_features, text_features):
        """
        计算全局相似度
        
        Args:
            image_features (torch.Tensor): 图像特征，形状为[B, feature_dim]
            text_features (torch.Tensor): 文本特征，形状为[B, feature_dim]
            
        Returns:
            global_similarity (torch.Tensor): 全局相似度，形状为[B]
        """
        # 计算全局相似度（余弦相似度）
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        global_similarity = torch.sum(image_features * text_features, dim=1)
        
        return global_similarity
        
    def compute_hierarchical_attention(self, visual_features, text_features):
        """
        计算层级交叉注意力
        
        Args:
            visual_features (torch.Tensor): 视觉特征，形状为[B, H*W, feature_dim]
            text_features (torch.Tensor): 文本特征，形状为[B, seq_len, feature_dim]
            
        Returns:
            local_similarity (torch.Tensor): 局部相似度，形状为[B]
            final_attention (torch.Tensor): 最终注意力矩阵，形状为[B, H*W, seq_len]
        """
        # 计算层级交叉注意力矩阵
        batch_size = visual_features.shape[0]
        
        # 投影特征
        projected_visual = self.query_projection(visual_features)  # [B, H*W, D]
        projected_text = self.key_projection(text_features)  # [B, seq_len, D]
        
        # 计算注意力分数
        attention_scores = torch.bmm(projected_visual, projected_text.transpose(1, 2))  # [B, H*W, seq_len]
        attention_scores = attention_scores / math.sqrt(projected_visual.size(-1))
        
        # Softmax归一化
        normalized_attention = F.softmax(attention_scores, dim=2)  # 在文本维度上归一化
        
        # 获取最大注意力值
        max_attention, _ = torch.max(normalized_attention, dim=2, keepdim=True)  # [B, H*W, 1]
        
        # 计算门控系数γ
        # 复制max_attention使其具有与visual_features相同的特征维度
        expanded_attn = max_attention.expand(-1, -1, visual_features.size(-1))
        
        # 拼接visual_features和扩展后的注意力
        concatenated = torch.cat([visual_features, expanded_attn], dim=2)
        gamma = torch.sigmoid(self.gamma_generator(concatenated))  # [B, H*W, 1]
        
        # 生成最终的对齐权重
        final_attention = gamma * normalized_attention
        
        # 计算局部相似度（注意力矩阵的平均值）
        local_similarity = torch.mean(final_attention, dim=(1, 2))
        
        return local_similarity, final_attention
        
    def forward(self, image_features, text_features, visual_local_features, text_local_features):
        """
        前向传播
        
        Args:
            image_features (torch.Tensor): 图像特征，形状为[B, feature_dim]
            text_features (torch.Tensor): 文本特征，形状为[B, feature_dim]
            visual_local_features (torch.Tensor): 视觉局部特征，形状为[B, feature_dim, H, W]
            text_local_features (torch.Tensor): 文本局部特征，形状为[B, seq_len, feature_dim]
            
        Returns:
            joint_similarity (torch.Tensor): 联合相似度，形状为[B]
            global_similarity (torch.Tensor): 全局相似度，形状为[B]
            local_similarity (torch.Tensor): 局部相似度，形状为[B]
            attention_matrix (torch.Tensor): 注意力矩阵，形状为[B, H*W, seq_len]
        """
        # 计算全局相似度
        global_similarity = self.compute_similarity(image_features, text_features)
        
        # 准备局部特征用于层级注意力匹配
        batch_size = image_features.shape[0]
        
        # 重塑局部特征
        visual_local_reshaped = visual_local_features.view(batch_size, visual_local_features.size(1), -1).transpose(1, 2)
        
        # 计算层级注意力和局部相似度
        local_similarity, attention_matrix = self.compute_hierarchical_attention(
            visual_local_reshaped, 
            text_local_features
        )
        
        # 计算联合相似度
        joint_similarity = global_similarity + 0.5 * local_similarity
        
        return joint_similarity, global_similarity, local_similarity, attention_matrix
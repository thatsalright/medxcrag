"""
病变感知编码器和文本编码器模块 - 基于MedCLIP实现
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging
from transformers import ViTModel, ViTConfig
from .attention import RecursiveAttentionPyramid
from .fusion import GatedFeatureFusion

logger = logging.getLogger(__name__)

class LesionAwareEncoder(nn.Module):
    """
    病变感知编码器模块，从医学影像中提取器官至病灶的多尺度视觉表征
    基于MedCLIP视觉编码器实现
    """
    def __init__(self, feature_dim=512, hidden_dim=512, num_time_steps=5, use_vit=True):
        """
        初始化病变感知编码器
        
        Args:
            feature_dim (int): 特征维度
            hidden_dim (int): 隐藏层维度
            num_time_steps (int): 递归注意力金字塔的时间步数
            use_vit (bool): 是否使用ViT作为视觉模型(True使用ViT, False使用ResNet50) - 已不再使用，保留参数兼容性
        """
        super(LesionAwareEncoder, self).__init__()
        
        # 加载标准ViT模型替代MedCLIP
        self.model_name = 'google/vit-base-patch16-224'  # 标准ViT基础模型
        self.vit_model = ViTModel.from_pretrained(self.model_name)
        
        # 设置ViT特征维度 (标准ViT为768)
        self.vit_dim = 768  # 标准ViT的特征维度
        
        # 添加投影层，将ViT特征转换为所需的特征维度
        self.projection = nn.Linear(self.vit_dim, feature_dim)
        
        # 加深后的VFPN网络 - 输入维度从medclip_dim变为vit_dim
        self.vfpn = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(self.vit_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            # 第二个卷积块
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            # 第三个卷积块
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            # 第四个卷积块
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            # 第五个卷积块
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        # 递归注意力金字塔
        self.recursive_attention = RecursiveAttentionPyramid(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_time_steps=num_time_steps
        )
        
        # 门控特征融合模块
        self.gated_fusion = GatedFeatureFusion(feature_dim)
        
        # ViT的特征网格大小，对于基础ViT模型的14x14 (224/16=14)
        self.feature_size = 14
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状为[B, 3, H, W]
            
        Returns:
            fused_features (torch.Tensor): 融合后的特征，形状为[B, feature_dim]
            global_features (torch.Tensor): 全局特征，形状为[B, feature_dim]
            final_local_features (torch.Tensor): 局部特征，形状为[B, feature_dim, H/16, W/16]
        """
        batch_size = x.size(0)
        
        # 通过ViT模型提取特征

        outputs = self.vit_model(x)
            
        # 获取全局特征 (CLS token)
        global_features_raw = outputs.pooler_output  # [batch_size, vit_dim]
        
        # 获取所有patch的特征序列，去掉CLS token
        patch_features = outputs.last_hidden_state[:, 1:, :]  # [batch_size, 196, vit_dim]
        
        # 投影全局特征到目标维度
        global_features = self.projection(global_features_raw)
        
        # 将特征序列重塑为二维网格
        # 对于224x224图像，patch大小为16，得到14x14的特征网格
        tokens_reshaped = patch_features.reshape(batch_size, self.feature_size, self.feature_size, self.vit_dim)
        
        # 将特征图转换为BCHW格式用于卷积
        x_intermediate = tokens_reshaped.permute(0, 3, 1, 2)  # [B, vit_dim, 14, 14]
        
        # 通过VFPN提取高分辨率强语义的特征图
        fpn_features = self.vfpn(x_intermediate)
        
        # 通过递归注意力金字塔处理
        final_local_features, local_hidden_states = self.recursive_attention(fpn_features)
        
        # 门控特征融合
        fused_features = self.gated_fusion(global_features, final_local_features)
        
        return fused_features, global_features, final_local_features


class TextEncoder(nn.Module):
    """
    Text encoder module using microsoft/BiomedVLP-CXR-BERT-specialized
    """
    def __init__(self, feature_dim=512, hidden_dim=512, num_time_steps=5, use_vit=True):
        """
        Initialize text encoder
        
        Args:
            feature_dim (int): Feature dimension
            hidden_dim (int): Hidden layer dimension
            num_time_steps (int): Number of time steps for recursive attention pyramid
            use_vit (bool): Whether to use ViT as vision model (not used with BiomedVLP)
        """
        super(TextEncoder, self).__init__()
        
        # Initialize tokenizer for BiomedVLP-CXR-BERT
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
        
        # 加载BERT文本编码器
        try:
            # Try to load the BiomedVLP-CXR-BERT model
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
            logger.info("成功加载BiomedVLP-CXR-BERT文本编码器")
        except Exception as e:
            # If loading fails, create a placeholder for the encoder
            logger.error(f"Warning: Could not load BiomedVLP-CXR-BERT model: {e}")
            self.text_encoder = None
        
        # Set vocab size based on tokenizer
        self.vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 30522
        
        # Set feature dimension to match BERT output
        self.bert_dim = 768  # BiomedVLP-CXR-BERT has 768 hidden dimension
        
        # Add BERT output projection layer
        self.bert_projection = nn.Linear(self.bert_dim, hidden_dim)
        
        # Stacked LSTM network
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, batch_first=True)
        
        # Attention parameters
        self.num_time_steps = num_time_steps
        self.attention_weights = nn.Parameter(torch.randn(self.num_time_steps, hidden_dim))
        
        # Gated feature fusion module
        self.gated_fusion = GatedFeatureFusion(feature_dim)
        
        # Semantic compression MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def validate_input_ids(self, input_ids):
        """
        Validate input IDs to be within vocabulary range
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            
        Returns:
            torch.Tensor: Validated token IDs
        """
        try:
            # Deep copy input to avoid modifying original
            corrected_ids = input_ids.clone()
            
            # Create masks for invalid IDs
            too_large_mask = corrected_ids >= self.vocab_size
            negative_mask = corrected_ids < 0
            invalid_mask = too_large_mask | negative_mask
            
            # Replace invalid IDs with known tokens
            if invalid_mask.any():
                # Replace large IDs with [UNK] token
                unk_token_id = self.tokenizer.unk_token_id if hasattr(self.tokenizer, 'unk_token_id') else 100
                corrected_ids[too_large_mask] = unk_token_id
                
                # Replace negative IDs with [PAD] token
                pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
                corrected_ids[negative_mask] = pad_token_id
            
            return corrected_ids
        except Exception as e:
            # Fallback to simple clamping in case of error
            return torch.clamp(input_ids, min=0, max=self.vocab_size-1)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass - 使用实际的文本特征而不是随机生成的特征
        
        Args:
            input_ids (torch.Tensor): Input token IDs [B, seq_len]
            attention_mask (torch.Tensor): Attention mask [B, seq_len]
            
        Returns:
            fused_text_feature (torch.Tensor): Fused text features [B, feature_dim]
            global_text_feature (torch.Tensor): Global text features [B, feature_dim]
            weighted_sum (torch.Tensor): Weighted sum of local features [B, seq_len, hidden_dim]
        """
        # Get dimensions and device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # 修复：使用实际的BERT模型提取特征而不是随机生成
        if self.text_encoder is not None:
            try:
                # 验证input_ids以确保它们在有效范围内
                validated_input_ids = self.validate_input_ids(input_ids)
                
                # 使用BERT模型获取真实特征

                bert_outputs = self.text_encoder(validated_input_ids, attention_mask=attention_mask)
                bert_features = bert_outputs.last_hidden_state
                    
                logger.debug(f"成功从BERT提取特征，形状: {bert_features.shape}")
            except Exception as e:
                logger.error(f"从BERT提取特征时出错: {e}，回退到随机特征")
                # 仅在出错时回退到随机特征
                bert_features = torch.randn(
                    (batch_size, seq_len, self.bert_dim), 
                    device=device, 
                    dtype=torch.float32
                ) * 0.02

        # 应用注意力掩码，确保填充标记的特征为零
        if attention_mask is not None:
            # 扩展attention_mask以匹配特征维度
            expanded_mask = attention_mask.unsqueeze(-1).expand(-1, -1, self.bert_dim)
            bert_features = bert_features * expanded_mask
        
        # Project BERT features to hidden dimension
        sequence_features = self.bert_projection(bert_features)
        
        # Get global text feature ([CLS] token)
        global_text_feature = sequence_features[:, 0, :]
        
        # Initialize hidden state
        h_t = None
        
        # Store all time steps' hidden states
        all_hidden_states = []
        
        # Recursive processing over multiple time steps
        for t in range(self.num_time_steps):
            # Use input features for first step, otherwise use previous output
            if t == 0:
                input_t = sequence_features
            else:
                input_t = all_hidden_states[-1].detach()
            
            # Clean any NaN/Inf values
            input_t = torch.nan_to_num(input_t, nan=0.0, posinf=0.0, neginf=0.0)
            
            # LSTM forward pass
            output, h_t = self.lstm(input_t, h_t)
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
                
            all_hidden_states.append(output)
        
        # Stack all time steps' hidden states
        stacked_states = torch.stack(all_hidden_states, dim=1)  # [B, T, seq_len, hidden_dim]
        
        # Calculate attention weights
        attention_scores = torch.matmul(stacked_states, self.attention_weights.unsqueeze(-1))  # [B, T, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # Softmax over time dimension
        
        # Weighted feature fusion
        weighted_sum = torch.sum(stacked_states * attention_weights, dim=1)  # [B, seq_len, hidden_dim]
        
        # Global average pooling
        pooled_feature = torch.mean(weighted_sum, dim=1)  # [B, hidden_dim]
        
        # Semantic compression
        compressed_feature = self.mlp(pooled_feature)
        
        # Gated feature fusion
        fused_text_feature = self.gated_fusion(global_text_feature, weighted_sum)
        
        return fused_text_feature, global_text_feature, weighted_sum
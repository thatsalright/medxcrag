

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
import time
from utils.metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)

class Evaluator:

    def __init__(self, model, val_loader, device):

        self.model = model
        self.val_loader = val_loader
        self.device = device
        
        # 缓存验证集特征的标志
        self.cached_features = None
        
        # 评估选项
        self.max_eval_samples = 1000  # 最大评估样本数，如果数据集太大可以限制样本数量
        self.use_fp16 = True  # 使用半精度加速
        self.similarity_batch_size = 128  # 相似度计算的批大小
    
    def extract_features(self, model_to_eval, limit_samples=None):
        # 检查是否可以使用缓存
        if self.cached_features is not None:
            logger.info("Using cached features")
            return self.cached_features
            
        # 保存所有图像和文本特征
        all_image_features = []
        all_text_features = []
        all_labels = []
        
        # 自动混合精度
        amp_enabled = self.use_fp16 and hasattr(torch.cuda, 'amp') and self.device.type == 'cuda'
        amp_dtype = torch.float16 if amp_enabled else torch.float32
        
        # 设置上下文管理器
        context_manager = torch.cuda.amp.autocast() if amp_enabled else torch.no_grad()
        
        # 计时
        start_time = time.time()
        
        # 计算需要处理的批次数
        total_batches = len(self.val_loader)
        if limit_samples is not None:
            samples_seen = 0
            batches_to_process = total_batches
            for i, (images, _, _, _, labels) in enumerate(self.val_loader):
                samples_seen += images.size(0)
                if samples_seen >= limit_samples:
                    batches_to_process = i + 1
                    break
        else:
            batches_to_process = total_batches
            
        logger.info(f"Extracting features from {batches_to_process} batches...")
        
        # 提取特征
        with context_manager:
            for batch_idx, (images, _, input_ids, attention_mask, labels) in enumerate(tqdm(self.val_loader, total=batches_to_process, desc="提取特征")):
                # 检查是否达到样本限制
                if limit_samples is not None and sum(len(f) for f in all_image_features) >= limit_samples:
                    break
                    
                # 移动到设备
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # 提取特征
                fused_image_features, _, _ = model_to_eval.encode_image(images)
                fused_text_features, _, _ = model_to_eval.encode_text(input_ids, attention_mask)
                
                # 归一化特征以便于后续计算余弦相似度
                fused_image_features = F.normalize(fused_image_features, p=2, dim=1)
                fused_text_features = F.normalize(fused_text_features, p=2, dim=1)
                
                # 添加到列表
                all_image_features.append(fused_image_features.cpu())
                all_text_features.append(fused_text_features.cpu())
                all_labels.append(labels)
                
                # 打印进度
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {batch_idx}/{batches_to_process} batches in {elapsed:.2f}s")
        
        # 拼接所有特征和标签
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 限制样本数量
        if limit_samples is not None and len(all_image_features) > limit_samples:
            all_image_features = all_image_features[:limit_samples]
            all_text_features = all_text_features[:limit_samples]
            all_labels = all_labels[:limit_samples]
        
        logger.info(f"Extracted features for {len(all_image_features)} samples in {time.time() - start_time:.2f}s")
        
        # 缓存特征
        self.cached_features = (all_image_features, all_text_features, all_labels)
        
        return all_image_features, all_text_features, all_labels
    
    def compute_similarity_matrix_fast(self, image_features, text_features):
        num_samples = len(image_features)
        logger.info(f"Computing similarity matrix for {num_samples} samples...")
        
        # 使用大型批处理和并行计算
        similarity_matrix = torch.zeros(num_samples, num_samples)
        batch_size = self.similarity_batch_size
        
        # 混合精度加速
        amp_enabled = self.use_fp16 and hasattr(torch.cuda, 'amp') and self.device.type == 'cuda'
        
        # 计时
        start_time = time.time()
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.cuda.amp.autocast() if amp_enabled else torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="计算相似度矩阵"):
                # 获取当前批次的结束索引
                i_end = min(i + batch_size, num_samples)
                
                # 将当前批次的图像特征移动到GPU
                img_batch = image_features[i:i_end].to(self.device)
                
                # 高效地计算与所有文本的相似度
                # 使用矩阵乘法一次计算当前批次与所有文本的相似度
                sim_chunk = torch.mm(img_batch, text_features.to(self.device).t())
                
                # 将结果移回CPU以节省GPU内存
                similarity_matrix[i:i_end] = sim_chunk.cpu()
                
                # 每处理10个批次打印一次进度
                if (i // batch_size) % 10 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    progress = i / num_samples
                    estimated_total = elapsed / progress if progress > 0 else 0
                    remaining = estimated_total - elapsed
                    logger.info(f"Processed {i}/{num_samples} samples ({progress*100:.1f}%) - "
                               f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
        
        logger.info(f"Similarity matrix computation completed in {time.time() - start_time:.2f}s")
        return similarity_matrix
        
    def evaluate(self):

        # 每次评估开始时清除缓存，确保使用最新模型权重
        self.clear_cache()
        
        # 开始计时
        eval_start_time = time.time()
        
        # 获取未包装的模型用于评估
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_to_eval = self.model.module
        else:
            model_to_eval = self.model
            
        # 设置为评估模式
        model_to_eval.eval()
        
        # 提取特征
        image_features, text_features, labels = self.extract_features(
            model_to_eval, 
            limit_samples=self.max_eval_samples
        )
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix_fast(image_features, text_features)
        
        # 计算指标
        logger.info("Computing retrieval metrics...")
        metrics = compute_retrieval_metrics(similarity_matrix)
        
        # 输出结果
        total_time = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {total_time:.2f}s")
        logger.info("Image to Text:")
        logger.info(f"R@1: {metrics['i2t_r1']:.2f}, R@5: {metrics['i2t_r5']:.2f}, "
                f"R@10: {metrics['i2t_r10']:.2f}, Median Rank: {metrics['i2t_medr']}")
        logger.info("Text to Image:")
        logger.info(f"R@1: {metrics['t2i_r1']:.2f}, R@5: {metrics['t2i_r5']:.2f}, "
                f"R@10: {metrics['t2i_r10']:.2f}, Median Rank: {metrics['t2i_medr']}")
        
        return metrics
        
    def evaluate_with_hard_negatives(self):
        # 开始计时
        eval_start_time = time.time()
        
        # 获取未包装的模型用于评估
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_to_eval = self.model.module
        else:
            model_to_eval = self.model
            
        # 设置为评估模式
        model_to_eval.eval()
        
        # 提取特征
        image_features, text_features, labels = self.extract_features(
            model_to_eval, 
            limit_samples=self.max_eval_samples
        )
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix_fast(image_features, text_features)
        
        # 生成硬负例掩码
        logger.info("Generating hard negative masks...")
        similarity_matrix_gpu = similarity_matrix.to(self.device)
        hard_negative_mask, count = model_to_eval.generate_hard_negatives(
            labels.to(self.device), 
            similarity_matrix_gpu
        )
        logger.info(f"Generated {count} hard negative pairs")
        
        # 使用硬负例计算指标
        logger.info("Computing metrics with hard negatives...")
        metrics = compute_retrieval_metrics(
            similarity_matrix.cpu(), 
            hard_negative_mask.cpu() if hard_negative_mask is not None else None
        )
        
        # 输出结果
        total_time = time.time() - eval_start_time
        logger.info(f"Hard negative evaluation completed in {total_time:.2f}s")
        logger.info("Image to Text (with hard negatives):")
        logger.info(f"R@1: {metrics['i2t_r1']:.2f}, R@5: {metrics['i2t_r5']:.2f}, "
                   f"R@10: {metrics['i2t_r10']:.2f}, Median Rank: {metrics['i2t_medr']}")
        logger.info("Text to Image (with hard negatives):")
        logger.info(f"R@1: {metrics['t2i_r1']:.2f}, R@5: {metrics['t2i_r5']:.2f}, "
                   f"R@10: {metrics['t2i_r10']:.2f}, Median Rank: {metrics['t2i_medr']}")
        
        return metrics
        
    def clear_cache(self):
        self.cached_features = None
        torch.cuda.empty_cache()  # 清理GPU内存
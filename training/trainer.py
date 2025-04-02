

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import logging
import time
import os
import numpy as np

from .loss import compute_contrastive_loss, compute_bidirectional_contrastive_loss
from .evaluator import Evaluator
from utils.checkpoint import save_checkpoint
from utils.logger import TensorboardLogger

logger = logging.getLogger(__name__)

def barrier_with_timeout(timeout=60):

    try:
        work = dist.barrier(async_op=True)
        return work.wait(timeout=timeout)
    except Exception as e:
        logger.warning(f"Process {dist.get_rank()} barrier timed out: {e}")
        return False

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, 
                start_epoch=0, best_metrics=None, is_distributed=False, 
                train_sampler=None, local_rank=0, is_main_process=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.start_epoch = start_epoch
        
        # 分布式训练相关
        self.is_distributed = is_distributed
        self.train_sampler = train_sampler
        self.local_rank = local_rank
        self.is_main_process = is_main_process
        
        # 创建优化器
        # 如果是分布式训练，优化器需要使用原始模型的参数
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model_for_optim = model.module
        else:
            model_for_optim = model
        
        # 预训练模型学习率因子 - 使用较小的学习率微调预训练模型
        pretrained_lr_factor = getattr(config, 'pretrained_lr_factor', 0.1)
        
        # 分离预训练模型参数和新添加的参数
        pretrained_params = []
        new_params = []
        
        # 收集ViT参数
        if hasattr(model_for_optim.image_encoder, 'vit_model'):
            pretrained_params.extend(model_for_optim.image_encoder.vit_model.parameters())
        
        # 收集BERT参数
        if hasattr(model_for_optim.text_encoder, 'text_encoder') and model_for_optim.text_encoder.text_encoder is not None:
            pretrained_params.extend(model_for_optim.text_encoder.text_encoder.parameters())
        
        # 收集其他所有参数
        for name, param in model_for_optim.named_parameters():
            if param.requires_grad and not any(p is param for p in pretrained_params):
                new_params.append(param)
        
        # 创建带有不同学习率的参数组
        param_groups = [
            {'params': pretrained_params, 'lr': config.learning_rate * pretrained_lr_factor},  # 预训练模型使用较小的学习率
            {'params': new_params, 'lr': config.learning_rate}                                 # 新添加的层使用正常学习率
        ]
        
        if is_main_process:
            logger.info(f"启用预训练模型训练 (ViT和BERT)，学习率: {config.learning_rate * pretrained_lr_factor:.6f}")
            logger.info(f"新增层学习率: {config.learning_rate:.6f}")
            logger.info(f"预训练参数数量: {sum(p.numel() for p in pretrained_params):,}")
            logger.info(f"新增参数数量: {sum(p.numel() for p in new_params):,}")
            
        # 创建优化器
        self.optimizer = optim.Adam(
            param_groups,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.lr_scheduler == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma
            )
        elif config.lr_scheduler == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs
            )
        else:
            self.lr_scheduler = None
        
        # 只在主进程上创建评估器和日志记录器
        if self.is_main_process:
            # 评估器
            self.evaluator = Evaluator(model, val_loader, device)
            
            # Tensorboard日志
            self.tb_logger = TensorboardLogger(config.log_dir)
            
            # 创建输出目录
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 训练指标
        self.train_losses = []
        self.val_metrics_history = {
            'i2t_r1': [], 'i2t_r5': [], 'i2t_r10': [], 'i2t_medr': [],
            't2i_r1': [], 't2i_r5': [], 't2i_r10': [], 't2i_medr': []
        }
        
        # 设置权重系数，用于综合评估不同的R@k指标
        self.i2t_weights = {
            'i2t_r1': getattr(config, 'i2t_r1_weight', 0.5),
            'i2t_r5': getattr(config, 'i2t_r5_weight', 0.3),
            'i2t_r10': getattr(config, 'i2t_r10_weight', 0.2)
        }
        
        # 只在主进程上处理最佳指标
        if self.is_main_process:
            # 如果有最佳指标，则使用它们，否则初始化
            if best_metrics:
                # 计算综合得分
                self.best_i2t_score = self._calculate_i2t_score(best_metrics)
                
                # 填充指标历史（如果从检查点恢复训练）
                if isinstance(best_metrics.get('i2t_r1'), (list, tuple)):
                    # 如果best_metrics中存储了历史数据，直接使用
                    for key in self.val_metrics_history:
                        if key in best_metrics and isinstance(best_metrics[key], (list, tuple)):
                            self.val_metrics_history[key] = list(best_metrics[key])
                else:
                    # 如果best_metrics只有单个值，添加到历史中
                    for key in best_metrics:
                        if key in self.val_metrics_history:
                            self.val_metrics_history[key].append(best_metrics[key])
            else:
                self.best_i2t_score = 0.0  # 初始化综合得分
            
            # 早停相关参数
            self.patience = getattr(config, 'early_stop_patience', 5)  # 默认patience为5
            self.counter = 0  # 早停计数器
            self.best_epoch = self.start_epoch  # 最佳模型的epoch
            
            # 验证日志记录器是否正常工作
            logger.info("=" * 80)
            logger.info("训练器初始化完成，日志记录器测试消息")
            logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            if self.is_distributed:
                logger.info(f"分布式训练: 使用 {dist.get_world_size()} 个GPU")
            logger.info("=" * 80)
        else:
            # 非主进程初始化一些需要的变量
            self.best_i2t_score = 0.0
            self.counter = 0
            self.best_epoch = self.start_epoch
    
    def _calculate_i2t_score(self, metrics):
        score = 0.0
        for metric, weight in self.i2t_weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        return score
        
    def print_metrics_table(self, metrics, epoch):

        if not self.is_main_process:
            return
            
        logger.info("=" * 80)
        logger.info(f"Evaluation Metrics - Epoch {epoch}")
        logger.info("-" * 80)
        logger.info(f"{'Metric':<15} | {'Value':>10} | {'Best':>10} | {'Improvement':>10} | {'Weight':>10}")
        logger.info("-" * 80)
        
        # 计算当前综合分数
        current_i2t_score = self._calculate_i2t_score(metrics)
        
        # i2t metrics with weights
        for metric in ['i2t_r1', 'i2t_r5', 'i2t_r10']:
            value = metrics[metric]
            best_value = max([hist_val for hist_val in self.val_metrics_history[metric]] + [0]) if self.val_metrics_history[metric] else 0
            improvement = value - best_value
            weight = self.i2t_weights.get(metric, 0.0)
            
            logger.info(f"{metric:<15} | {value:>10.2f} | {best_value:>10.2f} | {improvement:>+10.2f} | {weight:>10.2f}")
        
        # Combined score
        logger.info("-" * 80)
        logger.info(f"{'I2T Combined':<15} | {current_i2t_score:>10.2f} | {self.best_i2t_score:>10.2f} | {(current_i2t_score - self.best_i2t_score):>+10.2f} | {'1.0':>10}")
        
        # Other metrics without weights
        logger.info("-" * 80)
        for metric in ['i2t_medr', 't2i_r1', 't2i_r5', 't2i_r10', 't2i_medr']:
            value = metrics[metric]
            
            # For median rank, lower is better
            if metric in ['i2t_medr', 't2i_medr']:
                best_value = min([hist_val for hist_val in self.val_metrics_history[metric]] + [float('inf')]) if self.val_metrics_history[metric] else float('inf')
                improvement = best_value - value
            else:
                best_value = max([hist_val for hist_val in self.val_metrics_history[metric]] + [0]) if self.val_metrics_history[metric] else 0
                improvement = value - best_value
            
            logger.info(f"{metric:<15} | {value:>10.2f} | {best_value:>10.2f} | {improvement:>+10.2f} | {'N/A':>10}")
        
        logger.info("=" * 80)
    
    def train(self):

        if self.is_main_process:
            logger.info("=" * 80)
            logger.info(f"开始训练: 从epoch {self.start_epoch+1} 到 {self.config.epochs}")
            logger.info(f"使用i2t指标权重: {self.i2t_weights}")
            logger.info("=" * 80)
            
            # 强制刷新所有日志处理器
            for handler in logger.handlers:
                handler.flush()
            logger.info(f"Starting training from epoch {self.start_epoch+1} for {self.config.epochs} epochs")
            logger.info(f"Using i2t metrics with weights: {self.i2t_weights}")
        
        # if self.is_main_process and self.val_loader is not None:
        #     logger.info(f"Evaluating model before starting")
        #     metrics = self.evaluator.evaluate()
        # self.print_metrics_table(metrics, 0)
        
        for epoch in range(self.start_epoch+1, self.config.epochs + 1):
            try:
                # 训练一个epoch
                if self.is_main_process:
                    logger.info(f"Starting epoch {epoch}/{self.config.epochs}")
                
                epoch_loss = self.train_epoch(epoch)
                
                if self.is_main_process:
                    self.train_losses.append(epoch_loss)
                    logger.info(f"Epoch {epoch} training completed with loss: {epoch_loss:.4f}")
                
                # 更新学习率
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    if self.is_main_process:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        logger.info(f"Learning rate updated to {current_lr:.6f}")
                
                # 同步所有进程，但设置超时以避免永久挂起
                # 注意: barrier() 在验证前使用，在每个进程的训练循环结束后调用
                if self.is_distributed:
                    if self.is_main_process:
                        logger.info("Waiting for all processes to synchronize before evaluation...")
                    
                    barrier_success = barrier_with_timeout(timeout=120)  # 2分钟超时
                    
                    if not barrier_success and self.is_main_process:
                        logger.warning("Barrier timeout occurred, but continuing with evaluation")
                
                # 每个epoch后评估模型 - 只在主进程上
                if self.is_main_process and self.val_loader is not None:
                    logger.info(f"Evaluating model at epoch {epoch}...")
                    metrics = self.evaluator.evaluate()
                    
                    # 记录指标
                    for key, value in metrics.items():
                        if key not in self.val_metrics_history:
                            self.val_metrics_history[key] = []
                        self.val_metrics_history[key].append(value)
                        self.tb_logger.log_scalar(f"val/{key}", value, epoch)
                    
                    # 打印指标表格
                    self.print_metrics_table(metrics, epoch)
                    
                    # 计算当前综合得分
                    current_i2t_score = self._calculate_i2t_score(metrics)
                    
                    # 是否为最佳模型？
                    if current_i2t_score > self.best_i2t_score:
                        improvement = current_i2t_score - self.best_i2t_score
                        self.best_i2t_score = current_i2t_score
                        self.best_epoch = epoch
                        self.counter = 0  # 重置早停计数器
                        
                        # 保存模型 - 使用DDP时，只保存模块而不是整个DDP包装器
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            model_to_save = self.model.module
                        else:
                            model_to_save = self.model
                            
                        save_checkpoint(
                            model_to_save, self.optimizer, epoch, metrics,
                            os.path.join(self.config.checkpoint_dir, f"best_model.pth")
                        )
                        logger.info(f"New best model saved at epoch {epoch} with i2t combined score: {current_i2t_score:.2f} (improved by {improvement:.2f})")
                    else:
                        self.counter += 1
                        logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
                        
                        # 检查是否达到早停条件
                        if self.counter >= self.patience:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            logger.info(f"Best model was at epoch {self.best_epoch} with i2t combined score: {self.best_i2t_score:.2f}")
                            break
                    
                    # 保存检查点
                    if epoch % self.config.save_interval == 0:
                        # 同样，保存模块而不是DDP包装器
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            model_to_save = self.model.module
                        else:
                            model_to_save = self.model
                            
                        save_checkpoint(
                            model_to_save, self.optimizer, epoch, metrics,
                            os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                        )
                
                # 不在评估后添加barrier，允许其他进程继续进行下一个epoch的训练
                # 这避免了可能的分布式死锁
                
            except Exception as e:
                # 错误处理
                if self.is_main_process:
                    logger.error(f"Error in epoch {epoch}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.info("Attempting to continue with next epoch...")
                
                # 尝试重新同步进程
                if self.is_distributed:
                    try:
                        barrier_with_timeout(timeout=30)
                    except:
                        pass
        
        # 保存最终模型（如果不是最佳模型）- 只在主进程上
        if self.is_main_process:
            if epoch < self.config.epochs or self.best_epoch != epoch:  # 如果是因为早停而结束或者最后一个不是最佳
                # 同样，保存模块而不是DDP包装器
                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    model_to_save = self.model.module
                else:
                    model_to_save = self.model
                    
                save_checkpoint(
                    model_to_save, self.optimizer, epoch, None,
                    os.path.join(self.config.checkpoint_dir, "final_model.pth")
                )
            
            # 返回最佳模型指标
            best_metrics = {}
            
            # 检查是否有评估记录
            if len(self.val_metrics_history['i2t_r1']) > 0:
                # 找到最佳epoch对应的指标索引
                best_epoch_idx = self.best_epoch - (self.start_epoch + 1)
                
                # 收集所有指标的最佳值
                for key in self.val_metrics_history.keys():
                    if best_epoch_idx < len(self.val_metrics_history[key]):  # 确保索引有效
                        best_metrics[key] = self.val_metrics_history[key][best_epoch_idx]
                
                logger.info(f"Best model performance (epoch {self.best_epoch}):")
                logger.info(f"Image to Text: R@1: {best_metrics.get('i2t_r1', 0):.2f}, R@5: {best_metrics.get('i2t_r5', 0):.2f}, R@10: {best_metrics.get('i2t_r10', 0):.2f}")
                logger.info(f"Text to Image: R@1: {best_metrics.get('t2i_r1', 0):.2f}, R@5: {best_metrics.get('t2i_r5', 0):.2f}, R@10: {best_metrics.get('t2i_r10', 0):.2f}")
                logger.info(f"Combined i2t score: {self.best_i2t_score:.2f}")
            else:
                logger.warning("No validation metrics recorded during training.")
            
            return best_metrics
        else:
            # 非主进程返回None
            return None
    
    def train_epoch(self, epoch):

        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # 如果使用分布式训练，设置采样器的epoch
        if self.is_distributed and self.train_sampler is not None:
            if self.is_main_process:
                logger.info(f"Setting epoch {epoch} for distributed sampler")
            self.train_sampler.set_epoch(epoch)
        
        # 创建进度条 - 只在主进程上显示
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            print(f"Train loader has {len(self.train_loader)} batches")
        else:
            pbar = self.train_loader
        
        batch_count = 0
        total_samples = 0
        for batch_idx, (images, reports, input_ids, attention_mask, labels) in enumerate(pbar):
            batch_count += 1
            batch_size = images.size(0)
            total_samples += batch_size
            
            if batch_idx % 10 == 0 and self.is_main_process:
                print(f"Processing batch {batch_idx+1}/{len(self.train_loader)}")
                
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            joint_similarity, global_similarity, local_similarity, _ = self.model(images, input_ids, attention_mask)

            # 设置当前epoch到模型，用于硬负例挖掘中的预热判断
            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                self.model.module.current_epoch = epoch
                # 硬负例挖掘
                hard_negative_mask, neg_count = self.model.module.generate_hard_negatives(labels, joint_similarity)
                # if batch_idx % 50 == 0 and self.is_main_process:
                #     logger.info(f"Epoch {epoch}, Batch {batch_idx}: 选择了 {neg_count} 个硬负例")
            else:
                self.model.current_epoch = epoch
                # 硬负例挖掘
                hard_negative_mask, neg_count = self.model.generate_hard_negatives(labels, joint_similarity)
                # if batch_idx % 50 == 0 and self.is_main_process:
                #     logger.info(f"Epoch {epoch}, Batch {batch_idx}: 选择了 {neg_count} 个硬负例")

            # 计算损失 - 修改为传递完整的返回值元组
            loss = compute_bidirectional_contrastive_loss(
                joint_similarity, 
                temperature=self.model.module.temperature if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model.temperature,
                hard_negative_mask=(hard_negative_mask, neg_count)  # 传递完整的元组
            )
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
            self.optimizer.step()
            
            # 记录损失 - 乘以批大小以便后续计算加权平均值
            batch_loss = loss.item() * batch_size
            epoch_loss += batch_loss
            
            # 更新进度条 - 显示当前批次损失
            if self.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 输出进度和损失 - 每个log_interval批次
            if self.is_main_process and (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch}/{self.config.epochs}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Avg Loss: {epoch_loss/total_samples:.4f}")
        
        if self.is_main_process:
            print(f"Completed all {batch_count} batches for epoch {epoch}")
        
        # 计算平均损失 - 使用样本数加权
        epoch_loss /= total_samples
        
        # 如果是分布式训练，对所有进程的loss求平均
        if self.is_distributed:
            # 创建一个tensor来存储所有进程的loss
            world_size = dist.get_world_size()
            loss_tensor = torch.tensor([epoch_loss], device=self.device)
            
            if self.is_main_process:
                print(f"About to perform all_reduce for loss")
            
            # 安全地执行all_reduce
            try:
                work = dist.all_reduce(loss_tensor, async_op=True)
                work.wait(timeout=30)
                epoch_loss = loss_tensor.item() / world_size
                if self.is_main_process:
                    print(f"all_reduce completed successfully")
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"all_reduce failed: {e} - using local loss only")
        
        # 记录到Tensorboard并输出最终损失 - 只在主进程上
        if self.is_main_process:
            self.tb_logger.log_scalar("train/loss", epoch_loss, epoch)
            
            # 计算epoch耗时
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s. Average Loss: {epoch_loss:.4f}")
        
        return epoch_loss


import torch
import os
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, metrics=None, save_path=None):

    # 创建检查点字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    # 如果有评估指标，添加到检查点
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # 保存检查点
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer=None, path=None, device=None, map_location=None):
    if not path or not os.path.exists(path):
        logger.warning(f"Checkpoint path does not exist: {path}")
        return 0, None
    
    # 加载检查点 - 使用map_location参数（如果提供）
    if map_location is not None:
        checkpoint = torch.load(path, map_location=map_location)
    else:
        checkpoint = torch.load(path, map_location=device if device else torch.device('cpu'))
    
    # 加载模型状态 - 使用strict=False允许缺少的键
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Checkpoint loaded with strict=False (missing keys will be initialized)")
    except Exception as e:
        logger.warning(f"Error loading checkpoint with strict=False: {e}")
        # 如果加载失败，则打印更详细的错误信息
        import traceback
        logger.error(traceback.format_exc())
        
        # 尝试只加载匹配的键
        try:
            checkpoint_state = checkpoint['model_state_dict']
            model_state = model.state_dict()
            
            # 过滤出匹配的键
            matching_keys = {k: v for k, v in checkpoint_state.items() if k in model_state}
            model_state.update(matching_keys)
            model.load_state_dict(model_state)
            logger.info(f"Loaded only matching keys from checkpoint ({len(matching_keys)} of {len(checkpoint_state)} keys)")
        except Exception as inner_e:
            logger.error(f"Failed to load even matching keys: {inner_e}")
    
    # 如果提供了优化器，加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"Error loading optimizer state: {e}")
    
    logger.info(f"Loaded checkpoint from {path}, epoch: {checkpoint.get('epoch', 0)}")
    
    # 确保MedCLIP模型已正确加载（假设这是在DistributedDataParallel之外）
    if hasattr(model, 'module'):
        # 分布式模型情况
        if hasattr(model.module, 'image_encoder') and hasattr(model.module.image_encoder, 'medclip_model'):
            logger.info("Reinitializing MedCLIP model in image encoder...")
            # 重新初始化MedCLIP模型
            if hasattr(model.module.image_encoder.medclip_model, 'from_pretrained'):
                model.module.image_encoder.medclip_model.from_pretrained()
    else:
        # 非分布式模型情况
        if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'medclip_model'):
            logger.info("Reinitializing MedCLIP model in image encoder...")
            # 重新初始化MedCLIP模型
            if hasattr(model.image_encoder.medclip_model, 'from_pretrained'):
                model.image_encoder.medclip_model.from_pretrained()
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', None)
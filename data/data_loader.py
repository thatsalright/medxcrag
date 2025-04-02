
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import logging
from .datasets import MedicalImageTextDataset

logger = logging.getLogger(__name__)

def get_data_loaders(config):

    # 图像变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = MedicalImageTextDataset(
        image_dir=config.train_image_dir,
        report_file=config.train_report_file,
        transform=train_transform,
        max_length=config.max_text_length
    )
    
    val_dataset = MedicalImageTextDataset(
        image_dir=config.val_image_dir,
        report_file=config.val_report_file,
        transform=val_transform,
        max_length=config.max_text_length
    )
    
    test_dataset = None
    # 只有当配置中包含测试数据路径时才加载测试数据
    if hasattr(config, 'test_image_dir') and hasattr(config, 'test_report_file'):
        if os.path.exists(config.test_report_file):  # 检查文件是否存在
            test_dataset = MedicalImageTextDataset(
                image_dir=config.test_image_dir,
                report_file=config.test_report_file,
                transform=val_transform,
                max_length=config.max_text_length
            )
    
    # 是否使用DataParallel多GPU训练
    is_dp = getattr(config, 'use_dp', False) 
    
    # 对于使用DataParallel的情况，调整每个GPU上的批大小
    # 注意：DataParallel会自动分配数据到各个GPU
    
    # 当使用DataParallel时，保持原始批大小，因为DP会自动将批分割到每个GPU
    effective_batch_size = config.batch_size
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # 记录数据集和批大小信息
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    logger.info(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}")
    if is_dp and gpu_count > 1:
        logger.info(f"DataParallel enabled with {gpu_count} GPUs")
        logger.info(f"Global batch size: {effective_batch_size}, Per-GPU batch size: {effective_batch_size // gpu_count}")
    else:
        logger.info(f"Batch size: {effective_batch_size}")
    
    if test_dataset is not None:
        logger.info(f"Test set size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
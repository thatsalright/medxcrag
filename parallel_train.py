
import os
import argparse
import yaml
import torch
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from configs.default_config import Config
from utils.checkpoint import load_checkpoint, save_checkpoint
from models.model_hgmcr import HGMCR
from data.data_loader import get_data_loaders
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.helpers import set_seed

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="HGMCR模型训练")
    
    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=-1, 
                       help='Local rank for distributed training')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--name', type=str, default=None, help='实验名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    
    # 数据参数
    parser.add_argument('--train-image-dir', type=str, default=None, help='训练图像目录')
    parser.add_argument('--train-report-file', type=str, default=None, help='训练报告文件')
    parser.add_argument('--val-image-dir', type=str, default=None, help='验证图像目录')
    parser.add_argument('--val-report-file', type=str, default=None, help='验证报告文件')
    parser.add_argument('--batch-size', type=int, default=None, help='每个GPU的批大小')
    parser.add_argument('--num-workers', type=int, default=None, help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--feature-dim', type=int, default=None, help='特征维度')
    parser.add_argument('--hidden-dim', type=int, default=None, help='隐藏层维度')
    parser.add_argument('--num-time-steps', type=int, default=None, help='递归注意力金字塔的时间步数')
    parser.add_argument('--temperature', type=float, default=None, help='对比学习温度参数')
    parser.add_argument('--pretrained-medclip-path', type=str, default=None, help='预训练MedCLIP路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=None, help='基础学习率')
    parser.add_argument('--weight-decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--lr-scheduler', type=str, default=None, help='学习率调度器类型')
    parser.add_argument('--lr-step-size', type=int, default=None, help='学习率调度器步长')
    parser.add_argument('--lr-gamma', type=float, default=None, help='学习率调度器gamma')
    parser.add_argument('--grad-clip', type=float, default=None, help='梯度裁剪范数')
    parser.add_argument('--hard-negative-ratio', type=float, default=None, help='硬负例挖掘比例')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--eval-interval', type=int, default=None, help='评估间隔')
    parser.add_argument('--save-interval', type=int, default=None, help='保存间隔')
    parser.add_argument('--log-interval', type=int, default=None, help='日志间隔')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    return parser.parse_args()

def load_config_from_yaml(config_path):

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict

def configure_logging(config, is_main_process):

    # 创建日志目录
    if is_main_process:
        os.makedirs(config.log_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(config.log_dir, 'train.log')
    
    # 创建和配置日志记录器
    logger = setup_logger(name="HGMCR", log_file=log_file if is_main_process else None)
    
    # 仅在主进程记录信息
    if is_main_process:
        # 记录基本配置信息
        logger.info("=" * 80)
        logger.info(f"Experiment: {config.name}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Distributed training: {dist.get_world_size()} GPUs")
        logger.info("=" * 80)
        
        # 记录重要配置参数
        logger.info("Important configuration parameters:")
        logger.info(f"- Model: Hierarchical Gated Multi-scale Cross-modal Retrieval (HGMCR)")
        logger.info(f"- Feature dimension: {config.feature_dim}")
        logger.info(f"- Hidden dimension: {config.hidden_dim}")
        logger.info(f"- Time steps: {config.num_time_steps}")
        logger.info(f"- Batch size per GPU: {config.batch_size}")
        logger.info(f"- Global batch size: {config.batch_size * dist.get_world_size()}")
        logger.info(f"- Learning rate: {config.learning_rate}")
        logger.info(f"- Epochs: {config.epochs}")
        logger.info(f"- Temperature: {config.temperature}")
        logger.info(f"- Hard negative ratio: {config.hard_negative_ratio}")
        logger.info(f"- early_stop_patience: {config.early_stop_patience}")
        logger.info("=" * 80)
    
    # 设置全局日志级别为INFO
    logging.getLogger().setLevel(logging.INFO)
    
    # 为所有子模块设置日志记录器
    module_loggers = [
        "data", "models", "training", "utils", "inference"
    ]
    
    for module in module_loggers:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.INFO)
        if is_main_process:
            module_logger.handlers = logger.handlers  # 共享处理器
        else:
            module_logger.handlers = []  # 非主进程不输出日志
    
    return logger

def get_distributed_data_loaders(config, local_rank):

    import os
    import torchvision.transforms as transforms
    from data.datasets import MedicalImageTextDataset
    
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
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=True
    )
    
    # 创建数据加载器 - 训练集使用分布式采样器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 验证集和测试集只在主进程上创建
    val_loader = None
    test_loader = None
    
    if local_rank == 0:
        val_dataset = MedicalImageTextDataset(
            image_dir=config.val_image_dir,
            report_file=config.val_report_file,
            transform=val_transform,
            max_length=config.max_text_length
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,  # 可以用更大的batch size进行评估
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 只有当配置中包含测试数据路径时才加载测试数据
        if hasattr(config, 'test_image_dir') and hasattr(config, 'test_report_file'):
            if os.path.exists(config.test_report_file):  # 检查文件是否存在
                test_dataset = MedicalImageTextDataset(
                    image_dir=config.test_image_dir,
                    report_file=config.test_report_file,
                    transform=val_transform,
                    max_length=config.max_text_length
                )
                
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=config.batch_size * 2,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True
                )
    
    return train_loader, val_loader, test_loader, train_sampler

def main():
    args = parse_args()
    
    # 初始化分布式训练环境
    local_rank = args.local_rank
    
    # 初始化分布式进程组
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    is_main_process = (local_rank == 0)
    
    # 创建配置对象
    config = Config()
    
    # 如果指定了配置文件，从文件加载配置
    if args.config and os.path.exists(args.config):
        config_dict = load_config_from_yaml(args.config)
        config.update(config_dict)
        config.learning_rate = float(config.learning_rate)
        
    # 用命令行参数覆盖配置
    arg_dict = vars(args)
    # 移除'config'键和'local_rank'键，因为它们不是配置属性
    arg_dict.pop('config', None)
    arg_dict.pop('local_rank', None)
    config_dict = {k: v for k, v in arg_dict.items() if v is not None}
    
    # 设置输出目录
    if args.name:
        config.output_dir = f"outputs/{args.name}"
    config.update(config_dict)
    
    # 创建输出目录（仅主进程）
    if is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    # 配置日志记录器
    logger = configure_logging(config, is_main_process)
    
    # 设置随机种子 - 各进程需要不同的seed以避免采样重复
    seed = config.seed + local_rank
    set_seed(seed)
    if is_main_process:
        logger.info(f"Random seed set to {seed} for main process, others will use seed+rank")
    
    # 设置设备
    device = torch.device(f"cuda:{local_rank}")
    if is_main_process:
        logger.info(f"Using {world_size} GPUs for distributed training")
    
    # 打印配置（仅主进程）
    if is_main_process:
        config.print_config()
    
    # 获取分布式数据加载器
    if is_main_process:
        logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, train_sampler = get_distributed_data_loaders(config, local_rank)
    
    if is_main_process:
        logger.info(f"Train dataset: {len(train_loader.dataset)} samples")
        if val_loader:
            logger.info(f"Validation dataset: {len(val_loader.dataset)} samples")
    
    # 创建模型
    if is_main_process:
        logger.info("Creating model...")
    model = HGMCR(config)
    model = model.to(device)
    
    # 将模型包装为DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    
    # 打印模型信息（仅主进程）
    if is_main_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {num_params:,} trainable parameters")
    
    global_batch_size = config.batch_size * world_size
    scaled_lr = config.learning_rate * (global_batch_size / 256)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    if is_main_process:
        logger.info(f"Optimizer: Adam (base_lr={config.learning_rate}, scaled_lr={scaled_lr}, "
                    f"weight_decay={config.weight_decay})")
    
    # 如果指定了恢复训练的检查点路径，加载检查点
    start_epoch = 0
    best_metrics = None
    if args.resume and os.path.exists(args.resume):
        if is_main_process:
            logger.info(f"Loading checkpoint from {args.resume}")
        # 所有进程都需要加载相同的权重
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        start_epoch, best_metrics = load_checkpoint(model, optimizer, args.resume, device, map_location=map_location)
        if is_main_process:
            logger.info(f"Resuming training from epoch {start_epoch+1}")
    
    # 确保所有进程都已加载模型权重
    dist.barrier()
    
    trainer = Trainer(model, train_loader, val_loader, config, device, 
                     start_epoch=start_epoch, best_metrics=best_metrics,
                     is_distributed=True, train_sampler=train_sampler,
                     local_rank=local_rank, is_main_process=is_main_process)
    
    # 开始训练
    if is_main_process:
        logger.info("=" * 80)
        logger.info(f"Starting training for {config.epochs} epochs")
        logger.info("=" * 80)
    
    best_metrics = trainer.train()
    
    # 打印最佳结果（仅主进程）
    if is_main_process:
        logger.info("=" * 80)
        logger.info("Training completed.")
        from utils.helpers import format_metrics
        logger.info(f"Best metrics: {format_metrics(best_metrics)}")
        logger.info("=" * 80)
    
    # 清理分布式环境
    dist.destroy_process_group()
    
    return best_metrics

if __name__ == "__main__":
    main()
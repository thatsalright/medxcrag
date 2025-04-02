
import os
import argparse
import yaml
import torch
import logging
import time
import json
import torch.distributed as dist
from tqdm import tqdm
import torchvision.transforms as transforms
# 导入原始代码模块
from configs.default_config import Config
from utils.checkpoint import load_checkpoint
from models.model_hgmcr import HGMCR
from data.datasets import MedicalImageTextDataset
from data.data_loader import get_data_loaders  # 复用原有的数据加载函数
from utils.logger import setup_logger
from utils.helpers import set_seed, format_metrics
from training.evaluator import Evaluator  # 导入原始的评估器类

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="HGMCR模型批量评估")
    
    # 基本参数
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output-dir', type=str, default='eval_results', help='评估结果输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    
    # 数据参数
    parser.add_argument('--batch-size', type=int, default=32, help='评估批大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作进程数')
    
    # 评估参数
    parser.add_argument('--max-eval-samples', type=int, default=0, 
                        help='每个数据集最大评估样本数，设为0表示不限制')
    
    return parser.parse_args()

def load_config_from_yaml(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict

def configure_logging(config, output_dir):
    # 创建日志目录
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(log_dir, 'eval.log')
    
    # 创建和配置日志记录器
    logger = setup_logger(name="HGMCR_EVAL", log_file=log_file)
    
    # 记录基本配置信息
    logger.info("=" * 80)
    logger.info(f"HGMCR模型批量评估")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 80)
    
    return logger

def load_data(config):
    batch_size=config.batch_size
    num_workers=config.num_workers
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = MedicalImageTextDataset(
        image_dir=config.val_image_dir,
        report_file=config.val_report_file,
        transform=val_transform,
        max_length=config.max_text_length
    )
    
    # 创建数据加载器
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return val_loader

def load_model(config, checkpoint_path, device):
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 创建模型
    model = HGMCR(config)
    model = model.to(device)
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        epoch, best_metrics = load_checkpoint(model, None, checkpoint_path, device)
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        if best_metrics:
            logger.info(f"Checkpoint metrics: {format_metrics(best_metrics)}")
    else:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 设置为评估模式
    model.eval()
    
    return model

def evaluate_on_dataset(model, config, image_dir, report_file, args, device):

    
    # 创建结果目录
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"图像目录: {image_dir}")
    logger.info(f"报告文件: {report_file}")
    
    
    try:

        val_loader = load_data(config)

        evaluator = Evaluator(model, val_loader, device)
        
        # 评估模型
        start_time = time.time()
        metrics = evaluator.evaluate()
        eval_time = time.time() - start_time
        
        # # 打印指标
        # logger.info(f"评估完成，耗时: {eval_time:.2f}秒")
        # logger.info("检索指标:")
        # logger.info(f"图像到文本: R@1: {metrics['i2t_r1']:.2f}, R@5: {metrics['i2t_r5']:.2f}, R@10: {metrics['i2t_r10']:.2f}, MedR: {metrics['i2t_medr']:.1f}")
        # logger.info(f"文本到图像: R@1: {metrics['t2i_r1']:.2f}, R@5: {metrics['t2i_r5']:.2f}, R@10: {metrics['t2i_r10']:.2f}, MedR: {metrics['t2i_medr']:.1f}")

        result_file = os.path.join(results_dir, f"batch_evaluate.json")
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"结果保存到: {result_file}")
        
        print(metrics)
        # 清除评估器缓存，释放内存
        if hasattr(evaluator, 'clear_cache'):
            evaluator.clear_cache()
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
    


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    
    # 加载配置
    config = Config()
    if os.path.exists(args.config):
        config_dict = load_config_from_yaml(args.config)
        config.update(config_dict)
    else:
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(config.seed)
    
    # 配置日志
    logger = configure_logging(config, args.output_dir)
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(config, args.checkpoint, device)
    
    evaluate_on_dataset(model, config, config.val_image_dir, config.val_report_file, args, device)
    

if __name__ == "__main__":
    main()
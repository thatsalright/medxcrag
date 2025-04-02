
import os
from datetime import datetime

class Config:
    def __init__(self):
        # 基本设置
        self.name = "HGMCR"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.seed = 42  # 添加默认随机种子
        
        # 路径设置
        self.output_dir = f"outputs/{self.name}_{self.timestamp}"
        self.checkpoint_dir = ""
        self.log_dir = ""
        self.result_dir = ""
        
        # 数据设置
        self.train_image_dir = "data/train/images"
        self.train_report_file = "data/train/reports.csv"
        self.val_image_dir = "data/val/images"
        self.val_report_file = "data/val/reports.csv"
        self.test_image_dir = "data/test/images"
        self.test_report_file = "data/test/reports.csv"
        self.max_text_length = 128
        
        # 模型设置
        self.feature_dim = 512  # 特征维度
        self.hidden_dim = 512  # 隐藏层维度
        self.num_time_steps = 5  # 递归注意力金字塔的时间步数
        self.temperature = 0.05  # 对比学习温度参数
        
        # 训练设置
        self.batch_size = 32
        self.num_workers = 4
        self.epochs = 30
        self.learning_rate = 1e-4
        self.lr_scheduler = "step"  # 可选: "step", "cosine"
        self.lr_step_size = 10
        self.lr_gamma = 0.1
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.hard_negative_ratio = 0.3  # 硬负例挖掘比例
        self.resume = ""
        self.early_stop_patience = 5
        
        # 评估设置
        self.eval_interval = 1  # 每隔多少个epoch评估一次
        self.save_interval = 5  # 每隔多少个epoch保存模型
        self.log_interval = 10  # 每隔多少个batch打印一次日志
        
        # 推理设置
        self.top_k = 5  # 检索结果返回top-k
        self.visualize_attention = True  # 是否可视化注意力图
        
    def update(self, config_dict):
        """从字典中更新配置"""
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.result_dir = os.path.join(self.output_dir, "results")
        for key, value in config_dict.items():
            if hasattr(self, key):
                # 尝试将数值类型的字符串转换为浮点数
                if isinstance(value, str):
                    try:
                        # 检查字符串是否表示数值
                        if value.replace('.', '', 1).replace('e-', '', 1).replace('e+', '', 1).isdigit() or \
                        (value.startswith('-') and value[1:].replace('.', '', 1).replace('e-', '', 1).replace('e+', '', 1).isdigit()):
                            # 如果是数值字符串，转换为浮点数
                            value = float(value)
                    except (ValueError, AttributeError):
                        # 如果转换失败，保持原值
                        pass
                setattr(self, key, value)
            else:
                raise ValueError(f"Config has no attribute named '{key}'")
    
    def to_dict(self):
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def print_config(self):
        """打印配置"""
        print("="*50)
        print("HGMCR Configuration:")
        print("="*50)
        for key, value in sorted(self.to_dict().items()):
            print(f"{key}: {value}")
        print("="*50)

# 默认配置实例
default_config = Config()

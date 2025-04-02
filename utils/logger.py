

import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def setup_logger(name=None, log_file=None, level=logging.INFO):

    # 获取日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 移除所有现有处理器，确保不重复添加
    if logger.handlers:
        logger.handlers = []
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')  # 使用追加模式
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 设置传播标志，确保日志不会传播到父记录器
    logger.propagate = False
    
    return logger

class TensorboardLogger:
    def __init__(self, log_dir):

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        
    def log_scalar(self, tag, value, step=None):

        if step is None:
            self.step += 1
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, tag, scalars, step=None):

        if step is None:
            self.step += 1
            step = self.step
        self.writer.add_scalars(tag, scalars, step)
        
    def log_histogram(self, tag, values, step=None):

        if step is None:
            self.step += 1
            step = self.step
        self.writer.add_histogram(tag, values, step)
        
    def log_image(self, tag, img_tensor, step=None):

        if step is None:
            self.step += 1
            step = self.step
        self.writer.add_image(tag, img_tensor, step)
        
    def log_figure(self, tag, figure, step=None):

        if step is None:
            self.step += 1
            step = self.step
        self.writer.add_figure(tag, figure, step)
        
    def log_text(self, tag, text, step=None):

        if step is None:
            self.step += 1
            step = self.step
        self.writer.add_text(tag, text, step)
        
    def close(self):

        self.writer.close()

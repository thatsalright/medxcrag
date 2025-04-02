

import random
import numpy as np
import torch
import datetime
import os

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 为了完全的可重复性，可以设置cudnn的随机性
    # 但这可能会降低性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_time_string():

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_metrics(metrics):

    return " | ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])

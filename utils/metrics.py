
import torch
import numpy as np

def compute_retrieval_metrics(similarity_matrix, hard_negative_mask=None):
    # 如果使用硬负例掩码，将非负例位置的相似度设为一个很小的值
    if hard_negative_mask is not None:
        # 对角线也标记为1（包括正样本）
        diagonal_mask = torch.eye(similarity_matrix.size(0)).to(similarity_matrix.device)
        mask = diagonal_mask + hard_negative_mask
        
        # 将其他位置的相似度设为一个很小的值
        similarity_matrix = similarity_matrix * mask + (1 - mask) * (-1e10)
    
    # 计算图像到文本和文本到图像的排名
    i2t_ranks = []
    t2i_ranks = []
    
    # 图像到文本排名
    for i in range(similarity_matrix.size(0)):
        # 获取当前图像与所有文本的相似度
        sim_i = similarity_matrix[i]
        
        # 当前图像与对应文本的相似度
        gt_sim = sim_i[i]
        
        # 计算排名（有多少文本的相似度大于等于正样本）
        rank = (sim_i > gt_sim).sum().item() + 1
        
        i2t_ranks.append(rank)
    
    # 文本到图像排名
    for i in range(similarity_matrix.size(1)):
        # 获取所有图像与当前文本的相似度
        sim_i = similarity_matrix[:, i]
        
        # 当前文本与对应图像的相似度
        gt_sim = sim_i[i]
        
        # 计算排名（有多少图像的相似度大于等于正样本）
        rank = (sim_i > gt_sim).sum().item() + 1
        
        t2i_ranks.append(rank)
    
    # 转换为numpy数组方便计算
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    # 计算R@K
    i2t_r1 = 100.0 * np.mean(i2t_ranks <= 1)
    i2t_r5 = 100.0 * np.mean(i2t_ranks <= 5)
    i2t_r10 = 100.0 * np.mean(i2t_ranks <= 10)
    i2t_medr = np.median(i2t_ranks)
    
    t2i_r1 = 100.0 * np.mean(t2i_ranks <= 1)
    t2i_r5 = 100.0 * np.mean(t2i_ranks <= 5)
    t2i_r10 = 100.0 * np.mean(t2i_ranks <= 10)
    t2i_medr = np.median(t2i_ranks)
    
    # 返回指标字典
    metrics = {
        'i2t_r1': i2t_r1,
        'i2t_r5': i2t_r5,
        'i2t_r10': i2t_r10,
        'i2t_medr': i2t_medr,
        't2i_r1': t2i_r1,
        't2i_r5': t2i_r5,
        't2i_r10': t2i_r10,
        't2i_medr': t2i_medr
    }
    
    return metrics

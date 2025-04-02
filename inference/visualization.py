
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_attention_map(image, attention_matrix, save_path=None):

    # 调整注意力矩阵的尺寸以匹配图像尺寸
    h, w = image.shape[:2]
    
    # 将注意力矩阵从(H*W, seq_len)调整为(H, W, seq_len)，然后计算每个像素位置的平均注意力
    att_size = int(np.sqrt(attention_matrix.shape[0]))
    attention_map = attention_matrix.reshape(att_size, att_size, -1).mean(axis=2)
    
    # 上采样注意力图以匹配原始图像尺寸
    attention_map = cv2.resize(attention_map, (w, h))
    
    # 归一化注意力值到[0,1]范围
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
    
    # 创建热力图
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示注意力热力图
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(attention_map, alpha=0.5, cmap='jet')
    plt.colorbar(label='注意力')
    plt.title('注意力热力图')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_top_matching_terms(tokenizer, attention_matrix, tokens, top_k=5):

    # 获取文本中的术语
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # 计算每个图像区域对应的主要文本术语
    num_regions = attention_matrix.shape[0]
    
    # 选择几个代表性区域而不是所有区域
    region_step = max(1, num_regions // 9)  # 选择约9个代表性区域
    selected_regions = list(range(0, num_regions, region_step))
    
    top_terms = []
    
    for region_idx in selected_regions:
        region_attention = attention_matrix[region_idx]
        
        # 过滤掉特殊token（如[CLS], [SEP], [PAD]等）
        special_token_indices = [i for i, token in enumerate(decoded_tokens) if token in ['[CLS]', '[SEP]', '[PAD]']]
        filtered_attention = np.copy(region_attention)
        filtered_attention[special_token_indices] = -np.inf
        
        # 获取top-k术语
        top_token_indices = np.argsort(filtered_attention)[-top_k:][::-1]
        top_token_weights = [filtered_attention[idx] for idx in top_token_indices]
        top_token_terms = [decoded_tokens[idx] for idx in top_token_indices]
        
        top_terms.append({
            'region': region_idx,
            'terms': list(zip(top_token_terms, top_token_weights))
        })
    
    return top_terms

def plot_retrieval_results(query_item, results, is_image_query=True, save_path=None):

    if is_image_query:
        # 图像到文本检索
        plt.figure(figsize=(12, 8))
        
        # 显示查询图像
        query_img = plt.imread(query_item)
        plt.subplot(1, 2, 1)
        plt.imshow(query_img)
        plt.title('查询图像')
        plt.axis('off')
        
        # 显示检索结果
        plt.subplot(1, 2, 2)
        result_text = "检索结果:\n\n"
        for i, (text, sim) in enumerate(results):
            result_text += f"{i+1}. 相似度: {sim:.4f}\n{text}\n\n"
        plt.text(0.1, 0.5, result_text, fontsize=10, va='center')
        plt.axis('off')
    else:
        # 文本到图像检索
        num_results = len(results)
        plt.figure(figsize=(12, 4 + num_results * 3))
        
        # 显示查询文本
        plt.subplot(1, num_results + 1, 1)
        plt.text(0.1, 0.5, f"查询文本:\n\n{query_item}", fontsize=10, va='center')
        plt.axis('off')
        
        # 显示检索结果
        for i, (img_path, sim) in enumerate(results):
            plt.subplot(1, num_results + 1, i + 2)
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.title(f"相似度: {sim:.4f}")
            plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

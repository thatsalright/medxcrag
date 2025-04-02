
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer
import numpy as np
import logging
from tqdm import tqdm
import os

from inference.visualization import visualize_attention_map, analyze_top_matching_terms

logger = logging.getLogger(__name__)

class Inference:

    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = config
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",trust_remote_code=True)
        
        # 获取超参数
        self.max_length = config.max_text_length if config else 128
        self.top_k = config.top_k if config else 5
        
    def preprocess_image(self, image_path):

        try:
            original_image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
            return image_tensor, original_image
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None, None
    
    def preprocess_text(self, text):

        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        
        return input_ids, attention_mask
    
    def extract_image_features(self, image_tensor):

        with torch.no_grad():
            fused_features, global_features, local_features = self.model.image_encoder(image_tensor)
        
        return fused_features, global_features, local_features
    
    def extract_text_features(self, input_ids, attention_mask):

        with torch.no_grad():
            fused_features, global_features, local_features = self.model.text_encoder(input_ids, attention_mask)
        
        return fused_features, global_features, local_features
    
    def compute_similarity(self, image_features, text_features):

        similarity = F.cosine_similarity(image_features, text_features).item()
        return similarity
    
    def image_to_text_retrieval(self, image_path, text_database, output_dir=None):

        # 预处理图像
        image_tensor, original_image = self.preprocess_image(image_path)
        if image_tensor is None:
            return []
        
        # 提取图像特征
        image_fused, image_global, image_local = self.extract_image_features(image_tensor)
        
        # 计算与每个文本的相似度
        similarities = []
        
        for text in tqdm(text_database, desc="计算相似度"):
            # 预处理文本
            input_ids, attention_mask = self.preprocess_text(text)
            
            # 提取文本特征
            text_fused, text_global, text_local = self.extract_text_features(input_ids, attention_mask)
            
            # 计算相似度
            sim = self.compute_similarity(image_fused, text_fused)
            similarities.append(sim)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        top_similarities = [similarities[i] for i in top_indices]
        top_texts = [text_database[i] for i in top_indices]
        
        results = list(zip(top_texts, top_similarities))
        
        # 可视化注意力图
        if output_dir and self.config and self.config.visualize_attention:
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取最匹配文本的特征
            best_text = top_texts[0]
            input_ids, attention_mask = self.preprocess_text(best_text)
            _, _, text_local = self.extract_text_features(input_ids, attention_mask)
            
            # 计算注意力矩阵
            with torch.no_grad():
                visual_local_reshaped = image_local.view(1, image_local.size(1), -1).transpose(1, 2)
                _, attention_matrix = self.model.alignment_engine.compute_hierarchical_attention(
                    visual_local_reshaped, 
                    text_local
                )
            
            # 可视化
            attention_path = os.path.join(output_dir, 'attention_map.png')
            visualize_attention_map(
                np.array(original_image), 
                attention_matrix.cpu().numpy()[0], 
                save_path=attention_path
            )
            
            # 分析最匹配的术语
            terms_path = os.path.join(output_dir, 'top_terms.txt')
            top_terms = analyze_top_matching_terms(
                self.tokenizer, attention_matrix.cpu().numpy()[0], input_ids.cpu().numpy(), top_k=5
            )
            
            # 保存分析结果
            with open(terms_path, 'w') as f:
                for region in top_terms:
                    f.write(f"Region {region['region']}:\n")
                    for term, weight in region['terms']:
                        f.write(f"  {term}: {weight:.4f}\n")
                    f.write("\n")
        
        return results
    
    def text_to_image_retrieval(self, query_text, image_database, output_dir=None):

        # 预处理文本
        input_ids, attention_mask = self.preprocess_text(query_text)
        
        # 提取文本特征
        text_fused, text_global, text_local = self.extract_text_features(input_ids, attention_mask)
        
        # 计算与每个图像的相似度
        similarities = []
        valid_paths = []
        
        for img_path in tqdm(image_database, desc="计算相似度"):
            # 预处理图像
            image_tensor, _ = self.preprocess_image(img_path)
            if image_tensor is None:
                continue
            
            # 提取图像特征
            image_fused, image_global, image_local = self.extract_image_features(image_tensor)
            
            # 计算相似度
            sim = self.compute_similarity(image_fused, text_fused)
            similarities.append(sim)
            valid_paths.append(img_path)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        top_similarities = [similarities[i] for i in top_indices]
        top_images = [valid_paths[i] for i in top_indices]
        
        results = list(zip(top_images, top_similarities))
        
        # 保存匹配的图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            for i, (img_path, sim) in enumerate(results):
                img = Image.open(img_path).convert('RGB')
                img.save(os.path.join(output_dir, f"match_{i+1}.jpg"))
            
            # 可视化最佳匹配的注意力图
            if self.config and self.config.visualize_attention and results:
                best_img_path = top_images[0]
                image_tensor, original_image = self.preprocess_image(best_img_path)
                
                # 提取图像特征
                _, _, image_local = self.extract_image_features(image_tensor)
                
                # 计算注意力矩阵
                with torch.no_grad():
                    visual_local_reshaped = image_local.view(1, image_local.size(1), -1).transpose(1, 2)
                    _, attention_matrix = self.model.alignment_engine.compute_hierarchical_attention(
                        visual_local_reshaped, 
                        text_local
                    )
                
                # 可视化
                attention_path = os.path.join(output_dir, 'attention_map.png')
                visualize_attention_map(
                    np.array(original_image), 
                    attention_matrix.cpu().numpy()[0], 
                    save_path=attention_path
                )
        
        return results

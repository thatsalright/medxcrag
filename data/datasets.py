
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class MedicalImageTextDataset(Dataset):

    def __init__(self, image_dir, report_file, transform=None, max_length=128, num_disease_labels=14):

        self.image_dir = image_dir
        self.reports = self.load_reports(report_file)
        self.image_paths = list(self.reports.keys())
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载文本分词器
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",trust_remote_code=True)
        self.max_length = max_length
        
        # 加载疾病标签（如果有）
        self.disease_labels = self.load_disease_labels(report_file, num_disease_labels)
        
        logger.info(f"Loaded dataset with {len(self.image_paths)} samples from {image_dir}")
        
    def load_reports(self, report_file):
        reports = {}
        
        # 根据文件格式解析报告数据
        if report_file.endswith('.csv'):
            df = pd.read_csv(report_file)
            for _, row in df.iterrows():
                img_id = row['image_id']
                report = row['report']
                reports[img_id] = report
        elif report_file.endswith('.json'):
            with open(report_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    img_id = item['image_id']
                    report = item['report']
                    reports[img_id] = report
        else:
            # 简单文本文件格式
            with open(report_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_id = parts[0]
                        report = parts[1]
                        reports[img_id] = report
        
        return reports
    
    def load_disease_labels(self, report_file, num_disease_labels):
        disease_labels = {}
        
        # 尝试从报告文件中解析疾病标签
        if report_file.endswith('.csv'):
            df = pd.read_csv(report_file)
            if 'disease_labels' in df.columns:
                for _, row in df.iterrows():
                    img_id = row['image_id']
                    if isinstance(row['disease_labels'], str):
                        labels = [int(l) for l in row['disease_labels'].split(',')]
                    else:
                        labels = []
                    disease_labels[img_id] = labels
        elif report_file.endswith('.json'):
            with open(report_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    if 'disease_labels' in item:
                        img_id = item['image_id']
                        labels = item['disease_labels']
                        disease_labels[img_id] = labels
        
        # 如果没有找到疾病标签，创建空标签
        if not disease_labels:
            logger.warning(f"No disease labels found in {report_file}. Using empty labels.")
            for img_id in self.reports.keys():
                disease_labels[img_id] = []
        
        return disease_labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_id = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_id)
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # 如果图像损坏，返回零张量
            image = Image.new('RGB', (224, 224), color=0)
        
        if self.transform:
            image = self.transform(image)
        
        # 获取报告文本
        report = self.reports[image_id]
        
        # 对文本进行分词处理
        encoded_text = self.tokenizer(
            report,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        
        # 获取疾病标签
        disease_label = self.disease_labels.get(image_id, [])
        
        # 转换为one-hot向量
        num_diseases = 14  # 假设有14种常见疾病
        label_tensor = torch.zeros(num_diseases)
        for label in disease_label:
            if 0 <= label < num_diseases:
                label_tensor[label] = 1.0
        
        return image, report, input_ids, attention_mask, label_tensor

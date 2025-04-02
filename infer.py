import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import os
import yaml
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import argparse

# Import project modules
from models.model_hgmcr import HGMCR
from utils.checkpoint import load_checkpoint

class Config:

    def __init__(self):
        self.feature_dim = 512
        self.hidden_dim = 512
        self.num_time_steps = 5
        self.temperature = 0.05
        self.max_text_length = 512
        
    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def run_inference(config_path, checkpoint_path, image_path, reports_path, top_k=5, gpu=0):

    # Setup device
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = Config()
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config.update(config_dict)
    
    # Create and load model
    model = HGMCR(config).to(device)
    epoch, _ = load_checkpoint(model, None, checkpoint_path, device)
    print(f"Loaded model from epoch {epoch}")
    model.eval()
    
    # Load reports
    df = pd.read_csv(reports_path)
    reports = dict(zip(df['image_id'], df['report']))
    report_ids = list(reports.keys())
    report_texts = list(reports.values())
    
    # Load and process image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    
    # Encode image
    with torch.no_grad():
        image_features, _, _ = model.encode_image(image_tensor)
        image_features = F.normalize(image_features, p=2, dim=1)
    
    # Process and encode reports
    all_text_features = []
    batch_size = 32
    
    for i in range(0, len(report_texts), batch_size):
        batch_reports = report_texts[i:i+batch_size]
        
        encoded_texts = tokenizer(
            batch_reports,
            padding='max_length',
            truncation=True,
            max_length=config.max_text_length,
            return_tensors='pt'
        )
        
        input_ids = encoded_texts['input_ids'].to(device)
        attention_mask = encoded_texts['attention_mask'].to(device)
        
        with torch.no_grad():
            batch_text_features, _, _ = model.encode_text(input_ids, attention_mask)
            batch_text_features = F.normalize(batch_text_features, p=2, dim=1)
            all_text_features.append(batch_text_features)
    
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Compute similarities and get top-k matches
    similarities = torch.mm(image_features, all_text_features.t()).squeeze(0)
    top_values, top_indices = torch.topk(similarities, min(top_k, len(report_texts)))
    
    # Print results
    print(f"\nTop {top_k} similar reports for {os.path.basename(image_path)}:")
    print("-" * 80)
    for i, (score, idx) in enumerate(zip(top_values.cpu().numpy(), top_indices.cpu().numpy())):
        print(f"Rank {i+1} (Score: {score:.4f})")
        print(f"Report ID: {report_ids[idx]}")
        print(f"Report: {report_texts[idx]}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HGMCR Model Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--reports', type=str, required=True, help='Path to CSV file with reports')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top reports to show')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, args.image, args.reports, args.top_k, args.gpu)
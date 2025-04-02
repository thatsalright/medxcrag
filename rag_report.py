import base64
from PIL import Image
import io
import torch
import torch.nn.functional as F
import pandas as pd
import os
import yaml
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from llavamed import SimpleLLaVaMed
from llava.conversation import conv_templates
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

def retrival(image_path, model, all_text_features, top_k):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Encode image
    with torch.no_grad():
        image_features, _, _ = model.encode_image(image_tensor)
        image_features = F.normalize(image_features, p=2, dim=1)
    
    
    
    # Compute similarities and get top-k matches
    similarities = torch.mm(image_features, all_text_features.t()).squeeze(0)
    top_values, top_indices = torch.topk(similarities, min(top_k, len(report_texts)))
    
    top_results = []
    # Print results
    # print(f"\nTop {top_k} similar reports for {os.path.basename(image_path)}:")

    for i, (score, idx) in enumerate(zip(top_values.cpu().numpy(), top_indices.cpu().numpy())):
        # print(f"Rank {i+1} (Score: {score:.4f})")
        # print(f"Report ID: {report_ids[idx]}")
        # print(f"Report{i+1}: {report_texts[idx]}")
        top_results.append(f"Report{i+1}: {report_texts[idx]}\n")
    return top_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HGMCR Model Inference')
    parser.add_argument('--config', type=str, default="configs/infer_config.yaml")
    parser.add_argument('--checkpoint', type=str, default="checkpoint/HGMCR_ckpt.pth")
    parser.add_argument('--images', type=str, default="../dataset/mimic-cxr/test/images")
    parser.add_argument('--reports', type=str, default="../dataset/mimic-cxr/test/test_report.csv")
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    config_path = args.config
    checkpoint_path = args.checkpoint
    image_paths =  args.images
    reports_path = args.reports
    top_k = args.top_k
    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    config = Config()
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config.update(config_dict)

    # Load HGMCR
    model = HGMCR(config).to(device)
    load_checkpoint(model, None, checkpoint_path, device)
    model.eval()
    # Load LLaVaMed
    LLaVaMed_model = SimpleLLaVaMed(
        model_path="microsoft/llava-med-v1.5-mistral-7b",
        device="cuda",
        num_gpus=2
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)

    # Process and encode reports
    all_text_features = []
    batch_size = 32
    df = pd.read_csv(reports_path)
    reports = dict(zip(df['image_id'], df['report']))
    report_ids = list(reports.keys())
    report_texts = list(reports.values())
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

    prompt_template = "analyze this chest x-ray image <image> and describe this x-ray image as a radiologist. You can refer to this report with similar images{top_results}, The following key terms must be clearly described: Lung fields (clear/infiltrates/nodules), Cardiac size (normal/enlarged), Mediastinum (no shift/mass), Bones (fracture/bony destruction), Pneumothorax (present/absent), Pleural effusion (present/absent). The diagnostic section must include Conclusion and Recommendations. Now describe this x-ray image as a radiologist"
    
    file_paths = []
    for root, dirs, files in os.walk(image_paths):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    
    report_data = {"image":[], "similar_reports":[], "report":[]}
    for image in tqdm(file_paths):
        top_results = retrival(image, model, all_text_features, top_k)
        user_message = prompt_template.format(top_results=top_results)
        images = []
        with open(image, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            images.append(base64_image)
        
        if "<image>" not in user_message:
            user_message = user_message + " <image>"
            print(f"Added image reference. New prompt: {user_message}")

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], user_message)
        prompt = conv.get_prompt()

        full_output = ""
        for output in LLaVaMed_model.generate(
            prompt=prompt,
            images=images,
            temperature=0.7,
            max_new_tokens=256,
            stop=conv.sep2,
            stream=True
        ):
            # Extract just the assistant's part of the response (after [/INST])
            assistant_output = output.split("[/INST]")[-1] if "[/INST]" in output else output
            # print(assistant_output, end="\r")
            full_output = assistant_output
        report_data['image'].append(image)
        report_data['similar_reports'].append(top_results)
        report_data['report'].append(full_output)
        print("\n\nFull response:", full_output)
        # print(prompt)
    df = pd.DataFrame(report_data)
    df.to_csv("report.csv", index=False)

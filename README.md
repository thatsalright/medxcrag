# MEDXCRAG：医学跨模态检索增强的X光报告生成方法

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The data and pretrained model weights are available via Baidu Netdisk:

https://pan.baidu.com/s/1OPIMBlIcwOiuDcVY0esnxw?pwd=n7xx

## Pretrained Models

https://github.com/microsoft/LLaVA-Med

https://pan.baidu.com/s/1OPIMBlIcwOiuDcVY0esnxw?pwd=n7xx

## Usage

### HGMCR Inference 

```bash
python infer.py --config configs/infer_config.yaml --checkpoint checkpoint/HGMCR_ckpt.pth
```

### RAG Report Generation

```bash
python rag_report.py --hgmcr_checkpoint checkpoint/HGMCR_ckpt.pth --hgmcr_config configs/infer_config.yaml
```

### Training

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 parallel_train.py --config configs/mimic_train_config.yaml
```

### HGMCR Batch Evaluation

```bash
python hgmcr_batch.py --config configs/test_config.yaml --checkpoint checkpoint/HGMCR_ckpt.pth
```

### MEDXCRAG Evaluation

```bash
python medxcrag_evaluate.py --report ./mimic_report.csv
```

## Configuration

Modify the YAML files in the `configs/` directory to adjust model parameters, data paths, and training settings.




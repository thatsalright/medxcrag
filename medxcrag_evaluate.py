import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu 
from tqdm import tqdm
def load_data(report_path):
    df = pd.read_csv(report_path)

    df['report'] = df['report'].fillna('').astype(str)
    df['similar_reports'] = df['similar_reports'].fillna('').astype(str)
    
    generated_reports = df['report'].tolist()
    reference_reports = [eval(x) for x in df['similar_reports']]
    for idx, i in enumerate(reference_reports):
        reference_reports[idx] = i[0]
    # 确保两个列表长度相同
    assert len(generated_reports) == len(reference_reports), "生成报告和参考报告的数量不匹配！"
    
    return generated_reports, reference_reports

def preprocess_text(text):
    if not isinstance(text, str):
        if pd.isna(text):
            return []  
        else:
            text = str(text)  
    # 文本清理：移除多余的空格和特殊字符
    text = text.lower().strip()

    # 医学报告优化分词
    # 1. 先替换常见医学缩写中的点，避免被当作句子边界
    text = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1_DOT_\2', text)
    
    # 2. 使用正则表达式处理医学特殊术语
    # 保留常见医学测量值的完整形式，如"2.5 cm", "T2-weighted"
    text = re.sub(r'(\d+)\.(\d+)', r'\1_DECIMAL_\2', text)
    text = re.sub(r'(\w+)-(\w+)', r'\1_HYPHEN_\2', text)
    
    # 3. 使用NLTK的TreebankWordTokenizer处理
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # 4. 恢复之前替换的特殊字符
    tokens = [token.replace('_DOT_', '.').replace('_DECIMAL_', '.').replace('_HYPHEN_', '-') for token in tokens]
    
    # 5. 过滤掉医学报告中常见的无意义词（但不是全部停用词）
    medical_stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                        'and', 'or', 'but', 'if', 'then', 'of', 'to', 'in', 'for', 'with', 
                        'by', 'at', 'on', 'this', 'that', 'these', 'those'}
    tokens = [token for token in tokens if token.lower() not in medical_stopwords]
    
    return tokens

def calculate_bleu(generated, reference, weights=(0.25, 0.25, 0.25, 0.25)):
    if not generated or not reference:
        return 0.0
    
    # 使用 sacreBLEU 的分词器（默认对标 WMT 标准）
    tokenized_gen = preprocess_text(generated)
    tokenized_ref = preprocess_text(reference)
    
    # 边缘情况处理
    if not tokenized_gen or not tokenized_ref:
        return 0.0
    
    # 转换为 sacreBLEU 需要的格式（字符串列表）
    gen_text = ' '.join(tokenized_gen)
    ref_text = ' '.join(tokenized_ref)
    
    # 计算 BLEU（自动应用长度惩罚，tokenize='none' 因为我们已预处理）
    bleu_score = corpus_bleu(
        [gen_text], 
        [[ref_text]], 
        tokenize='none',  # 禁用 sacreBLEU 的默认分词
        smooth_method='exp'  # 指数平滑（对短文本更友好）
    ).score / 100  # sacreBLEU 返回百分比值
    
    return bleu_score

def calculate_meteor(generated, reference, language='en', tokenizer_type='medical'):
    if not generated:
        return 0.0
    
    # 分词
    tokenized_generated = preprocess_text(generated)
    tokenized_reference = preprocess_text(reference)
    
    # 处理边缘情况：空token列表
    if not tokenized_generated or not tokenized_reference:
        return 0.0
    
    # 计算METEOR评分
    return meteor_score([tokenized_reference], tokenized_generated)

def calculate_rouge(generated, reference, language='en', tokenizer_type='medical'):

    if not generated:
        return 0.0
    
    # 分词，然后将标记转换回字符串(保留分词的边界)
    tokenized_generated = preprocess_text(generated)
    tokenized_reference = preprocess_text(reference)
    
    # 处理边缘情况：空token列表
    if not tokenized_generated or not tokenized_reference:
        return 0.0
    
    gen_text = ' '.join(tokenized_generated)
    ref_text = ' '.join(tokenized_reference)
    
    # 初始化rouge评分器
    # use_stemmer=True使用词干提取，对医学术语可能有帮助
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # 计算ROUGE-L评分
    rouge_scores = scorer.score(ref_text, gen_text)
    
    # 返回F-score
    return rouge_scores['rougeL'].fmeasure

def evaluate_reports(generated_reports, reference_reports):

    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    meteor_scores = []
    rouge_l_scores = []
    
    for gen, ref in zip(generated_reports, reference_reports):
        # 计算BLEU-1,2,3,4
        bleu_1_scores.append(calculate_bleu(gen, ref, weights=(1, 0, 0, 0)))
        
        # 计算METEOR
        meteor_scores.append(calculate_meteor(gen, ref))
        
        # 计算ROUGE-L
        rouge_l_scores.append(calculate_rouge(gen, ref))
    
    # 计算平均值
    metrics = {
        'BLEU': np.mean(bleu_1_scores),
        'METEOR': np.mean(meteor_scores),
        'ROUGE-L': np.mean(rouge_l_scores)
    }
    
    return metrics

def main():
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, required=True)
    parser.add_argument('--output', type=str, default='evaluation_results.txt')
    
    args = parser.parse_args()
    
    generated_reports, reference_reports = load_data(args.report)
    
    # 修改评估函数，应用选定的分词器
    def evaluate_with_tokenizer(generated_reports, reference_reports):
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        meteor_scores = []
        rouge_l_scores = []
        
        for gen, ref in tqdm(zip(generated_reports, reference_reports)):
            # 使用指定的分词器处理文本
            tokenized_gen = preprocess_text(gen)
            tokenized_ref = preprocess_text(ref)
            
            # 将分词后的文本转换回字符串，用于ROUGE计算
            gen_text = ' '.join(tokenized_gen)
            ref_text = ' '.join(tokenized_ref)
            
            # 计算BLEU-1,2,3,4 (使用分词后的列表)
            smooth = SmoothingFunction().method1
            bleu_1_scores.append(sentence_bleu([tokenized_ref], tokenized_gen, 
                                             weights=(1.0, 0, 0, 0), 
                                             smoothing_function=smooth))
            # 计算METEOR (使用分词后的列表)
            meteor_scores.append(meteor_score([tokenized_ref], tokenized_gen))
            
            # 计算ROUGE-L (使用分词后的文本字符串)
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(ref_text, gen_text)
            rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
        
        # 计算平均值
        metrics = {
            'BLEU': np.mean(bleu_1_scores),
            'METEOR': np.mean(meteor_scores),
            'ROUGE-L': np.mean(rouge_l_scores)
        }
        
        return metrics
    
    metrics = evaluate_with_tokenizer(generated_reports, reference_reports)
    
    # 打印评估结果
    print("\n评估指标:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 将结果保存到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"评估指标:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\n结果已保存到 '{args.output}'")

if __name__ == "__main__":
    main()
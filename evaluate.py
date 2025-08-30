import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emoji表情库 - 小红书风格
EMOJI_POOL = [
    "✨", "💕", "💖", "💝", "💗", "💓", "💞", "💟", "💌", "💋",
    "🌸", "🌺", "🌷", "🌹", "🌻", "🌼", "🌿", "🍀", "🌱", "��",
    "��", "⭐", "💫", "🌙", "☀️", "��", "☁️", "🌤️", "🌥️", "��️",
    "🎀", "🎈", "🎉", "🎊", "🎋", "🎍", "🎎", "🎏", "🎐", "🎀",
    "💎", "💍", "💐", "💒", "💓", "💔", "💕", "💖", "💗", "💘",
    "😍", "🥰", "😘", "😋", "😊", "😉", "😌", "😇", "🤗", "🤩",
    "🔥", "💯", "💪", "👏", "🙌", "🤝", "👍", "👌", "✌️", "🤞",
    "🎯", "🎪", "🎨", "🎭", "🎬", "🎤", "🎧", "🎵", "��", "��"
]

class MetricsCalculator:
    """计算各种评估指标"""
    
    def __init__(self):
        # 加载句子嵌入模型用于计算文本相似度
        try:
            # 尝试使用本地模型或备用模型
            model_names = [
                'paraphrase-multilingual-MiniLM-L12-v2',
                'all-MiniLM-L6-v2',
                'distiluse-base-multilingual-cased-v2'
            ]
            
            self.sentence_model = None
            for model_name in model_names:
                try:
                    logger.info(f"尝试加载句子嵌入模型: {model_name}")
                    self.sentence_model = SentenceTransformer(model_name)
                    logger.info(f"成功加载句子嵌入模型: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"无法加载模型 {model_name}: {e}")
                    continue
            
            if self.sentence_model is None:
                logger.warning("所有句子嵌入模型都无法加载，将使用简单的文本相似度计算")
                
        except Exception as e:
            logger.warning(f"无法加载句子嵌入模型: {e}")
            self.sentence_model = None
    
    def calculate_text_similarity(self, pred_texts, true_texts):
        """计算文本相似度"""
        if self.sentence_model is None:
            # 使用简单的字符级相似度作为备用
            logger.info("使用字符级相似度作为备用方法")
            similarities = []
            for pred_text, true_text in zip(pred_texts, true_texts):
                # 计算字符重叠度
                pred_chars = set(pred_text.lower())
                true_chars = set(true_text.lower())
                
                if len(pred_chars) == 0 or len(true_chars) == 0:
                    similarities.append(0.0)
                else:
                    intersection = len(pred_chars.intersection(true_chars))
                    union = len(pred_chars.union(true_chars))
                    similarity = intersection / union if union > 0 else 0.0
                    similarities.append(similarity)
            
            return np.array(similarities)
        
        try:
            # 计算嵌入
            pred_embeddings = self.sentence_model.encode(pred_texts, convert_to_tensor=True)
            true_embeddings = self.sentence_model.encode(true_texts, convert_to_tensor=True)
            
            # 计算余弦相似度
            similarities = []
            for pred_emb, true_emb in zip(pred_embeddings, true_embeddings):
                sim = 1 - cosine(pred_emb.cpu().numpy(), true_emb.cpu().numpy())
                similarities.append(sim)
            
            return np.array(similarities)
        except Exception as e:
            logger.warning(f"计算文本相似度时出错: {e}")
            # 回退到字符级相似度
            return self.calculate_text_similarity(pred_texts, true_texts)
    
    def calculate_emoji_density(self, texts):
        """计算emoji密度"""
        emoji_counts = []
        for text in texts:
            count = sum(1 for char in text if ord(char) > 127 and char in EMOJI_POOL)
            emoji_counts.append(count / len(text) if len(text) > 0 else 0)
        return np.array(emoji_counts)
    
    def calculate_length_metrics(self, pred_texts, true_texts):
        """计算长度相关指标"""
        pred_lengths = [len(text) for text in pred_texts]
        true_lengths = [len(text) for text in true_texts]
        
        length_mae = mean_absolute_error(true_lengths, pred_lengths)
        length_mse = mean_squared_error(true_lengths, pred_lengths)
        length_rmse = np.sqrt(length_mse)
        
        return {
            'length_mae': length_mae,
            'length_mse': length_mse,
            'length_rmse': length_rmse,
            'pred_lengths': pred_lengths,
            'true_lengths': true_lengths
        }
    
    def calculate_xiaohongshu_style_score(self, texts):
        """计算小红书风格得分"""
        style_scores = []
        for text in texts:
            score = 0
            
            # Emoji密度得分 (0-30分)
            emoji_count = sum(1 for char in text if ord(char) > 127 and char in EMOJI_POOL)
            emoji_density = emoji_count / len(text) if len(text) > 0 else 0
            score += min(30, emoji_density * 100)
            
            # 小红书词汇得分 (0-25分)
            xiaohongshu_words = ["绝了", "太可了", "神仙", "宝藏", "绝美", "超爱", "爱了爱了", 
                                "种草", "安利", "推荐", "分享", "我的", "我觉得", "心动"]
            word_count = sum(1 for word in xiaohongshu_words if word in text)
            score += min(25, word_count * 5)
            
            # 感叹号得分 (0-15分)
            exclamation_count = text.count('！') + text.count('!')
            score += min(15, exclamation_count * 3)
            
            # 长度适中得分 (0-15分)
            if 20 <= len(text) <= 200:
                score += 15
            elif 10 <= len(text) <= 300:
                score += 10
            else:
                score += max(0, 15 - abs(len(text) - 100) / 10)
            
            # 情感词汇得分 (0-15分)
            emotion_words = ["喜欢", "爱", "心动", "美", "好看", "漂亮", "可爱", "棒", "好"]
            emotion_count = sum(1 for word in emotion_words if word in text)
            score += min(15, emotion_count * 3)
            
            style_scores.append(score)
        
        return np.array(style_scores)
    
    def calculate_diversity_metrics(self, texts):
        """计算多样性指标"""
        if not texts:
            return {'unique_ratio': 0, 'vocab_diversity': 0, 'length_diversity': 0}
        
        # 唯一性比例
        unique_texts = set(texts)
        unique_ratio = len(unique_texts) / len(texts)
        
        # 词汇多样性
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        vocab_size = len(set(all_words))
        total_words = len(all_words)
        vocab_diversity = vocab_size / total_words if total_words > 0 else 0
        
        # 长度多样性
        lengths = [len(text) for text in texts]
        length_diversity = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        
        return {
            'unique_ratio': unique_ratio,
            'vocab_diversity': vocab_diversity,
            'length_diversity': length_diversity
        }
    
    def calculate_fluency_metrics(self, texts):
        """计算流畅性指标"""
        fluency_scores = []
        for text in texts:
            score = 0
            
            # 标点符号使用
            punctuation_count = sum(1 for char in text if char in '，。！？；：')
            score += min(20, punctuation_count * 2)
            
            # 句子完整性
            if text.endswith(('。', '！', '？', '~', '～')):
                score += 15
            
            # 长度合理性
            if 10 <= len(text) <= 300:
                score += 25
            else:
                score += max(0, 25 - abs(len(text) - 100) / 10)
            
            # 重复度惩罚
            words = text.split()
            if len(words) > 1:
                unique_words = set(words)
                repetition_ratio = len(unique_words) / len(words)
                score += repetition_ratio * 20
            
            # 特殊字符比例
            special_chars = sum(1 for char in text if ord(char) > 127)
            special_ratio = special_chars / len(text) if len(text) > 0 else 0
            if 0.1 <= special_ratio <= 0.3:  # 合理的特殊字符比例
                score += 20
            else:
                score += max(0, 20 - abs(special_ratio - 0.2) * 100)
            
            fluency_scores.append(score)
        
        return np.array(fluency_scores)
    
    def calculate_comprehensive_score(self, pred_texts, true_texts):
        """计算综合得分"""
        # 获取各项指标
        text_similarities = self.calculate_text_similarity(pred_texts, true_texts)
        emoji_densities = self.calculate_emoji_density(pred_texts)
        style_scores = self.calculate_xiaohongshu_style_score(pred_texts)
        fluency_scores = self.calculate_fluency_metrics(pred_texts)
        diversity_metrics = self.calculate_diversity_metrics(pred_texts)
        
        # 计算综合得分 (0-100分)
        comprehensive_scores = []
        for i in range(len(pred_texts)):
            score = 0
            
            # 文本相似度 (30分)
            score += text_similarities[i] * 30
            
            # 小红书风格 (25分)
            score += style_scores[i] * 0.25  # 归一化到25分
            
            # 流畅性 (20分)
            score += fluency_scores[i] * 0.2  # 归一化到20分
            
            # Emoji密度 (15分)
            score += min(15, emoji_densities[i] * 100)
            
            # 多样性奖励 (10分)
            score += diversity_metrics['unique_ratio'] * 10
            
            comprehensive_scores.append(score)
        
        return np.array(comprehensive_scores)

# 在evaluate.py中修改推理时的prompt
# 修改evaluate_model函数，优化显存使用
def evaluate_model(model_dir, json_path, output_dir="./evaluation_results"):
    """评估模型性能"""
    logger.info(f"开始评估模型: {model_dir}")
    
    # 加载模型和处理器 - 使用量化减少显存
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用半精度减少显存
        low_cpu_mem_usage=True  # 减少CPU内存使用
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    # 加载数据集
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用验证集进行评估
    eval_items = data['val']
    logger.info(f"评估数据集大小: {len(eval_items)}")
    
    # 初始化评估工具
    metrics_calculator = MetricsCalculator()
    
    # 生成预测
    pred_texts = []
    true_texts = []
    
    model.eval()
    
    # 减少评估样本数量，避免显存不足
    eval_samples = min(50, len(eval_items))  # 只评估前50个样本
    logger.info(f"实际评估样本数量: {eval_samples}")
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(eval_items[:eval_samples], desc="生成预测")):
            try:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 获取真实文本
                description = None
                possible_fields = ['raw_description', 'description', 'text', 'caption', 'content']
                for field in possible_fields:
                    if field in item:
                        description = item[field]
                        break
                
                if description is None:
                    continue
                
                true_texts.append(description)
                
                # 生成预测文本
                image_path = item['image_path'].replace('\\', '/')
                if not os.path.exists(image_path):
                    image_path = os.path.join("data", os.path.basename(image_path))
                
                if not os.path.exists(image_path):
                    continue
                
                # 优化图像加载和处理
                try:
                    image = Image.open(image_path).convert('RGB')
                    # 调整图像大小，减少显存使用
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                except Exception as e:
                    logger.warning(f"图像加载失败 {image_path}: {e}")
                    continue
                
                # 构建输入 - 使用更自然的prompt
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "请用小红书博主的语气描述这张图片，要活泼可爱，多用emoji表情！✨"}
                        ],
                    }
                ]
                
                # 准备输入
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(model.device)
                
                # 生成描述 - 优化生成参数，减少显存使用
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=128,  # 减少生成长度
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=False  # 禁用缓存减少显存
                )
                
                # 解码输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.tokenizer.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                pred_texts.append(output_text[0])
                
                # 删除中间变量，释放显存
                del inputs, generated_ids, generated_ids_trimmed, output_text
                
            except Exception as e:
                logger.warning(f"生成预测时出错: {e}")
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    if len(pred_texts) == 0:
        logger.error("没有成功生成任何预测")
        return
    
    logger.info(f"成功生成 {len(pred_texts)} 个预测")
    
    # 计算各种指标
    text_similarities = metrics_calculator.calculate_text_similarity(pred_texts, true_texts)
    emoji_densities = metrics_calculator.calculate_emoji_density(pred_texts)
    length_metrics = metrics_calculator.calculate_length_metrics(pred_texts, true_texts)
    style_scores = metrics_calculator.calculate_xiaohongshu_style_score(pred_texts)
    fluency_scores = metrics_calculator.calculate_fluency_metrics(pred_texts)
    diversity_metrics = metrics_calculator.calculate_diversity_metrics(pred_texts)
    comprehensive_scores = metrics_calculator.calculate_comprehensive_score(pred_texts, true_texts)
    
    # 保存评估数据
    evaluation_data = {
        'text_similarities': text_similarities.tolist(),
        'emoji_densities': emoji_densities.tolist(),
        'true_lengths': length_metrics['true_lengths'],
        'pred_lengths': length_metrics['pred_lengths'],
        'length_mae': length_metrics['length_mae'],
        'length_mse': length_metrics['length_mse'],
        'length_rmse': length_metrics['length_rmse'],
        'style_scores': style_scores.tolist(),
        'fluency_scores': fluency_scores.tolist(),
        'diversity_metrics': diversity_metrics,
        'comprehensive_scores': comprehensive_scores.tolist(),
        'pred_texts': pred_texts,
        'true_texts': true_texts
    }
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存评估数据为JSON文件
    evaluation_data_path = os.path.join(output_dir, 'evaluation_data.json')
    with open(evaluation_data_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    # 打印主要指标
    logger.info("=" * 50)
    logger.info("评估结果摘要:")
    logger.info(f"文本相似度均值: {np.mean(text_similarities):.4f}")
    logger.info(f"Emoji密度均值: {np.mean(emoji_densities):.4f}")
    logger.info(f"风格得分均值: {np.mean(style_scores):.2f}")
    logger.info(f"流畅性得分均值: {np.mean(fluency_scores):.2f}")
    logger.info(f"综合得分均值: {np.mean(comprehensive_scores):.2f}")
    logger.info(f"长度MAE: {length_metrics['length_mae']:.2f}")
    logger.info("=" * 50)
    
    logger.info(f"评估完成！数据已保存到: {evaluation_data_path}")

# 添加一个轻量级评估函数
def quick_evaluate(model_dir, json_path, num_samples=20):
    """快速评估模型性能（使用更少的样本）"""
    logger.info(f"快速评估模型: {model_dir}")
    
    # 加载模型和处理器
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    # 加载数据集
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    eval_items = data['val'][:num_samples]  # 只使用前num_samples个样本
    logger.info(f"快速评估样本数量: {len(eval_items)}")
    
    # 生成预测
    pred_texts = []
    true_texts = []
    
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(tqdm(eval_items, desc="快速评估")):
            try:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 获取真实文本
                description = None
                possible_fields = ['raw_description', 'description', 'text', 'caption', 'content']
                for field in possible_fields:
                    if field in item:
                        description = item[field]
                        break
                
                if description is None:
                    continue
                
                true_texts.append(description)
                
                # 生成预测文本
                image_path = item['image_path'].replace('\\', '/')
                if not os.path.exists(image_path):
                    image_path = os.path.join("data", os.path.basename(image_path))
                
                if not os.path.exists(image_path):
                    continue
                
                image = Image.open(image_path).convert('RGB')
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                
                # 构建输入
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "请用小红书博主的语气描述这张图片，要活泼可爱，多用emoji表情！✨"}
                        ],
                    }
                ]
                
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(model.device)
                
                # 生成描述
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=100,  # 更短的生成长度
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=False
                )
                
                # 解码输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.tokenizer.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                pred_texts.append(output_text[0])
                
                # 删除中间变量
                del inputs, generated_ids, generated_ids_trimmed, output_text
                
            except Exception as e:
                logger.warning(f"生成预测时出错: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    if len(pred_texts) == 0:
        logger.error("没有成功生成任何预测")
        return
    
    # 计算简单指标
    metrics_calculator = MetricsCalculator()
    emoji_densities = metrics_calculator.calculate_emoji_density(pred_texts)
    style_scores = metrics_calculator.calculate_xiaohongshu_style_score(pred_texts)
    
    # 打印结果
    logger.info("=" * 50)
    logger.info("快速评估结果:")
    logger.info(f"Emoji密度均值: {np.mean(emoji_densities):.4f}")
    logger.info(f"风格得分均值: {np.mean(style_scores):.2f}")
    logger.info("=" * 50)
    
    # 显示几个示例
    logger.info("生成示例:")
    for i in range(min(3, len(pred_texts))):
        logger.info(f"预测 {i+1}: {pred_texts[i]}")
        logger.info(f"真实 {i+1}: {true_texts[i]}")
        logger.info("-" * 30)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估小红书风格图像描述模型')
    parser.add_argument('--model_dir', type=str, default="./qwen2-finetuned0824_3",
                       help='模型目录路径')
    parser.add_argument('--json_path', type=str, default="data/qwen_vl_descriptions.json",
                       help='数据集JSON文件路径')
    parser.add_argument('--quick', action='store_true',
                       help='使用快速评估模式（更少的样本）')
    parser.add_argument('--samples', type=int, default=20,
                       help='快速评估时的样本数量')
    
    args = parser.parse_args()
    
    if args.quick:
        # 快速评估
        quick_evaluate(args.model_dir, args.json_path, args.samples)
    else:
        # 完整评估
        evaluate_model(args.model_dir, args.json_path)
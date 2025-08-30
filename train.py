# 在train.py中添加训练数据保存功能
import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import EarlyStoppingCallback

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emoji表情库 - 小红书风格
EMOJI_POOL = [
    "✨", "💕", "💖", "💝", "💗", "💓", "💞", "💟", "💌", "💋",
    "🌸", "🌺", "🌷", "🌹", "🌻", "🌼", "��", "🍀", "��", "",
    "", "⭐", "💫", "🌙", "☀️", "", "☁️", "��️", "🌥️", "️",
    "🎀", "🎈", "🎉", "🎊", "🎋", "🎍", "🎎", "🎏", "🎐", "🎀",
    "💎", "💍", "💐", "💒", "💓", "💔", "💕", "💖", "💗", "💘",
    "😍", "🥰", "😘", "😋", "😊", "😉", "😌", "😇", "🤗", "🤩",
    "🔥", "💯", "💪", "👏", "🙌", "🤝", "👍", "👌", "✌️", "🤞",
    "🎯", "🎪", "🎨", "🎭", "🎬", "🎤", "��", "", "🎶", ""
]

# 数据集类
class QwenVLDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, processor, max_length=2048):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_items = data['train'] + data['val']
        self.items = []
        
        # 验证图像文件
        for item in all_items:
            # 检查是否有描述字段
            description = None
            possible_fields = ['raw_description', 'description', 'text', 'caption', 'content']
            for field in possible_fields:
                if field in item:
                    description = item[field]
                    break
            
            # 如果没有描述字段，跳过这个item
            if description is None:
                logger.warning(f"Skipping item without description: {list(item.keys())}")
                continue
            
            # 检查描述长度，太短的描述可能质量不好
            if len(description.strip()) < 10:
                logger.warning(f"Skipping item with too short description: {description[:50]}...")
                continue
            
            image_path = item['image_path'].replace('\\', '/')
            if not os.path.exists(image_path):
                # 尝试从data目录下查找
                image_path = os.path.join("data", os.path.basename(image_path))
            
            if os.path.exists(image_path):
                try:
                    # 尝试打开图像文件
                    with Image.open(image_path) as img:
                        img.verify()  # 验证图像完整性
                    item['image_path'] = image_path  # 更新为正确的路径
                    item['description'] = description  # 统一字段名为description
                    self.items.append(item)
                except Exception as e:
                    logger.warning(f"Invalid image file {image_path}: {str(e)}")
            else:
                logger.warning(f"Image file not found: {image_path}")
        
        self.processor = processor
        self.max_length = max_length
        logger.info(f"Loaded {len(self.items)} valid items")
        
        # 添加调试信息
        if len(self.items) > 0:
            # 显示第一个item的字段
            first_item = self.items[0]
            logger.info(f"First item fields: {list(first_item.keys())}")
            logger.info(f"First item description: {first_item.get('description', 'NO DESCRIPTION')[:100]}...")

    def __len__(self):
        return len(self.items)

    # 在train.py中修改QwenVLDataset类的__getitem__方法
    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item['image_path']
        description = item['description']  # 使用统一的字段名
        
        # 读取图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return {}
        
        # 优化：使用更自然的小红书风格prompt，避免分段格式
        selected_prompt = "请用小红书博主的语气描述这张图片，要活泼可爱，多用emoji表情！✨"
        
        # 优化：对原始描述进行小红书风格增强，提高emoji密度
        enhanced_description = self._enhance_xiaohongshu_style(description)
        
        # 构建对话格式 - 使用更自然的prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": selected_prompt}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": enhanced_description}
                ]
            }
        ]
        
        # 正确计算用户提示长度的方法
        full_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # 正确方法：直接使用tokenizer计算长度
        full_tokens = self.processor.tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # 找到assistant回复开始的位置
        assistant_start_token = self.processor.tokenizer.encode("<|im_start|>assistant")[0]
        input_ids = full_tokens.input_ids[0]
        
        # 创建正确的labels
        labels = input_ids.clone()
        
        # 找到assistant开始的位置
        try:
            assistant_pos = (input_ids == assistant_start_token).nonzero(as_tuple=True)[0][0].item() + 1
            # 将assistant之前的内容设为-100
            labels[:assistant_pos] = -100
        except:
            # 如果找不到assistant token，保守处理：只计算最后1/3的损失
            labels[:len(labels)//3*2] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': full_tokens.attention_mask[0],
            'labels': labels
        }

    def _enhance_xiaohongshu_style(self, description):
        """增强小红书风格，提高emoji密度"""
        import random
        
        # 扩展emoji前缀和后缀库
        emoji_prefixes = ["✨", "💕", "💖", "💝", "🌸", "🌟", "💫", "🎀", "💎", "��"]
        emoji_suffixes = ["✨", "💕", "💖", "🔥", "💯", "💪", "👏", "🎉", "💝", "��"]
        emoji_middle = ["💕", "💖", "💓", "💗", "💝", "🌸", "🌺", "🌷", "🌹", "🌻", "🌼", "��", "��", "��", "⭐", "💫", "🌙", "☀️", "🎀", "🎈", "🎉", "🎊", "💎", "💍", "💐", "😍", "🥰", "😘", "😋", "😊", "😉", "😌", "😇", "🤗", "🤩", "🔥", "💯", "💪", "👏", "🙌", "🤝", "👍", "👌", "✌️", "��"]
        
        # 小红书风格词汇替换
        style_replacements = {
            "好看": "绝美",
            "漂亮": "神仙颜值",
            "喜欢": "超爱",
            "推荐": "安利",
            "不错": "太可了",
            "好": "绝了",
            "棒": "宝藏",
            "美": "绝美",
            "可爱": "超可爱",
            "喜欢": "超爱",
            "推荐": "安利",
            "分享": "种草",
            "值得": "超值",
            "性价比": "性价比超高",
            "入手": "拔草",
            "购买": "入手",
            "使用": "体验",
            "感受": "体验感",
            "效果": "效果绝了",
            "质量": "品质超棒"
        }
        
        # 避免序号格式的词汇替换
        avoid_numbering = {
            "第一": "首先",
            "第二": "其次", 
            "第三": "最后",
            "1.": "",
            "2.": "",
            "3.": "",
            "一、": "",
            "二、": "",
            "三、": "",
            "1、": "",
            "2、": "",
            "3、": ""
        }
        
        # 词汇替换
        for old_word, new_word in style_replacements.items():
            if old_word in description:
                description = description.replace(old_word, new_word)
        
        # 避免序号格式
        for old_word, new_word in avoid_numbering.items():
            if old_word in description:
                description = description.replace(old_word, new_word)
        
        # 随机添加emoji前缀 (80%概率)
        if random.random() < 0.8:
            description = random.choice(emoji_prefixes) + " " + description
        
        # 随机在句子中间插入emoji (60%概率)
        if random.random() < 0.6 and len(description) > 20:
            # 在句子中间随机位置插入emoji
            words = description.split()
            if len(words) > 3:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, random.choice(emoji_middle))
                description = " ".join(words)
        
        # 随机添加emoji后缀 (80%概率)
        if random.random() < 0.8:
            description = description + " " + random.choice(emoji_suffixes)
        
        # 随机在句子末尾添加感叹号 (70%概率)
        if random.random() < 0.7 and not description.endswith(('！', '!')):
            description = description + "！"
        
        # 随机在句子中间添加感叹号 (40%概率)
        if random.random() < 0.4 and len(description) > 15:
            # 在句子中间随机位置添加感叹号
            if '，' in description:
                parts = description.split('，')
                if len(parts) > 1:
                    insert_pos = random.randint(0, len(parts) - 1)
                    parts[insert_pos] = parts[insert_pos] + "！"
                    description = '，'.join(parts)
        
        return description

# 自定义数据整理器
def collate_fn(batch):
    # 过滤掉无效样本
    batch = [item for item in batch if item is not None and len(item.get('input_ids', [])) > 0]
    
    if not batch:
        # 返回一个空的batch而不是None
        return {
            'input_ids': torch.empty(0, 1, dtype=torch.long),
            'attention_mask': torch.empty(0, 1, dtype=torch.long),
            'labels': torch.empty(0, 1, dtype=torch.long)
        }
    
    # 找到最大长度
    max_len = max(len(item['input_ids']) for item in batch)
    
    # 填充到最大长度
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        padding_len = max_len - seq_len
        
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.full((padding_len,), 0, dtype=torch.long)  # 使用0作为pad token
        ]))
        
        attention_mask.append(torch.cat([
            item['attention_mask'],
            torch.zeros(padding_len, dtype=torch.long)
        ]))
        
        labels.append(torch.cat([
            item['labels'],
            torch.full((padding_len,), -100, dtype=torch.long)
        ]))
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }

# 自定义Trainer类，用于记录训练数据
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # 添加**kwargs来接收额外参数
        result = super().compute_loss(model, inputs, return_outputs)
        
        # 处理返回值，可能是单个loss或(loss, outputs)元组
        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result
        
        # 记录训练损失
        if self.state.global_step % 10 == 0:  # 每10步记录一次
            self.train_losses.append(loss.item())
            # 安全地获取学习率
            try:
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    self.learning_rates.append(self.lr_scheduler.get_last_lr()[0])
                else:
                    self.learning_rates.append(0.0)
            except:
                self.learning_rates.append(0.0)
        
        return result
    
    def evaluation_step(self, model, inputs):
        loss = super().evaluation_step(model, inputs)
        self.eval_losses.append(loss.loss.item())
        return loss
    
    def save_training_data(self, output_dir):
        """保存训练数据"""
        training_data = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'total_steps': len(self.train_losses) * 10,
            'eval_steps': len(self.eval_losses)
        }
        
        # 保存为JSON文件
        training_data_path = os.path.join(output_dir, 'training_data.json')
        with open(training_data_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练数据已保存到: {training_data_path}")
        
# 主训练函数
def train():
    # 模型和训练参数
    model_name = "./Qwen2-VL-2B-Instruct"
    json_path = "data/qwen_vl_descriptions.json"
    output_dir = "./qwen2-finetuned0824_3"  # 新的输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载模型和处理器
    logger.info("Loading model and processor...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    
    # 优化LoRA配置
    lora_config = LoraConfig(
        r=32,  # 增加rank，提高表达能力
        lora_alpha=64,  # 增加alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,  # 增加dropout防止过拟合
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载tokenizer和processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # 创建数据集
    logger.info("Loading dataset...")
    dataset = QwenVLDataset(json_path, processor)
    
    # 检查数据集大小
    if len(dataset) == 0:
        raise ValueError("数据集为空！请检查JSON文件和图像路径。")
    
    # 优化：增加训练数据量
    train_size = min(2000, len(dataset)) 
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
    
    # 优化训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,  # 增加训练轮数
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,  # 增加梯度累积
        learning_rate=1e-3,  # 降低学习率，更稳定
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine_with_restarts",  # 使用带重启的cosine调度
        warmup_ratio=0.1,  # 增加warmup比例
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        max_grad_norm=0.5,  # 增加梯度裁剪阈值
        warmup_steps=200,  # 增加warmup步数
        save_total_limit=2,  # 只保存最好的2个模型
    )
    
    # 创建自定义Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # 增加patience
    )
    
    # 开始训练
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # 保存训练数据
    trainer.save_training_data(output_dir)
    
    # 保存最终模型
    logger.info("Saving final model...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train()
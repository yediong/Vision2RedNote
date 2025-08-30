import os
import torch
import logging
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_dir):
    """加载训练好的模型"""
    logger.info(f"加载模型: {model_dir}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    logger.info("模型加载完成")
    return model, processor

# 修改推理参数，提高生成质量
def generate_description(image_path, model, processor, prompt=None):
    """生成图像描述"""
    try:
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        
        # 默认prompt - 使用更简洁的表达
        if prompt is None:
            prompt = "请用小红书博主的语气描述这张图片，要活泼可爱，多用emoji表情！✨"
        
        # 构建输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        # 准备输入
        chat_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=chat_prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # 生成描述 - 优化生成参数
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,  # 减少生成长度
                do_sample=True,
                temperature=0.9,  # 提高温度，增加创造性
                top_p=0.95,  # 提高top_p
                top_k=100,  # 增加top_k
                repetition_penalty=1.2,  # 增加重复惩罚
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # 避免重复n-gram
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
        
        return output_text[0]
        
    except Exception as e:
        logger.error(f"生成描述时出错: {e}")
        return f"生成失败: {str(e)}"

def batch_generate_descriptions(image_dir, model, processor, output_file="generated_descriptions.txt"):
    """批量生成图像描述"""
    logger.info(f"开始批量生成描述，图像目录: {image_dir}")
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 批量生成描述
    results = []
    for i, image_path in enumerate(image_files):
        logger.info(f"处理第 {i+1}/{len(image_files)} 个图像: {os.path.basename(image_path)}")
        
        description = generate_description(image_path, model, processor)
        
        results.append({
            'image_path': image_path,
            'description': description
        })
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("图像路径\t生成的描述\n")
        f.write("-" * 50 + "\n")
        for result in results:
            f.write(f"{result['image_path']}\t{result['description']}\n")
    
    logger.info(f"批量生成完成，结果已保存到: {output_file}")
    return results

def interactive_inference(model, processor):
    """交互式推理"""
    logger.info("进入交互式推理模式，输入 'quit' 退出")
    
    while True:
        # 获取用户输入
        image_path = input("\n请输入图像路径 (或输入 'quit' 退出): ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在 {image_path}")
            continue
        
        # 可选的自定义prompt
        custom_prompt = input("请输入自定义prompt (直接回车使用默认): ").strip()
        if not custom_prompt:
            custom_prompt = None
        
        # 生成描述
        print("正在生成描述...")
        description = generate_description(image_path, model, processor, custom_prompt)
        
        print(f"\n生成的描述:\n{description}\n")
        print("-" * 50)

def main():
    """主函数"""
    # 模型路径
    model_dir = "./qwen2-finetuned0824_3"  # 修改为你的模型路径
    
    # 检查模型是否存在
    if not os.path.exists(model_dir):
        logger.error(f"模型路径不存在: {model_dir}")
        logger.info("请先运行 train.py 训练模型，或修改 model_dir 路径")
        return
    
    # 加载模型
    model, processor = load_model(model_dir)
    
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 单张图像推理")
    print("2. 批量图像推理")
    print("3. 交互式推理")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 单张图像推理
        # image_path = input("请输入图像路径: ").strip()
        image_path = "data/Train/10.jpg"
        if os.path.exists(image_path):
            description = generate_description(image_path, model, processor)
            print(f"\n生成的描述:\n{description}")
        else:
            print(f"错误: 文件不存在 {image_path}")
    
    elif choice == "2":
        # 批量图像推理
        image_dir = input("请输入图像目录路径: ").strip()
        if os.path.exists(image_dir):
            batch_generate_descriptions(image_dir, model, processor)
        else:
            print(f"错误: 目录不存在 {image_dir}")
    
    elif choice == "3":
        # 交互式推理
        interactive_inference(model, processor)
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
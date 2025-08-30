# -*- coding: utf-8 -*-
import os
import json
import glob
import asyncio
import base64
import mimetypes
import time
import logging
from openai import AsyncOpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("descriptions/pipeline.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QwenVLPipeline")

class QwenVLPipeline:
    """
    基于Qwen-VL的图片描述生成系统，专为生成小红书风格描述设计
    """
    def __init__(
        self,
        api_key: str = "sk-5f28127ea6524835b3304469e54d05f5",
        model_name: str = "qwen-vl-max",  # 使用正确的Qwen-VL模型名称
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 修正：移除末尾空格
        output_dir: str = "descriptions"
    ):
        """
        初始化Qwen-VL管道
        
        参数:
        api_key: DashScope API Key
        model_name: 使用的模型名称，推荐qwen-vl-max或qwen-vl-plus
        base_url: API基础URL
        output_dir: 输出目录
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API Key不能为空，请提供有效的DashScope API Key")
        
        # 修正：确保base_url没有多余空格
        self.base_url = base_url.strip()
        
        # 创建OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 小红书风格的系统提示词
        self.system_prompt = """你是一位拥有50万粉丝的小红书爆款笔记创作者，擅长用温暖治愈的语言捕捉生活中的美好瞬间。请为图片创作一篇高互动率的小红书笔记，要求：

1. 标题必须有吸引力，使用emoji和感叹号，如【XXX太戳了！✨】
2. 开头用"谁懂啊家人们！！"或类似亲切称呼，营造闺蜜聊天氛围
3. 正文描述3-5个画面细节，每个细节前用特殊符号(❶/❷/❸)标注
4. 使用至少4个相关话题标签，格式为#标签名
5. 结尾提供"配图tips"，说明拍摄技巧
6. 整体语气活泼、温暖、治愈，充满生活气息
7. 避免使用专业术语，像普通人分享惊喜发现一样自然
8. 适当使用网络流行语，但不过度

请确保描述准确反映图片内容，不要编造不存在的元素。"""
    
    def load_existing_results(self):
        """加载现有的结果文件"""
        result_path = f"{self.output_dir}/qwen_vl_descriptions.json"
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载现有结果失败: {str(e)}", exc_info=True)
        # 返回空结构
        return {
            "train": [],
            "val": [],
            "metadata": {
                "model": self.model_name,
                "system_prompt": self.system_prompt,
                "total_images": 0,
                "timestamp": 0,
                "api_key_used": self.api_key[:8] + "..." if self.api_key else "N/A"
            }
        }
    
    def get_processed_images(self, results=None):
        """获取已处理的图片路径列表"""
        if results is None:
            results = self.load_existing_results()
        
        processed_images = []
        if 'train' in results:
            processed_images.extend([item['image_path'] for item in results['train'] if 'image_path' in item])
        if 'val' in results:
            processed_images.extend([item['image_path'] for item in results['val'] if 'image_path' in item])
        return processed_images
    
    def _get_image_mime_type(self, image_path: str) -> str:
        """根据文件扩展名获取正确的MIME类型"""
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return "image/jpeg"
        elif ext == '.png':
            return "image/png"
        elif ext == '.webp':
            return "image/webp"
        else:
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type and mime_type.startswith('image/'):
                return mime_type
            return "image/jpeg"  # 默认使用JPEG
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=10))
    async def generate_description(self, image_path: str) -> dict:
        """
        为单张图片生成小红书风格描述
        """
        start_time = time.time()
        logger.info(f"开始处理图片: {image_path}")
        
        try:
            # 读取并编码图片
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 获取正确的MIME类型
            mime_type = self._get_image_mime_type(image_path)
            
            # 构建消息 - 严格遵循官方示例格式
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                # 修正：添加"data:"前缀，这是官方示例要求的
                                "url": f"data:{mime_type};base64,{encoded_string}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请为这张图片创作一篇小红书风格的笔记，要求符合系统提示中的所有要点。"
                        }
                    ]
                }
            ]
            
            logger.debug(f"调用API: model={self.model_name}, image_path={image_path}")
            
            # 使用OpenAI客户端调用API（严格遵循官方示例）
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            description = response.choices[0].message.content
            processing_time = time.time() - start_time
            logger.info(f"成功生成描述 (耗时: {processing_time:.2f}秒): {image_path}")
            
            # 提取关键信息
            return {
                "image_path": image_path,
                "raw_description": description,
                "title": self._extract_title(description),
                "hashtags": self._extract_hashtags(description),
                "key_details": self._extract_key_details(description),
                "processing_time": processing_time,
                "timestamp": time.time()
            }
        except Exception as e:
            error_msg = f"生成描述失败 {image_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            return {
                "image_path": image_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }
    
    def _extract_title(self, description: str) -> str:
        """从描述中提取标题"""
        if "【" in description and "】" in description:
            return description.split("【")[1].split("】")[0]
        return "小红书风格图片描述"
    
    def _extract_hashtags(self, description: str) -> list:
        """提取话题标签"""
        hashtags = []
        for word in description.split():
            if word.startswith("#"):
                hashtags.append(word[1:])
        return hashtags[:5]
    
    def _extract_key_details(self, description: str) -> list:
        """提取关键细节"""
        details = []
        import re
        pattern = r'[❶❷❸❹❺][^\n]+'
        matches = re.findall(pattern, description)
        for match in matches:
            detail = match[1:].strip()
            details.append(detail)
        return details[:5]
    
    async def process_directory(self, directory: str, batch_size: int = 2):
        """
        处理整个目录的图片，只处理未处理过的图片
        """
        # 获取现有结果和已处理的图片
        existing_results = self.load_existing_results()
        processed_images = self.get_processed_images(existing_results)
        
        image_paths = glob.glob(os.path.join(directory, "*.jpg")) + \
                     glob.glob(os.path.join(directory, "*.jpeg")) + \
                     glob.glob(os.path.join(directory, "*.png"))
        
        # 过滤掉已处理的图片
        new_image_paths = [img for img in image_paths if img not in processed_images]
        
        if not new_image_paths:
            logger.info(f"目录 {directory} 中所有图片已处理完毕")
            print(f"目录 {directory} 中所有图片已处理完毕")
            # 返回空列表，表示没有新图片需要处理
            return []
        
        logger.info(f"在目录 {directory} 中发现 {len(image_paths)} 张图片，{len(new_image_paths)} 张新图片需要处理")
        print(f"发现 {len(new_image_paths)} 张新图片需要处理...")
        
        descriptions = []
        for i in tqdm(range(0, len(new_image_paths), batch_size), desc="处理新图片"):
            batch = new_image_paths[i:i+batch_size]
            tasks = [self.generate_description(img_path) for img_path in batch]
            results = await asyncio.gather(*tasks)
            descriptions.extend(results)
            
            # 保存中间结果
            self._save_results(descriptions, f"{self.output_dir}/intermediate_results.json")
            
            # 添加请求间隔
            await asyncio.sleep(2.0)
        
        return descriptions
    
    def _save_results(self, results: list, output_path: str):
        """保存结果到JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(results)} 条结果到 {output_path}")
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}", exc_info=True)

    def save_final_results(self, train_results: list, val_results: list):
        """保存最终结果，合并新旧结果"""
        # 加载现有结果
        existing_results = self.load_existing_results()
        
        # 创建已处理图片的字典，用于快速查找
        existing_train_dict = {item['image_path']: item for item in existing_results['train'] if 'image_path' in item}
        existing_val_dict = {item['image_path']: item for item in existing_results['val'] if 'image_path' in item}
        
        # 将新结果添加到现有结果中（新结果优先）
        for item in train_results:
            if 'image_path' in item:
                existing_train_dict[item['image_path']] = item
        
        for item in val_results:
            if 'image_path' in item:
                existing_val_dict[item['image_path']] = item
        
        # 转换回列表
        train_results = list(existing_train_dict.values())
        val_results = list(existing_val_dict.values())
        
        # 更新元数据
        metadata = existing_results.get('metadata', {})
        metadata.update({
            "model": self.model_name,
            "system_prompt": self.system_prompt,
            "total_images": len(train_results) + len(val_results),
            "timestamp": time.time(),
            "api_key_used": self.api_key[:8] + "..." if self.api_key else "N/A"
        })
        
        final_results = {
            "train": train_results,
            "val": val_results,
            "metadata": metadata
        }
        
        output_path = f"{self.output_dir}/qwen_vl_descriptions.json"
        self._save_results(final_results, output_path)
        return output_path

async def main():
    # 创建日志目录
    os.makedirs("descriptions", exist_ok=True)
    
    print("="*50)
    print("Qwen-VL图片描述生成系统 - 增量处理版")
    print("注意: 本程序会自动跳过已处理的图片")
    print("="*50)
    
    # 检查API Key
    API_KEY = "sk-5f28127ea6524835b3304469e54d05f5"
    print(f"使用API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    
    try:
        # 初始化管道 - 使用正确的Qwen-VL模型
        pipeline = QwenVLPipeline(
            api_key=API_KEY,
            model_name="qwen-vl-plus",  # 使用qwen-vl-plus避免Arrearage错误
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 确保无空格
            output_dir="descriptions"
        )
        
        # 检查现有结果
        existing_results = pipeline.load_existing_results()
        processed_images = pipeline.get_processed_images(existing_results)
        total_processed = len(processed_images)
        
        print(f"\n✅ 使用模型: {pipeline.model_name}")
        print(f"✅ API Base URL: {pipeline.base_url}")
        print("✅ 已正确设置base_url（无末尾空格）")
        print("✅ 已正确设置image_url格式（包含data:前缀）")
        print(f"✅ 已有 {total_processed} 张图片的描述已生成")
        
        # 处理训练集
        print("\n===== 开始处理训练集图片 =====")
        train_results = await pipeline.process_directory("data/Train")
        
        # 处理验证集
        print("\n===== 开始处理验证集图片 =====")
        val_results = await pipeline.process_directory("data/Val")
        
        # 保存最终结果
        output_path = pipeline.save_final_results(train_results, val_results)
        print(f"\n✅ 所有描述已生成并保存至: {output_path}")
        
        # 统计结果
        existing_results = pipeline.load_existing_results()
        total = len(existing_results["train"]) + len(existing_results["val"])
        success = len([r for r in existing_results["train"] + existing_results["val"] if 'error' not in r])
        failed = total - success
        new_success = len([r for r in train_results + val_results if 'error' not in r])
        
        print(f"\n📊 处理统计:")
        print(f"总计: {total} 张图片")
        print(f"成功: {success} 张图片 ({success/total:.1%})")
        print(f"失败: {failed} 张图片 ({failed/total:.1%})")
        print(f"本次新增: {new_success} 张成功图片")
        print(f"日志已保存到: descriptions/pipeline.log")
        
        if failed > 0:
            print("\n⚠️  如果仍有失败，请检查:")
            print("1. 百炼平台是否已为API Key授权Qwen-VL模型权限")
            print("2. 图片是否过大（Qwen-VL单图最大支持16384 Token）")
            print("3. API Key是否有效")
    
    except Exception as e:
        logger.critical("程序初始化失败", exc_info=True)
        print(f"\n❌ 程序初始化失败: {str(e)}")
        print("\n请检查以下问题:")
        print("1. 是否在百炼平台(https://bailian.console.aliyun.com)创建了Qwen-VL应用?")
        print("2. 是否在'业务空间管理'中为API Key授权了Qwen-VL模型权限?")
        print("3. API Key是否正确?")
        print("4. base_url是否包含多余空格?")

if __name__ == "__main__":
    print("启动程序...")
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"程序运行出错: {str(e)}")
        traceback.print_exc()
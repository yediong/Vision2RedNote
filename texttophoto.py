import os
import time
import requests
from datetime import datetime

# 通义万相API配置 - 修正为正确的文生图服务端点
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
API_KEY = "sk-86aa4599b51049e3b665a32305735be0"  
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-DashScope-Async": "enable"  # 必须启用异步模式
}

# 路径配置
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

def generate_image(prompt: str):
    """调用API生成图片并保存到本地"""
    try:
        # 1. 创建生成任务
        task_id = create_task(prompt)
        if not task_id:
            raise Exception("任务创建失败")
        
        # 2. 轮询获取结果（最长等待60秒）
        image_url = poll_task_result(task_id, timeout=60)
        if not image_url:
            raise Exception("图片生成超时或失败")
        
        # 3. 下载并保存图片
        save_image(image_url, prompt)
        
        print(f"图片生成保存成功！文件路径: {os.path.join(OUTPUT_DIR, get_filename(prompt))}")
        return os.path.join(OUTPUT_DIR, get_filename(prompt))
    
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

def create_task(prompt: str) -> str:
    """提交文生图任务并返回task_id"""
    payload = {
        "model": "wanx2.1-t2i-turbo",  # 指定极速版模型
        "input": {
            "prompt": prompt,
            "negative_prompt": "模糊, 低质量, 水印"  # 反向提示词提升质量
        },
        "parameters": {
            "size": "1024*1024",  # 修正分辨率格式（移除空格）
            "n": 1,                # 生成数量
            "prompt_extend": True  # 开启智能提示词优化
        }
    }
    response = requests.post(API_URL, json=payload, headers=HEADERS)
    
    if response.status_code != 200:
        error_msg = response.json().get('message', '未知错误')
        print(f"API错误 [{response.status_code}]: {error_msg}")
        return None
    
    response_data = response.json()
    # 检查API返回是否包含有效task_id
    if "output" in response_data and "task_id" in response_data["output"]:
        return response_data["output"]["task_id"]
    else:
        print("API响应中未找到task_id:", response.text)
        return None

def poll_task_result(task_id: str, timeout=60) -> str:
    """轮询任务状态直到完成（增强错误处理）"""
    CHECK_URL = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"  
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(CHECK_URL, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"轮询错误 [{response.status_code}]: {response.text}")
            return None
            
        data = response.json()
        task_status = data["output"]["task_status"]
        
        if task_status == "SUCCEEDED":
            return data["output"]["results"][0]["url"]  # 返回图片URL
        
        # 增加对失败状态的检查
        if task_status in ["FAILED", "CANCELED"]:
            error_msg = data["output"].get("error", {}).get("message", "任务失败")
            print(f"任务执行失败 [{task_status}]: {error_msg}")
            return None
        
        time.sleep(2)  # 每2秒检查一次
    
    print("任务轮询超时")
    return None

def get_filename(prompt: str) -> str:
    """生成符合命名规则的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # 清洗提示词：取前10字符，仅保留字母数字，空值处理
    clean_prompt = "".join(c for c in prompt[:10] if c.isalnum()) or "image"
    return f"wanx_{timestamp}_{clean_prompt}.png"

def save_image(image_url: str, prompt: str):
    """下载图片并按规则命名保存"""
    filename = get_filename(prompt)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # 下载图片（增加重试机制）
    for _ in range(3):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            return
        except Exception as e:
            print(f"图片下载失败: {str(e)}，重试中...")
            time.sleep(1)
    
    raise Exception("图片下载多次失败")


def send_image(user_prompt: str):
    #style_user_prompt=user_prompt
    fileplace=generate_image(user_prompt)
    print(f"\n图片已生成，保存至: {os.path.abspath(OUTPUT_DIR)}")
    return fileplace

#!/usr/bin/env python3
"""
FLUX.2-dev 图生图测试脚本

使用方法:
    python test_img2img.py input.jpg "转换提示词"
    
示例:
    python test_img2img.py photo.jpg "Transform into Studio Ghibli animation style"
"""

import base64
import sys
from pathlib import Path

import requests


# API 配置
API_URL = "https://rodneycornwell--example-flux2-web.modal.run"
API_KEY = "longlikun"


def encode_image(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_text_to_image(prompt: str, output_path: str = "output_t2i.jpg"):
    """测试文生图功能"""
    print(f"🎨 测试文生图模式...")
    print(f"📝 Prompt: {prompt}")
    
    payload = {
        "api_key": API_KEY,
        "prompt": prompt,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        Path(output_path).write_bytes(response.content)
        print(f"✅ 文生图成功! 保存到: {output_path}")
        return True
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(response.text)
        return False


def test_image_to_image(
    input_image: str, 
    prompt: str, 
    output_path: str = "output_i2i.jpg"
):
    """测试图生图功能"""
    print(f"🎨 测试图生图模式...")
    print(f"🖼️  输入图片: {input_image}")
    print(f"📝 Prompt: {prompt}")
    
    # 检查输入文件是否存在
    if not Path(input_image).exists():
        print(f"❌ 输入图片不存在: {input_image}")
        return False
    
    # 编码图片
    print("⏳ 正在编码图片...")
    try:
        base64_image = encode_image(input_image)
        print(f"✅ 图片编码完成 (大小: {len(base64_image) / 1024:.2f} KB)")
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return False
    
    # 构建请求
    payload = {
        "api_key": API_KEY,
        "prompt": prompt,
        "input_images": [base64_image],
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
    }
    
    # 发送请求
    print("⏳ 正在发送请求到 Modal...")
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        Path(output_path).write_bytes(response.content)
        print(f"✅ 图生图成功! 保存到: {output_path}")
        return True
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(response.text)
        return False


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  文生图: python test_img2img.py --t2i \"提示词\"")
        print("  图生图: python test_img2img.py input.jpg \"转换提示词\"")
        print("\n示例:")
        print("  python test_img2img.py --t2i \"A futuristic cityscape\"")
        print("  python test_img2img.py photo.jpg \"Transform into anime style\"")
        sys.exit(1)
    
    # 文生图模式
    if sys.argv[1] == "--t2i":
        if len(sys.argv) < 3:
            print("❌ 请提供提示词")
            sys.exit(1)
        prompt = sys.argv[2]
        test_text_to_image(prompt)
    
    # 图生图模式
    else:
        if len(sys.argv) < 3:
            print("❌ 请提供输入图片和提示词")
            sys.exit(1)
        
        input_image = sys.argv[1]
        prompt = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "output_i2i.jpg"
        
        test_image_to_image(input_image, prompt, output_path)


if __name__ == "__main__":
    main()

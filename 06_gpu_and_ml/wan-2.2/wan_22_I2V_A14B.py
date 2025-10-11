# ---
# output-directory: "/tmp/wan22-i2v"
# ---

# # 使用 Modal 部署 Wan2.2-I2V-A14B 图片生成视频模型

# Wan2.2-I2V-A14B 是一个专门用于图片生成视频的 MoE 架构模型。
# 它采用双专家设计：高噪声专家处理早期阶段，低噪声专家处理细节精修。
# 支持 480P 和 720P 分辨率，总参数 27B（每步激活 14B）。

# 模型主页: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
# GitHub: https://github.com/Wan-Video/Wan2.2

from io import BytesIO
from pathlib import Path
from typing import Optional

import modal
from fastapi import File, Form, UploadFile
from fastapi.responses import Response


# 1. 定义容器镜像：安装所有必要的库
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "transformers==4.44.2",  # 稳定版本
        "diffusers>=0.34.0",  # 支持 Wan2.2
        "Pillow>=10.2.0",
        "huggingface-hub>=0.22.0",
        "accelerate>=0.29.0",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "fastapi",
        "python-multipart",
        "numpy",
        "imageio[ffmpeg]",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# 定义模型名称和缓存路径
MODEL_NAME = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
CACHE_DIR = Path("/cache")

# 创建持久化存储卷
cache_volume = modal.Volume.from_name("hf-hub-cache-wan22-i2v", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# HuggingFace API 密钥
secrets = [modal.Secret.from_name("huggingface-secret")]

app = modal.App("example-wan22-i2v-video-generation")

@app.cls(
    image=image,
    gpu="H200",  # I2V-A14B 需要 80GB 显存
    volumes=volumes,
    secrets=secrets,
    timeout=1800,
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def enter(self):
        """
        容器启动时运行一次：下载并加载模型到GPU。
        """
        import torch
        from diffusers import WanImageToVideoPipeline

        print(f"正在加载模型: {MODEL_NAME}")
        self.device = "cuda"
        self.dtype = torch.bfloat16

        # 加载 Image-to-Video Pipeline
        print("加载 Wan I2V Pipeline...")
        try:
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
            )
        except ValueError as e:
            print(f"常规加载失败，尝试兼容模式: {e}")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
        
        self.pipe.to(self.device)
        
        # 获取 VAE 缩放因子和 patch size
        self.vae_scale_factor = self.pipe.vae_scale_factor_spatial
        self.patch_size = self.pipe.transformer.config.patch_size[1]

        print("模型加载完成！")
        print(f"VAE 空间缩放因子: {self.vae_scale_factor}")
        print(f"Patch 大小: {self.patch_size}")

    def _calculate_video_size(self, image_width: int, image_height: int, max_area: int = 480 * 832):
        """
        根据输入图片尺寸和最大面积计算视频尺寸。
        保持原图比例，并确保尺寸是 vae_scale_factor * patch_size 的倍数。
        """
        import numpy as np
        
        aspect_ratio = image_height / image_width
        mod_value = self.vae_scale_factor * self.patch_size
        
        # 根据最大面积和宽高比计算尺寸
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        return width, height

    @modal.method()
    def generate_video(
        self,
        image_bytes: bytes,
        prompt: str,
        max_area: int = 480 * 832,  # 480P: 480*832, 720P: 720*1280
        num_frames: int = 81,  # 默认 81 帧（约 5 秒 @ 16fps）
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        seed: int = 0,
    ) -> bytes:
        """
        图片生成视频的核心函数。
        
        参数:
        - image_bytes: 输入图片的字节流（必填）
        - prompt: 文本提示词（描述视频内容）
        - max_area: 最大面积（480*832=480P, 720*1280=720P）
        - num_frames: 帧数（默认 81，约 5 秒 @ 16fps）
        - num_inference_steps: 推理步数（默认 40）
        - guidance_scale: 引导强度（默认 3.5）
        - seed: 随机种子
        
        返回:
        - 生成的视频文件（MP4格式）的字节流
        """
        import torch
        from PIL import Image
        from diffusers.utils import export_to_video
        import numpy as np

        print(f"收到图片生成视频任务")
        print(f"提示词: '{prompt}'")

        # 加载并处理输入图片
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        print(f"原始图片尺寸: {image.size}")

        # 计算视频尺寸（保持原图比例）
        width, height = self._calculate_video_size(image.width, image.height, max_area)
        print(f"调整后尺寸: {width}x{height}")
        
        # 调整图片大小
        image = image.resize((width, height))

        # 负向提示词（中文 + 英文）
        negative_prompt = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        # 设置生成器
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成视频
        print(f"开始生成视频... 分辨率: {width}x{height}, 帧数: {num_frames}")
        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]

        print(f"视频生成完成！帧数: {len(output)}")

        # 导出为 MP4（I2V 默认使用 16fps）
        print("正在导出视频...")
        video_path = "/tmp/i2v_output.mp4"
        export_to_video(output, video_path, fps=16)

        # 读取视频文件为字节流
        video_bytes = Path(video_path).read_bytes()
        print(f"视频文件大小: {len(video_bytes) / 1024 / 1024:.2f} MB")

        return video_bytes


@app.function(image=image, timeout=1800)
@modal.fastapi_endpoint(method="POST")
async def generate_video_api(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_area: int = Form(480 * 832),  # 480P 默认
    num_frames: int = Form(81),
    num_inference_steps: int = Form(40),
    guidance_scale: float = Form(3.5),
    seed: int = Form(0),
):
    """
    Web API 端点，用于通过 HTTP POST 请求生成视频。
    
    使用 multipart/form-data 格式：
    - image: 输入图片文件（必填）
    - prompt: 文本提示词（必填）
    - max_area: 最大面积（默认 480*832=480P，可设置 720*1280=720P）
    - num_frames: 帧数（默认 81）
    - num_inference_steps: 推理步数（默认 40）
    - guidance_scale: 引导强度（默认 3.5）
    - seed: 随机种子（默认 0）
    """
    print(f"收到来自 Web 的请求，提示词: '{prompt}'")

    # 读取上传的图片
    image_bytes = await image.read()
    print(f"收到输入图片，大小: {len(image_bytes)} bytes")

    # 远程调用核心生成函数
    video_bytes = Model().generate_video.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        max_area=max_area,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # 返回生成的视频
    return Response(content=video_bytes, media_type="video/mp4")


@app.local_entrypoint()
def main(
    image_path: str,
    prompt: str = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    output_path: str = "/tmp/wan22-i2v/output.mp4",
    resolution: str = "480p",  # 480p 或 720p
    num_frames: int = 81,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
    seed: int = 0,
):
    """
    本地入口函数：调用云端模型生成视频，保存结果。
    
    用法示例:
    1. 生成 480P 视频:
       modal run wan22_i2v_deploy.py --image-path ./cat.jpg \
           --prompt "一只猫在海滩上冲浪"
    
    2. 生成 720P 视频:
       modal run wan22_i2v_deploy.py --image-path ./cat.jpg \
           --prompt "一只猫在海滩上冲浪" \
           --resolution 720p
    """
    input_image_path = Path(image_path)
    output_video_path = Path(output_path)

    if not input_image_path.exists():
        print(f"错误：找不到输入图片 {input_image_path}")
        return

    # 根据分辨率设置最大面积
    max_area = 720 * 1280 if resolution.lower() == "720p" else 480 * 832
    resolution_name = "720P" if resolution.lower() == "720p" else "480P"

    print(f"🎬 正在读取输入图片: {input_image_path}")
    image_bytes = input_image_path.read_bytes()

    print(f"🎬 模式: 图片生成视频 (I2V)")
    print(f"🎬 分辨率: {resolution_name}")
    print(f"🎬 提示词: '{prompt}'")
    print(f"🎬 帧数: {num_frames} ({num_frames/16:.1f}秒 @ 16fps)")
    print(f"🎬 正在云端生成视频，这可能需要几分钟...")

    # 调用远程生成函数
    video_bytes = Model().generate_video.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        max_area=max_area,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # 保存视频
    output_video_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎬 正在保存视频到: {output_video_path}")
    output_video_path.write_bytes(video_bytes)
    print(f"✅ 完成！视频已保存到: {output_video_path}")
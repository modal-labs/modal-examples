# ---
# output-directory: "/tmp/wan22-video"
# ---

# # 使用 Modal 部署 Wan2.2-TI2V-5B 视频生成模型

# 在这个例子中，我们将在云端GPU上运行阿里的 Wan2.2-TI2V-5B 模型。
# 这是一个支持文本生成视频(T2V)和图片生成视频(I2V)的混合模型，
# 可以生成 720P@24fps 的高质量视频。

# 模型主页: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
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
    .apt_install("git", "ffmpeg")  # ffmpeg 用于视频处理
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "transformers==4.44.2",  # 固定一个稳定版本，避免 offload_state_dict 问题
        "diffusers>=0.34.0",  # 使用最新版本以支持 Wan2.2-VAE
        "Pillow>=10.2.0",
        "huggingface-hub>=0.22.0",
        "accelerate>=0.29.0",
        "sentencepiece",  # T5 模型需要
        "protobuf",
        "ftfy",  # WanPipeline 文本处理需要
        "fastapi",
        "python-multipart",
        "numpy",
        "imageio[ffmpeg]",  # 视频导出
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# 定义模型名称和缓存路径
MODEL_NAME = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
CACHE_DIR = Path("/cache")

# 创建一个持久化的存储卷来缓存模型
cache_volume = modal.Volume.from_name("hf-hub-cache-wan22", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# 从Modal平台安全地获取HuggingFace的API密钥
secrets = [modal.Secret.from_name("huggingface-secret")]

app = modal.App("example-wan22-ti2v-video-generation")

@app.cls(
    image=image,
    gpu="A100-80GB",  # Wan2.2-TI2V-5B 在 80GB GPU 上可以快速运行
    volumes=volumes,
    secrets=secrets,
    timeout=1800,  # 30分钟超时，视频生成需要较长时间
    scaledown_window=300,  # 修复：使用新的参数名
)
class Model:
    @modal.enter()
    def enter(self):
        """
        容器启动时运行一次：下载并加载模型到GPU。
        """
        import torch
        from diffusers import WanPipeline

        print(f"正在加载模型: {MODEL_NAME}")
        self.device = "cuda"
        self.dtype = torch.bfloat16

        # 直接加载完整的 Pipeline，让它自动处理 VAE
        print("加载 Wan Pipeline（包含 VAE）...")
        try:
            # 尝试正常加载
            self.pipe = WanPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
            )
        except ValueError as e:
            # 如果出现形状不匹配错误，使用低内存模式和忽略不匹配
            print(f"常规加载失败，尝试使用兼容模式: {e}")
            self.pipe = WanPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
        
        self.pipe.to(self.device)

        print("模型加载完成！")

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 42,
    ) -> bytes:
        """
        核心的视频生成函数。
        
        参数:
        - prompt: 文本提示词
        - image_bytes: 可选的输入图片（如果提供则为 I2V，否则为 T2V）
        - height: 视频高度（默认 704）
        - width: 视频宽度（默认 1280）
        - num_frames: 帧数（默认 121，约5秒@24fps）
        - num_inference_steps: 推理步数（默认 50）
        - guidance_scale: 引导强度（默认 5.0）
        - seed: 随机种子
        
        返回:
        - 生成的视频文件（MP4格式）的字节流
        """
        import torch
        from PIL import Image
        from diffusers.utils import export_to_video

        print(f"收到新的视频生成任务")
        print(f"提示词: '{prompt}'")
        print(f"模式: {'图片生成视频 (I2V)' if image_bytes else '文本生成视频 (T2V)'}")
        print(f"分辨率: {width}x{height}, 帧数: {num_frames}")

        # 负向提示词（中文 + 英文）
        negative_prompt = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        # 如果提供了图片，则进行 I2V 生成
        image = None
        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            print(f"输入图片尺寸: {image.size}")

        # 设置生成器
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成视频
        print("开始生成视频...")
# 替换你原来的 output = self.pipe(...) 部分

        if image_bytes:
            # 如果提供图片，则加载 I2V 模型
            from diffusers import WanI2VPipeline
            i2v_pipe = WanI2VPipeline.from_pretrained(
                MODEL_NAME.replace("TI2V", "I2V"),  # 通常 I2V 模型有独立权重
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
            ).to(self.device)

            output = i2v_pipe(
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
        else:
            # 否则使用 T2V pipeline
            output = self.pipe(
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

        # 导出为 MP4
        print("正在导出视频...")
        video_path = "/tmp/output_video.mp4"
        export_to_video(output, video_path, fps=24)

        # 读取视频文件为字节流
        video_bytes = Path(video_path).read_bytes()
        print(f"视频文件大小: {len(video_bytes) / 1024 / 1024:.2f} MB")

        return video_bytes


@app.function(image=image, timeout=1800)
@modal.fastapi_endpoint(method="POST")  # 修复：使用新的装饰器名称
async def generate_video_api(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    height: int = Form(704),
    width: int = Form(1280),
    num_frames: int = Form(121),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(5.0),
    seed: int = Form(42),
):
    """
    Web API 端点，用于通过 HTTP POST 请求生成视频。
    
    使用 multipart/form-data 格式：
    - prompt: 文本提示词（必填）
    - image: 可选的输入图片文件
    - height: 视频高度（默认 704）
    - width: 视频宽度（默认 1280）
    - num_frames: 帧数（默认 121）
    - num_inference_steps: 推理步数（默认 50）
    - guidance_scale: 引导强度（默认 5.0）
    - seed: 随机种子（默认 42）
    """
    print(f"收到来自 Web 的请求，提示词: '{prompt}'")

    # 读取上传的图片文件（如果有）
    image_bytes = None
    if image:
        image_bytes = await image.read()
        print(f"收到输入图片，大小: {len(image_bytes)} bytes")

    # 远程调用核心生成函数
    video_bytes = Model().generate_video.remote(
        prompt=prompt,
        image_bytes=image_bytes,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # 返回生成的视频
    return Response(content=video_bytes, media_type="video/mp4")


@app.local_entrypoint()
def main(
    prompt: str = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage",
    image_path: Optional[str] = None,
    output_path: str = "/tmp/wan22-video/output.mp4",
    height: int = 704,
    width: int = 1280,
    num_frames: int = 121,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    """
    本地入口函数：调用云端模型生成视频，保存结果。
    
    用法示例:
    1. 文本生成视频 (T2V):
       modal run wan22_deploy.py --prompt "一只可爱的熊猫在竹林里玩耍"
    
    2. 图片生成视频 (I2V):
       modal run wan22_deploy.py --prompt "这只猫在海滩上冲浪" --image-path ./cat.jpg
    """
    output_video_path = Path(output_path)

    # 读取输入图片（如果提供）
    image_bytes = None
    if image_path:
        input_image_path = Path(image_path)
        if not input_image_path.exists():
            print(f"错误：找不到输入图片 {input_image_path}")
            return
        print(f"🎬 正在读取输入图片: {input_image_path}")
        image_bytes = input_image_path.read_bytes()

    mode = "图片生成视频 (I2V)" if image_bytes else "文本生成视频 (T2V)"
    print(f"🎬 模式: {mode}")
    print(f"🎬 提示词: '{prompt}'")
    print(f"🎬 分辨率: {width}x{height}")
    print(f"🎬 帧数: {num_frames} ({num_frames/24:.1f}秒 @ 24fps)")
    print(f"🎬 正在云端生成视频，这可能需要几分钟...")

    # 调用远程生成函数
    video_bytes = Model().generate_video.remote(
        prompt=prompt,
        image_bytes=image_bytes,
        height=height,
        width=width,
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
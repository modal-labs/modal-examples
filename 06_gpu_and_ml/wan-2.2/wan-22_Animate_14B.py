# ---
# output-directory: "/tmp/wan22-video"
# ---

# # 使用 Modal 部署 Wan2.2-Animate-14B 视频生成模型

# 在这个例子中，我们将在云端GPU上运行阿里的 Wan2.2-Animate-14B 模型。
# 这是一个文本生成视频 (T2V) 动画模型，
# 可以生成 720P@24fps 的高质量视频。

# 模型主页: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
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
        "Pillow>=10.2.0",
        "huggingface-hub>=0.22.0",
        "accelerate>=0.29.0",
        "sentencepiece",  # T5 模型需要
        "protobuf",
        "ftfy",  # WanPipeline 文本处理需要
        "fastapi",
        "python-multipart",
        "numpy",
        "decord",
        "opencv-python",
        "imageio[ffmpeg]",  # 视频导出
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2",
        "cd /opt/Wan2.2 && pip install -r requirements.txt"
    )
)

# 定义 Wan2.2 仓库路径
WAN_PATH = Path("/opt/Wan2.2")

# 创建一个持久化的存储卷来缓存模型
cache_volume = modal.Volume.from_name("hf-hub-cache-wan22", create_if_missing=True)
volumes = {Path("/cache"): cache_volume}

# 从Modal平台安全地获取HuggingFace的API密钥
secrets = [modal.Secret.from_name("huggingface-secret")]

app = modal.App("example-wan22-animate")

@app.cls(
    image=image,
    gpu="H100",  # Wan2.2-Animate-14B 建议使用 80GB H100 GPU
    volumes=volumes,
    secrets=secrets,
    timeout=3600,  # 60分钟超时，视频生成需要较长时间
    scaledown_window=300,  # 修复：使用新的参数名
)
class Model:
    @modal.enter()
    def enter(self):
        """
        容器启动时运行一次:下载并加载模型到GPU。
        """
        import torch
        import sys
        sys.path.append(str(WAN_PATH))
        from wan2_2.models import WanModel
        from wan2_2.pipeline import WanAnimatePipeline

        print("加载 Wan2.2 Animate 14B 模型...")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.pipe = WanAnimatePipeline.from_pretrained(
            WAN_PATH / "models/Wan2.2-Animate-14B",
            torch_dtype=self.dtype,
            device=self.device
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
        import numpy as np

        print(f"收到新的视频生成任务")
        print(f"提示词: '{prompt}'")
        print(f"模式: {'图片生成视频 (I2V)' if image_bytes else '文本生成视频 (T2V)'}")

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
            print(f"输入图片原始尺寸: {image.size}")
            
            # 对于 I2V，需要根据图片比例调整尺寸
            # TI2V-5B 的 size 参数表示面积，宽高比跟随输入图片
            max_area = height * width  # 使用传入的 height 和 width 计算面积
            aspect_ratio = image.height / image.width
            
            # 计算合适的尺寸（必须是 32 的倍数，因为 patch_size=2, scale_factor=16）
            mod_value = 32
            calc_height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            calc_width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            
            # 调整图片大小
            image = image.resize((calc_width, calc_height))
            print(f"调整后图片尺寸: {image.size}")
            
            # 使用调整后的尺寸
            height = calc_height
            width = calc_width

        print(f"视频分辨率: {width}x{height}, 帧数: {num_frames}")

        # 设置生成器
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成视频
        print("开始生成视频...")
        output = self.pipe(
            prompt=prompt,
            image=image,  # None 表示 T2V，有值表示 I2V
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
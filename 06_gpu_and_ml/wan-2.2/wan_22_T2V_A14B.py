# ---
# output-directory: "/tmp/wan22-t2v-a14b"
# ---

# # 使用 Modal 部署 Wan2.2-T2V-A14B 文字生成视频模型

# Wan2.2-T2V-A14B 是专业的文本生成视频 MoE 模型。
# 采用双专家 MoE 架构：总参数 27B，每步激活 14B。
# 支持 720P 高清视频生成，生成质量达到商业级水平。

# 模型主页: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers
# GitHub: https://github.com/Wan-Video/Wan2.2

from io import BytesIO
from pathlib import Path

import modal
from fastapi import Form
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
MODEL_NAME = "Wan-AI/Wan2.2-Animate-14B"
CACHE_DIR = Path("/cache")

# 创建持久化存储卷
cache_volume = modal.Volume.from_name("hf-hub-cache-wan22-t2v-a14b", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# HuggingFace API 密钥
secrets = [modal.Secret.from_name("huggingface-secret")]

app = modal.App("example-wan22-t2v-a14b-video-generation")

@app.cls(
    image=image,
    gpu="H100",  # H100 是最佳选择：更快的速度 + 80GB 显存
    volumes=volumes,
    secrets=secrets,
    timeout=2400,
    scaledown_window=300,
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

        # 加载 Text-to-Video Pipeline
        print("加载 Wan T2V Pipeline...")
        try:
            self.pipe = WanPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
            )
        except ValueError as e:
            print(f"常规加载失败，尝试兼容模式: {e}")
            self.pipe = WanPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
        
        self.pipe.to(self.device)

        print("模型加载完成！")
        print(f"模型: Wan2.2-T2V-A14B (MoE 27B/14B激活)")

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,  # 默认 81 帧（约 5 秒 @ 16fps）
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 42,
    ) -> bytes:
        """
        文本生成视频的核心函数。
        
        参数:
        - prompt: 文本提示词（必填）
        - height: 视频高度（默认 720）
        - width: 视频宽度（默认 1280，720P）
        - num_frames: 帧数（默认 81，约 5 秒 @ 16fps）
        - num_inference_steps: 推理步数（默认 50）
        - guidance_scale: 引导强度（默认 5.0）
        - seed: 随机种子
        
        返回:
        - 生成的视频文件（MP4格式）的字节流
        """
        import torch
        from diffusers.utils import export_to_video

        print(f"收到文本生成视频任务")
        print(f"提示词: '{prompt}'")
        print(f"分辨率: {width}x{height}")
        print(f"帧数: {num_frames}")

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
        print("开始生成视频...")
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

        # 导出为 MP4（T2V 使用 16fps）
        print("正在导出视频...")
        video_path = "/tmp/t2v_output.mp4"
        export_to_video(output, video_path, fps=16)

        # 读取视频文件为字节流
        video_bytes = Path(video_path).read_bytes()
        print(f"视频文件大小: {len(video_bytes) / 1024 / 1024:.2f} MB")

        return video_bytes


@app.function(image=image, timeout=1800)
@modal.fastapi_endpoint(method="POST")
async def generate_video_api(
    prompt: str = Form(...),
    height: int = Form(720),
    width: int = Form(1280),
    num_frames: int = Form(81),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(5.0),
    seed: int = Form(42),
):
    """
    Web API 端点，用于通过 HTTP POST 请求生成视频。
    
    使用 application/x-www-form-urlencoded 或 multipart/form-data：
    - prompt: 文本提示词（必填）
    - height: 视频高度（默认 720）
    - width: 视频宽度（默认 1280）
    - num_frames: 帧数（默认 81）
    - num_inference_steps: 推理步数（默认 50）
    - guidance_scale: 引导强度（默认 5.0）
    - seed: 随机种子（默认 42）
    """
    print(f"收到来自 Web 的请求，提示词: '{prompt}'")

    # 远程调用核心生成函数
    video_bytes = Model().generate_video.remote(
        prompt=prompt,
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
    output_path: str = "/tmp/wan22-t2v-a14b/output.mp4",
    height: int = 720,
    width: int = 1280,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    """
    本地入口函数：调用云端模型生成视频，保存结果。
    
    用法示例:
    1. 基础使用:
       modal run wan22_t2v_a14b_deploy.py \
           --prompt "一只可爱的熊猫在竹林里玩耍"
    
    2. 自定义参数:
       modal run wan22_t2v_a14b_deploy.py \
           --prompt "日落时分，城市街道上车水马龙" \
           --num-frames 121 \
           --num-inference-steps 60 \
           --guidance-scale 6.0
    
    3. 高质量长视频:
       modal run wan22_t2v_a14b_deploy.py \
           --prompt "壮丽的山川风景，云雾缭绕" \
           --num-frames 161 \
           --num-inference-steps 80
    """
    output_video_path = Path(output_path)

    print(f"🎬 模式: 文本生成视频 (T2V)")
    print(f"🎬 模型: Wan2.2-T2V-A14B (MoE 27B/14B)")
    print(f"🎬 提示词: '{prompt}'")
    print(f"🎬 分辨率: {width}x{height} (720P)")
    print(f"🎬 帧数: {num_frames} ({num_frames/16:.1f}秒 @ 16fps)")
    print(f"🎬 推理步数: {num_inference_steps}")
    print(f"🎬 引导强度: {guidance_scale}")
    print(f"🎬 正在云端生成视频，这可能需要几分钟...")

    # 调用远程生成函数
    video_bytes = Model().generate_video.remote(
        prompt=prompt,
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
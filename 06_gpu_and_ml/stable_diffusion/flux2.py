# ---
# output-directory: "/tmp/flux2"
# args: ["--prompt", "A cinematic photo of a baby penguin"]
# ---

# # Run Flux2 (FLUX.2-dev) for high-quality image generation

# 本示例展示了如何在 Modal 上使用 Black Forest Labs 的 FLUX.2-dev 模型生成高质量图片。
# FLUX.2 是新一代的图像生成模型，提供了更好的图片质量、一致性和控制能力。

# ## 设置镜像和依赖

import base64
import time
from io import BytesIO
from pathlib import Path
from pydantic import BaseModel

import modal
from modal import fastapi_endpoint
from fastapi.responses import Response, JSONResponse

# 我们使用完整的 CUDA 工具包来构建容器镜像

cuda_version = "12.4.0"  # 不应大于主机 CUDA 版本
flavor = "devel"  # 包含完整的 CUDA 工具包
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

# 安装依赖。Flux2 需要最新版本的 diffusers 库来支持 Flux2Pipeline
# 我们从 GitHub 主分支安装以获得最新的 Flux2 支持

flux2_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "fastapi[standard]",
        "invisible_watermark==0.2.0",
        "transformers>=4.47.0",
        "huggingface_hub[hf_transfer]>=0.34.0",
        "hf-transfer",
        "accelerate>=0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        "git+https://github.com/huggingface/diffusers.git@main",  # 使用主分支以支持 Flux2
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

# 配置 torch.compile 缓存以加快后续容器的编译速度

flux2_image = flux2_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)


class ImageRequest(BaseModel):
    """图片生成请求模型"""
    api_key: str  # API 密钥，必填
    prompt: str = "A cinematic photo of a baby penguin"
    input_images: list[str] | None = None  # 可选：base64 编码的输入图片列表，用于图生图
    width: int = 1024  # 宽度，默认 1024
    height: int = 1024  # 高度，默认 1024
    num_inference_steps: int = 28  # 推理步数，默认 28（可选范围：28-50）
    guidance_scale: float = 4.0  # 引导强度，默认 4.0


# 构建 Modal App，设置默认镜像，并导入 Flux2Pipeline

app = modal.App("example-flux2", image=flux2_image)

with flux2_image.imports():
    import torch
    from diffusers import Flux2Pipeline
    from diffusers.utils import load_image
    from PIL import Image

# ## 定义参数化的 Model 推理类

# 1. 使用 @modal.enter() 装饰的方法运行模型设置，包括加载权重并移至 GPU
# 2. 使用 @modal.method() 装饰的方法运行实际推理

# *注意: 访问 Hugging Face 上的 FLUX.2-dev 模型需要同意许可协议。
# 请在 https://huggingface.co/black-forest-labs/FLUX.2-dev 接受许可后，
# 创建名为 `huggingface-secret` 的 Modal Secret。*

MINUTES = 60  # 秒
MODEL_ID = "black-forest-labs/FLUX.2-dev"
NUM_INFERENCE_STEPS = 50  # 推理步数（官方推荐50步，28步可作为速度与质量的折衷）
GUIDANCE_SCALE = 4.0  # 引导强度（Flux2 推荐值）


@app.cls(
    gpu="h200",  # 使用 H200 GPU (141GB)，为 FLUX.2-dev 提供最大内存和最强性能
    scaledown_window=20 * MINUTES,
    timeout=60 * MINUTES,  # 为编译留出充足的时间
    volumes={  # 添加 Volumes 以存储可序列化的编译工件
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Model:
    compile: bool = (  # 是否使用 torch.compile 优化
        modal.parameter(default=False)
    )

    @modal.enter()
    def enter(self):
        """初始化模型"""
        print(f"🔥 Loading FLUX.2-dev model from {MODEL_ID}...")
        pipe = Flux2Pipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16
        ).to("cuda")  # 将模型移至 GPU
        
        self.pipe = optimize(pipe, compile=self.compile)
        print("✅ Model loaded successfully!")

    def _generate_image(
        self, 
        prompt: str,
        input_images: list[str] | None = None,
        width: int = 1024, 
        height: int = 1024,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
    ) -> bytes:
        """内部图像生成方法，支持文生图和图生图"""
        mode = "图生图" if input_images else "文生图"
        print(f"🎨 模式: {mode}")
        print(f"🎨 Generating image with size {width}x{height}, steps={num_inference_steps}, guidance={guidance_scale}...")
        print(f"📝 Prompt: {prompt}")
        
        # 处理输入图片（如果提供）
        decoded_images = None
        if input_images:
            print(f"🖼️  Processing {len(input_images)} input image(s)...")
            decoded_images = []
            for i, base64_str in enumerate(input_images):
                try:
                    # 解码 base64 → bytes → PIL Image（全程在内存中）
                    image_data = base64.b64decode(base64_str)
                    pil_image = Image.open(BytesIO(image_data))
                    decoded_images.append(pil_image)
                    print(f"  ✅ Image {i+1}: {pil_image.size} ({pil_image.mode})")
                except Exception as e:
                    print(f"  ❌ Failed to decode image {i+1}: {e}")
                    raise ValueError(f"Invalid base64 image at index {i}")
        
        # 准备 pipeline 参数
        pipe_kwargs = {
            "prompt": prompt,
            "output_type": "pil",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": torch.Generator(device="cuda").manual_seed(42),
        }
        
        # 如果有输入图片，添加到参数中
        if decoded_images:
            pipe_kwargs["image"] = decoded_images
        
        # 执行生成
        out = self.pipe(**pipe_kwargs).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()

    @modal.method()
    def inference(
        self, 
        prompt: str,
        input_images: list[str] | None = None,
        width: int = 1024, 
        height: int = 1024,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
    ) -> bytes:
        """供 modal run 和 API 调用的推理方法"""
        return self._generate_image(prompt, input_images, width, height, num_inference_steps, guidance_scale)


# ## Web API 端点

# API 密钥（硬编码）
API_KEY = "longlikun"

@app.function()
@modal.fastapi_endpoint(method="POST")
def web(request: ImageRequest):
    """公共 API 端点，接收 POST 请求生成图片（支持文生图和图生图）"""
    # 验证 API 密钥
    if request.api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid API key", "message": "请提供正确的 API 密钥"}
        )
    
    image_bytes = Model().inference.remote(
        request.prompt,
        request.input_images,
        request.width,
        request.height,
        request.num_inference_steps,
        request.guidance_scale,
    )
    return Response(content=image_bytes, media_type="image/jpeg")


# ## 命令行入口点

# 使用以下命令运行：
# ```bash
# modal run flux2.py
# ```
#
# 可选参数：
# - --prompt: 提示词（默认："A cinematic photo of a baby penguin"）
# - --width: 图片宽度（默认：1024）
# - --height: 图片高度（默认：1024）
# - --num-inference-steps: 推理步数（默认：28）
# - --guidance-scale: 引导强度（默认：4.0）
# - --compile: 使用 torch.compile 优化（默认：False）


@app.local_entrypoint()
def main(
    prompt: str = "A cinematic photo of a baby penguin playing with colorful blocks, soft lighting, shallow depth of field",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    compile: bool = False,
):
    """本地命令行入口点"""
    print(f"🚀 Starting Flux2 image generation...")
    print(f"📝 Prompt: {prompt}")
    print(f"📐 Size: {width}x{height}")
    print(f"🔢 Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(
        prompt, width, height, num_inference_steps, guidance_scale
    )
    latency = time.time() - t0
    print(f"⚡ Inference latency: {latency:.2f} seconds")

    output_path = Path("/tmp") / "flux2" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"💾 Saving output to {output_path}")
    output_path.write_bytes(image_bytes)
    print(f"✅ Image saved successfully!")


# ## 使用 torch.compile 加速 Flux2

# PyTorch 2 添加了编译器来优化 PyTorch 执行期间动态创建的计算图。
# 这有助于缩小与 TensorRT 和 TensorFlow 等静态图框架的性能差距。

# 编译在首次迭代时可能需要长达 20 分钟。
# 我们缓存来自 nvcc、triton 和 inductor 的编译输出，
# 这可以将编译时间减少一个数量级。

# 使用以下命令启用编译：
# ```bash
# modal run flux2.py --compile
# ```


def optimize(pipe, compile=False):
    """优化 pipeline 以提高推理速度"""
    # 融合 Transformer 和 VAE 中的 QKV 投影
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # 切换内存布局为 Torch 首选的 channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # 设置 torch compile 标志
    config = torch._inductor.config
    config.disable_progress = False  # 显示进度条
    config.conv_1x1_as_mm = True  # 将 1x1 卷积视为矩阵乘法
    # 调整自动调优算法
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # 不要将逐点操作融合到矩阵乘法中

    # 标记计算密集型模块（Transformer 和 VAE decoder）进行编译
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # 触发 torch 编译
    print("🔦 Running torch compilation (may take up to 20 minutes)...")

    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]

    print("🔦 Finished torch compilation")

    return pipe

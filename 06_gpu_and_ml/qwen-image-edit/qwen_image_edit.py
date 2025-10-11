# ---
# output-directory: "/tmp/qwen-image-edit"
# ---

# # 使用 Modal 部署千问图片编辑模型 (Qwen-Image-Edit) - Diffusers 最终版

# 在这个例子中，我们将在云端GPU上运行阿里的 Qwen-Image-Edit 模型。
# 我们将使用 Hugging Face 官方推荐的 diffusers 库中的 QwenImageEditPipeline，
# 这种方法更稳定、代码也更简洁。

# **新增功能**: 我们还添加了一个 Web API 端点，以便通过 HTTP POST 请求调用此功能。

# 模型主页: https://huggingface.co/Qwen/Qwen-Image-Edit

from io import BytesIO
from pathlib import Path

import modal
from fastapi import File, Form, UploadFile
from fastapi.responses import Response


# 1. 定义容器镜像：安装所有必要的库
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", # 使用一个兼容的CUDA版本
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install(
        # 使用 PyTorch Nightly 版本以支持 diffusers 的最新功能
        "torch",
        "torchvision",
        "transformers>=4.52.0",
        "diffusers>=0.27.0",
        "Pillow>=10.2.0",
        "huggingface-hub>=0.22.0",
        "accelerate>=0.29.0",
        "fastapi",             # **新增**: 添加 FastAPI 用于构建 Web 端点
        "python-multipart",    # **新增**: 用于处理文件上传
        # PyTorch 的 Nightly index URL 不同，需要指定
        extra_index_url="https://download.pytorch.org/whl/nightly/cu121",
    )
)

# 定义模型名称和缓存路径
MODEL_NAME = "Qwen/Qwen-Image-Edit"
CACHE_DIR = Path("/cache")

# 创建一个持久化的存储卷来缓存模型，避免每次启动都重新下载
cache_volume = modal.Volume.from_name("hf-hub-cache-qwen", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# 从Modal平台安全地获取HuggingFace的API密钥
secrets = [modal.Secret.from_name("huggingface-secret")]

# 重新命名App以示区分
app = modal.App("example-qwen-image-edit-diffusers")

@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=240,
)
class Model:
    @modal.enter()
    def enter(self):
        """
        容器启动时运行一次：下载并加载模型到GPU。
        """
        import torch
        from diffusers import QwenImageEditPipeline

        print(f"正在加载模型: {MODEL_NAME}")
        self.device = "cuda"

        # 使用 Diffusers Pipeline 以高精度加载模型
        self.pipe = QwenImageEditPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        ).to(self.device)
        
        print("模型加载完成！")

    @modal.method()
    def inference(self, image_bytes: bytes, prompt: str) -> bytes:
        """
        这是核心的推理函数，从本地接收图片和指令，在云端GPU上执行，并返回新图片。
        """
        import torch
        from PIL import Image

        print(f"收到新的推理任务，指令: '{prompt}'")
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
                # --- 新增：定义负向提示词 ---
        negative_prompt = (
            "blurry text, distorted text, artifacts, watermark, signature, "
            "模糊的文字, 变形的文字, 乱码, 错误的笔画, low quality, "
            "乱序文字, 不自然的字符, 错误的字体"
        )

        edited_image = self.pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=8.0,
            generator=torch.Generator(device=self.device).manual_seed(42),
        ).images[0]
        
        print("推理完成，正在返回图片。")

        byte_stream = BytesIO()
        edited_image.save(byte_stream, format="PNG")
        output_image_bytes = byte_stream.getvalue()

        return output_image_bytes


# 修复：添加 @app.function() 装饰器
@app.function(image=image)  # ✅ 关键修复：关联到 app
@modal.fastapi_endpoint(method="POST")
async def edit_image(image: UploadFile = File(...), prompt: str = Form(...)):
    """
    一个用于编辑图片的 Web API 端点，使用 multipart/form-data 进行 POST 请求。
    - 'image': 需要编辑的图片文件。
    - 'prompt': 用于编辑的文本指令。
    """
    print(f"收到来自 Web 的请求，指令: '{prompt}'")
    # 读取上传的图片文件为字节流
    image_bytes = await image.read()

    # 远程调用我们的核心推理函数
    output_image_bytes = Model().inference.remote(image_bytes, prompt)

    # 将生成的图片以 PNG 格式返回
    return Response(content=output_image_bytes, media_type="image/png")


@app.local_entrypoint()
def main(
    image_path: str = str(Path(__file__).parent.parent / "demo_images/dog.png"),
    output_path: str = "/tmp/qwen-image-edit/output.png",
    prompt: str = "把它变成一只熊猫",
):
    """
    本地入口函数：读取本地图片，调用云端模型，保存结果。
    """
    input_image_path = Path(image_path)
    output_image_path = Path(output_path)
    
    if not input_image_path.exists():
        print(f"错误：找不到输入图片 {input_image_path}。")
        print("请确保在当前目录下有一张名为 'dog.png' 的图片，或者通过 --image-path 指定路径。")
        return

    print(f"🎨 正在读取输入图片: {input_image_path}")
    input_image_bytes = input_image_path.read_bytes()

    print(f"🎨 正在使用指令 '{prompt}' 编辑图片...")
    output_image_bytes = Model().inference.remote(image_bytes, prompt)

    output_image_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 正在保存输出图片到: {output_image_path}")
    output_image_path.write_bytes(output_image_bytes)
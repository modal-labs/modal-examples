# ---
# output-directory: "/tmp/wan22-animate"
# ---

# # 使用 Modal 部署 Wan2.2-Animate-14B 角色动画和替换模型

# Wan2.2-Animate-14B 是一个统一的角色动画和替换模型。
# 
# 功能：
# 1. Animation 模式：让静态角色图片按照参考视频的动作动起来
# 2. Replacement 模式：将视频中的角色替换成指定角色
#
# ⚠️  注意：此模型需要使用 GitHub 原始代码，不支持 Diffusers

# 模型主页: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
# GitHub: https://github.com/Wan-Video/Wan2.2
# 项目页面: https://humanaigc.github.io/wan-animate

from pathlib import Path
from typing import Optional, Literal
import modal

# 1. 定义容器镜像：克隆 GitHub 仓库并安装依赖
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("pip==24.0")
    # 先安装 PyTorch（flash_attn 编译需要）
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        # 克隆 Wan2.2 GitHub 仓库
        "cd /root && git clone https://github.com/Wan-Video/Wan2.2.git",
    )
    # 单独安装除了 flash_attn 之外的依赖（flash_attn 编译太慢且容易失败）
    .pip_install(
        "transformers>=4.44.0",
        "diffusers>=0.30.0",
        "accelerate>=0.29.0",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "Pillow>=10.2.0",
        "numpy",
        "opencv-python",
        "imageio[ffmpeg]",
        "einops",
        "omegaconf",
        "safetensors",
        "huggingface-hub",
        # 预处理需要的库
        "mediapipe",
        "insightface",
        "onnxruntime-gpu",
    )
    # 跳过 flash_attn，它编译太慢且不是必需的
    # 如果真的需要，可以用预编译版本或在运行时使用 scaled_dot_product_attention
)

# 定义模型路径和缓存
MODEL_NAME = "Wan-AI/Wan2.2-Animate-14B"
CACHE_DIR = Path("/cache")
REPO_DIR = Path("/root/Wan2.2")

# 创建持久化存储卷
cache_volume = modal.Volume.from_name("hf-hub-cache-wan22-animate", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# HuggingFace API 密钥
secrets = [modal.Secret.from_name("huggingface-secret")]

app = modal.App("example-wan22-animate-character-animation")


@app.cls(
    image=image,
    gpu="H100",  # Animate 需要大显存
    volumes=volumes,
    secrets=secrets,
    timeout=3600,  # 1小时超时（预处理 + 推理需要较长时间）
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def enter(self):
        """
        容器启动时运行一次：下载模型权重
        """
        import os
        import subprocess
        
        print(f"正在下载模型: {MODEL_NAME}")
        
        # 设置工作目录
        os.chdir(REPO_DIR)
        
        # 下载模型权重到缓存目录
        model_path = CACHE_DIR / "Wan2.2-Animate-14B"
        if not model_path.exists():
            print("首次运行，正在下载模型权重...")
            subprocess.run([
                "huggingface-cli", "download",
                MODEL_NAME,
                "--local-dir", str(model_path)
            ], check=True)
        else:
            print("模型已缓存，跳过下载")
        
        self.model_path = model_path
        print("模型准备完成！")

    @modal.method()
    def preprocess(
        self,
        video_bytes: bytes,
        image_bytes: bytes,
        mode: Literal["animation", "replacement"] = "animation",
        resolution_width: int = 1280,
        resolution_height: int = 720,
    ) -> bytes:
        """
        预处理步骤：处理输入视频和参考图片
        
        参数:
        - video_bytes: 输入视频的字节流
        - image_bytes: 参考角色图片的字节流
        - mode: "animation" 或 "replacement"
        - resolution_width: 视频宽度（默认 1280）
        - resolution_height: 视频高度（默认 720）
        
        返回:
        - 预处理结果的打包字节流
        """
        import os
        import subprocess
        import tarfile
        from io import BytesIO
        
        print(f"开始预处理 - 模式: {mode}")
        
        # 创建临时目录
        temp_dir = Path("/tmp/animate_input")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存输入文件
        video_path = temp_dir / "video.mp4"
        image_path = temp_dir / "image.jpeg"
        video_path.write_bytes(video_bytes)
        image_path.write_bytes(image_bytes)
        
        # 预处理输出目录
        output_dir = temp_dir / "process_results"
        output_dir.mkdir(exist_ok=True)
        
        # 构建预处理命令
        os.chdir(REPO_DIR)
        
        cmd = [
            "python", "./wan/modules/animate/preprocess/preprocess_data.py",
            "--ckpt_path", str(self.model_path / "process_checkpoint"),
            "--video_path", str(video_path),
            "--refer_path", str(image_path),
            "--save_path", str(output_dir),
            "--resolution_area", str(resolution_width), str(resolution_height),
        ]
        
        # 根据模式添加特定参数
        if mode == "animation":
            cmd.extend(["--retarget_flag", "--use_flux"])
        else:  # replacement
            cmd.extend([
                "--iterations", "3",
                "--k", "7",
                "--w_len", "1",
                "--h_len", "1",
                "--replace_flag"
            ])
        
        print(f"运行预处理命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # 打包预处理结果
        print("打包预处理结果...")
        tar_buffer = BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(output_dir, arcname='process_results')
        
        result_bytes = tar_buffer.getvalue()
        print(f"预处理完成，结果大小: {len(result_bytes) / 1024 / 1024:.2f} MB")
        
        return result_bytes

    @modal.method()
    def generate(
        self,
        preprocessed_bytes: bytes,
        mode: Literal["animation", "replacement"] = "animation",
        use_multi_gpu: bool = False,
    ) -> bytes:
        """
        生成步骤：使用预处理结果生成最终视频
        
        参数:
        - preprocessed_bytes: 预处理结果的打包字节流
        - mode: "animation" 或 "replacement"
        - use_multi_gpu: 是否使用多GPU（当前单GPU部署设为 False）
        
        返回:
        - 生成的视频字节流
        """
        import os
        import subprocess
        import tarfile
        from io import BytesIO
        
        print(f"开始生成视频 - 模式: {mode}")
        
        # 解压预处理结果
        temp_dir = Path("/tmp/animate_generate")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        tar_buffer = BytesIO(preprocessed_bytes)
        with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
            tar.extractall(temp_dir)
        
        process_results_dir = temp_dir / "process_results"
        
        # 构建生成命令
        os.chdir(REPO_DIR)
        
        cmd = [
            "python", "generate.py",
            "--task", "animate-14B",
            "--ckpt_dir", str(self.model_path),
            "--src_root_path", str(process_results_dir),
            "--refert_num", "1",
        ]
        
        # 添加模式特定参数
        if mode == "replacement":
            cmd.extend(["--replace_flag", "--use_relighting_lora"])
        
        print(f"运行生成命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # 查找生成的视频文件
        output_video = None
        for video_file in process_results_dir.glob("**/*.mp4"):
            if "output" in video_file.name.lower() or "result" in video_file.name.lower():
                output_video = video_file
                break
        
        if not output_video:
            # 如果没找到，使用第一个 mp4 文件
            output_video = next(process_results_dir.glob("**/*.mp4"), None)
        
        if not output_video:
            raise FileNotFoundError("未找到生成的视频文件")
        
        print(f"找到生成视频: {output_video}")
        video_bytes = output_video.read_bytes()
        print(f"视频大小: {len(video_bytes) / 1024 / 1024:.2f} MB")
        
        return video_bytes


@app.local_entrypoint()
def main(
    video_path: str,
    image_path: str,
    mode: str = "animation",
    output_path: str = "/tmp/wan22-animate/output.mp4",
    resolution_width: int = 1280,
    resolution_height: int = 720,
):
    """
    本地入口函数：完整的角色动画/替换流程
    
    用法示例:
    
    1. Animation 模式（让静态角色动起来）:
       modal run wan22_animate_deploy.py \
           --video-path ./dance_video.mp4 \
           --image-path ./character.jpg \
           --mode animation
    
    2. Replacement 模式（替换视频中的角色）:
       modal run wan22_animate_deploy.py \
           --video-path ./original_video.mp4 \
           --image-path ./new_character.jpg \
           --mode replacement
    """
    video_path = Path(video_path)
    image_path = Path(image_path)
    output_path = Path(output_path)
    
    # 验证输入文件
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return
    if not image_path.exists():
        print(f"错误：找不到图片文件 {image_path}")
        return
    
    # 验证模式
    if mode not in ["animation", "replacement"]:
        print(f"错误：模式必须是 'animation' 或 'replacement'")
        return
    
    print(f"🎭 模式: {mode.upper()}")
    print(f"🎬 输入视频: {video_path}")
    print(f"🖼️  角色图片: {image_path}")
    print(f"📐 分辨率: {resolution_width}x{resolution_height}")
    
    # 读取输入文件
    print("\n📤 上传输入文件...")
    video_bytes = video_path.read_bytes()
    image_bytes = image_path.read_bytes()
    
    # 步骤 1: 预处理
    print("\n🔄 步骤 1/2: 预处理视频和图片（这可能需要几分钟）...")
    model = Model()
    preprocessed_bytes = model.preprocess.remote(
        video_bytes=video_bytes,
        image_bytes=image_bytes,
        mode=mode,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
    )
    print(f"✅ 预处理完成")
    
    # 步骤 2: 生成视频
    print("\n🎨 步骤 2/2: 生成最终视频（这可能需要较长时间）...")
    video_result_bytes = model.generate.remote(
        preprocessed_bytes=preprocessed_bytes,
        mode=mode,
    )
    
    # 保存结果
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_bytes(video_result_bytes)
    print(f"\n✅ 完成！视频已保存到: {output_path}")
    print(f"💾 文件大小: {len(video_result_bytes) / 1024 / 1024:.2f} MB")
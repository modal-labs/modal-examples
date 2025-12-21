#!/usr/bin/env python3
# modal_app.py
#
# 运行方式: modal deploy modal_app.py
# (注意: 你需要先在 Modal 平台上设置一个名为 "huggingface-read" 的 Secret, 值为你的 Hugging Face 读取令牌)
# (Hugging Face Token 获取地址: https://huggingface.co/settings/tokens)

import modal
import os

# 1. 定义镜像 (Image)
# -----------------
# 我们将基于一个 Debian 镜像, 安装所有必要的系统和 Python 依赖项。

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .git_clone("https://github.com/MeiGen-AI/InfiniteTalk.git", checkout="main", path="/repo")
    .pip_install_from_requirements("/repo/requirements.txt")
    .pip_install(  # 根据 README, 覆盖/安装特定的 torch 和 xformers (用于 GPU)
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(  # 安装 README 中提到的其他依赖
        "misaki[en]", "ninja", "psutil", "packaging", "wheel",
        "flash_attn==2.7.4.post1",
        "librosa"
    )
    .workdir("/repo")
)

app = modal.App(name="infinitetalk-demo", image=image)  

# 2. 定义持久化存储 (Volume)
# -----------------------
WEIGHTS_DIR = "/models"
MODELS_VOLUME = modal.Volume.from_name("infinitetalk-models", create_if_missing=True)


# 3. 在镜像构建时下载模型
# ---------------------
@app.function(
    volumes={WEIGHTS_DIR: MODELS_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-read")],
    timeout=3600,  # 允许 60 分钟下载
    _allow_background_volume_commits=True  # 允许后台提交 Volume
)
def download_models():
    import subprocess
    
    models = {
        "Wan2.1-I2V-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P",
        "chinese-wav2vec2-base": "TencentGameMate/chinese-wav2vec2-base",
        "InfiniteTalk": "MeiGen-AI/InfiniteTalk"
    }

    for local_name, hub_name in models.items():
        local_path = os.path.join(WEIGHTS_DIR, local_name)
        
        # 改进的检查逻辑: 检查目录是否存在且不为空
        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"✓ Model {local_name} already exists at {local_path}")
            continue
            
        print(f"⬇ Downloading {hub_name} to {local_path}...")
        try:
            subprocess.run(
                [
                    "huggingface-cli", "download", hub_name,
                    "--local-dir", local_path,
                    "--local-dir-use-symlinks", "False"
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Successfully downloaded {local_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {local_name}: {e.stderr}")
            raise
            
    # 处理 README 中的特殊下载命令
    base_path = os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base")
    pr_file_path = os.path.join(base_path, "model.safetensors")
    if not os.path.exists(pr_file_path):
        print("⬇ Downloading special file for chinese-wav2vec2-base...")
        try:
            subprocess.run(
                [
                    "huggingface-cli", "download",
                    "TencentGameMate/chinese-wav2vec2-base",
                    "model.safetensors",
                    "--revision", "refs/pr/1",
                    "--local-dir", base_path,
                    "--local-dir-use-symlinks", "False"
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Successfully downloaded model.safetensors")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download model.safetensors: {e.stderr}")
            raise
    
    # 提交 Volume 变更
    MODELS_VOLUME.commit()
    print("✓ Model download complete and volume committed.")


# 4. 定义 Web 应用类
# -----------------
@app.cls(
    gpu="A10G",  # 或使用 "A100" 以获得更好的性能
    volumes={WEIGHTS_DIR: MODELS_VOLUME},
    timeout=1800,  # 30分钟超时
    container_idle_timeout=600,  # 10分钟后关闭空闲容器
    allow_concurrent_inputs=1,  # 一次只处理一个请求
    # 可选: 添加内存限制
    # memory=32768,  # 32GB RAM
)
class GradioApp:
    @modal.enter
    def build_app(self):
        import sys
        import os
        import gradio as gr
        
        print(f"📁 Current working directory: {os.getcwd()}")
        if os.getcwd() != "/repo":
            os.chdir("/repo")
            print(f"📁 Changed working directory to /repo")
        
        sys.path.insert(0, "/repo")  # 使用 insert 确保优先级

        try:
            # 验证模型文件存在
            model_paths = [
                os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
                os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
                os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
            ]
            for path in model_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model path not found: {path}")
            print("✓ All model paths verified")
            
            # 导入 InfiniteTalk 仓库中的 app.py 里的函数
            from app import parse_args, build_demo

            print("🔧 Patching sys.argv for argument parsing...")
            sys.argv = [
                "app.py",
                "--ckpt_dir", os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
                "--wav2vec_dir", os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
                "--infinitetalk_dir", os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
                "--num_persistent_param_in_dit", "0",
                "--motion_frame", "9",
            ]
            
            print(f"📝 Parsing arguments...")
            args = parse_args()
            
            print("🏗️  Building Gradio demo (this may take a few minutes)...")
            demo = build_demo(args)
            print("✓ Gradio demo built successfully")

            # Gradio 7.x 的正确方式
            self.demo = demo
            
        except Exception as e:
            print(f"✗ Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    @modal.web_endpoint(method="GET")
    def web(self):
        """处理 GET 请求"""
        return self.demo
    
    @modal.asgi_app()
    def serve(self):
        """提供完整的 ASGI 应用"""
        return self.demo.app


# 5. 本地入口点 (用于测试)
# ---------------------
@app.local_entrypoint()
def main():
    """本地测试入口"""
    print("Downloading models...")
    download_models.remote()
    print("Models downloaded. Deploy with: modal deploy modal_app.py")
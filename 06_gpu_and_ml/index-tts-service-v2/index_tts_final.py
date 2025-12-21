#!/usr/bin/env python3
# modal_app.py - InfiniteTalk 部署到 Modal
#
# 运行方式: modal deploy modal_app.py
# 前置条件: modal secret create huggingface-read HF_TOKEN=hf_xxx

import modal
import os

# ==================== 1. 镜像配置 ====================
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .git_clone("https://github.com/MeiGen-AI/InfiniteTalk.git", checkout="main", path="/repo")
    .pip_install_from_requirements("/repo/requirements.txt")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "misaki[en]", "ninja", "psutil", "packaging", "wheel",
        "flash_attn==2.7.4.post1", "librosa"
    )
    .workdir("/repo")
)

app = modal.App(name="infinitetalk-demo", image=image)

# ==================== 2. 持久化存储 ====================
WEIGHTS_DIR = "/models"
MODELS_VOLUME = modal.Volume.from_name("infinitetalk-models", create_if_missing=True)


# ==================== 3. Web 应用类 ====================
@app.cls(
    gpu="A100G",  # 推荐 A100 以获得更好性能
    volumes={WEIGHTS_DIR: MODELS_VOLUME},
    timeout=1800,
    container_idle_timeout=600,
    allow_concurrent_inputs=1,
    secrets=[modal.Secret.from_name("huggingface-read")],  # 移到这里
)
class GradioApp:
    
    @modal.enter
    def setup(self):
        """容器启动时执行: 下载模型 + 构建应用"""
        import sys
        import os
        from huggingface_hub import snapshot_download
        
        print("\n" + "=" * 70)
        print("🚀 初始化 InfiniteTalk 服务...")
        print("=" * 70)
        
        # ===== 步骤 1: 智能检查并下载模型 =====
        self._download_models_if_needed()
        
        # ===== 步骤 2: 设置 Python 环境 =====
        print("\n📁 配置工作目录...")
        if os.getcwd() != "/repo":
            os.chdir("/repo")
        sys.path.insert(0, "/repo")
        print(f"✅ 工作目录: {os.getcwd()}")
        
        # ===== 步骤 3: 构建 Gradio 应用 =====
        self._build_gradio_app()
        
        print("\n" + "=" * 70)
        print("✅ InfiniteTalk 服务就绪!")
        print("=" * 70 + "\n")
    
    def _download_models_if_needed(self):
        """检查并下载所需模型 (使用 Python API)"""
        from huggingface_hub import snapshot_download
        
        models = {
            "Wan2.1-I2V-14B-480P": {
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P",
                "check_files": ["model_index.json"]  # 关键文件检查
            },
            "chinese-wav2vec2-base": {
                "repo_id": "TencentGameMate/chinese-wav2vec2-base",
                "check_files": ["config.json", "model.safetensors"]
            },
            "InfiniteTalk": {
                "repo_id": "MeiGen-AI/InfiniteTalk",
                "check_files": ["single/infinitetalk.safetensors"]
            }
        }
        
        for local_name, config in models.items():
            local_path = os.path.join(WEIGHTS_DIR, local_name)
            repo_id = config["repo_id"]
            check_files = config["check_files"]
            
            # 智能检查: 验证关键文件存在
            all_exist = all(
                os.path.exists(os.path.join(local_path, f)) 
                for f in check_files
            )
            
            if all_exist:
                print(f"✅ {local_name} 已存在")
                continue
            
            print(f"⏳ 下载 {repo_id} → {local_path}")
            print(f"   (这可能需要几分钟...)")
            
            try:
                # 特殊处理: chinese-wav2vec2-base 需要特定 revision
                if local_name == "chinese-wav2vec2-base":
                    print("   ⚠️  使用特殊 revision: refs/pr/1")
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_path,
                        local_dir_use_symlinks=False,
                        revision="refs/pr/1",
                        allow_patterns=["model.safetensors", "*.json"]
                    )
                else:
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_path,
                        local_dir_use_symlinks=False,
                    )
                
                print(f"✅ {local_name} 下载完成")
                
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                raise
        
        # 提交 Volume (Modal 会在函数结束时自动提交,但显式调用更安全)
        print("\n💾 提交模型到持久化卷...")
        MODELS_VOLUME.commit()
        print("✅ Volume 已提交")
    
    def _build_gradio_app(self):
        """构建 Gradio 应用"""
        import sys
        
        print("\n🏗️  构建 Gradio 界面...")
        
        try:
            # 验证模型路径
            model_paths = {
                "ckpt_dir": os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
                "wav2vec_dir": os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
                "infinitetalk_dir": os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
            }
            
            for name, path in model_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"❌ {name} 路径不存在: {path}")
            print("✅ 所有模型路径验证通过")
            
            # 导入并解析参数
            from app import parse_args, build_demo
            
            sys.argv = [
                "app.py",
                "--ckpt_dir", model_paths["ckpt_dir"],
                "--wav2vec_dir", model_paths["wav2vec_dir"],
                "--infinitetalk_dir", model_paths["infinitetalk_dir"],
                "--num_persistent_param_in_dit", "0",  # 低内存模式
                "--motion_frame", "9",
            ]
            
            args = parse_args()
            self.demo = build_demo(args)
            print("✅ Gradio 应用构建完成")
            
        except Exception as e:
            print(f"❌ 构建失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @modal.asgi_app()
    def serve(self):
        """提供 Gradio ASGI 应用"""
        return self.demo.app


# ==================== 4. 本地测试入口 ====================
@app.local_entrypoint()
def main():
    """本地测试: modal run modal_app.py"""
    print("🧪 启动测试服务器...")
    print("📝 提示: 使用 'modal deploy modal_app.py' 进行生产部署")
    
    # 本地测试时不需要预下载,进入 setup 会自动处理
    C7192C64E5Z378F2AB1
# --- 修正和优化的代码 ---
import modal
import os
import io
import sys
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import tempfile
import traceback

# --- 步骤 1: 定义容器镜像 ---
# 镜像定义保持不变，非常完善
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2", "numpy", "fastapi", "uvicorn",
        "soundfile", "pydub", "librosa", "phonemizer==3.2.1", "einops",
        "unidecode", "transformers", "accelerate", "scipy", "matplotlib",
        "tensorboard", "g2p_en", "jieba", "pypinyin", "cn2an", "inflect",
        "eng_to_ipa", "ko_pron", "indic_transliteration", "num2words",
        "gradio", "huggingface_hub",
    )
    # 安装 pynini 和 WeTextProcessing（避免运行时编译问题）
    .run_commands(
        "pip install pynini==2.1.6",
        "pip install WeTextProcessing --no-deps"
    )
)

# --- 步骤 2: 创建 Modal App 和持久化存储 ---
app = modal.App("index-tts2-service")
# 使用持久化存储来缓存模型，避免每次冷启动都重新下载
model_volume = modal.Volume.from_name("indextts2-models-volume", create_if_missing=True)
MODEL_DIR = "/models"

# --- 步骤 3: 下载器函数 ---
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=1800,
)
def download_models():
    """
    从 Hugging Face Hub 下载 IndexTTS-2 模型文件到持久化存储。
    """
    # 必须的文件列表，包括配置文件
    required_files = [
        "hubert_base.pt",
        "sovits-vctk-16k.pth",
        "vits-vctk-16k.pth",
        "config.yaml" # 添加了必要的配置文件
    ]
    
    # 检查是否所有必需文件都已存在
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files):
        print("✅ 模型文件已存在，跳过下载。")
        return

    print("⏳ 开始下载 IndexTTS-2 模型文件...")
    
    try:
        from huggingface_hub import snapshot_download
        
        print("    使用 huggingface_hub 下载模型...")
        snapshot_download(
            repo_id="IndexTeam/IndexTTS-2",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=["*.pth", "*.pt", "*.yaml"] # 只下载需要的文件
        )
        print("✅ 模型下载成功 (via huggingface_hub)")
        
    except Exception as e:
        print(f"huggingface_hub 下载失败: {e}")
        print("    fallback 到 wget 下载...")
        
        # Fallback: 使用 wget 下载
        base_url = "https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/"
        # 确保 wget 下载列表和检查列表一致
        model_files_to_download = required_files + ["ny-vctk-16k.pth"] # 额外模型

        for filename in model_files_to_download:
            url = base_url + filename
            destination_path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(destination_path):
                print(f"    - {filename} 已存在，跳过。")
                continue
            
            print(f"    - 正在下载 {filename}...")
            download_cmd = f"wget -O {destination_path} {url}"
            result = os.system(download_cmd)
            
            if result == 0 and os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
                print(f"      ✅ {filename} 下载成功")
            else:
                print(f"      ❌ {filename} 下载失败")
        
    # 将下载的文件提交到持久卷
    model_volume.commit()
    print("💾 模型已保存到持久卷！")


# --- 步骤 4: TTS 服务类 ---
@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    gpu="A10G",  # A10G性价比更高，可按需换成 T4 或 A100
    scaledown_window=300,
    timeout=600,
    enable_memory_snapshot=True,
)
class IndexTTS2Service:
    @modal.enter()
    def load_model(self):
        """
        容器启动时执行：克隆代码库、安装、加载模型到内存。
        启用 memory_snapshot 后，这部分只会在第一次启动时完整运行。
        """
        print("⏳ 正在初始化 IndexTTS-2 服务...")
        
        repo_path = "/tmp/index-tts"
        
        try:
            # 1. 克隆并安装 index-tts 仓库
            if not os.path.exists(repo_path):
                print(f"    - 正在克隆 index-tts 仓库到 {repo_path}...")
                os.system(f"git clone https://github.com/index-tts/index-tts.git {repo_path}")
                
                print("    - 正在安装 index-tts 依赖...")
                # 使用 -e 安装会自动处理 requirements.txt
                os.system(f"cd {repo_path} && pip install -e .")
            
            # 将仓库路径添加到 Python 解释器路径
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            
            # 2. 导入模型类
            # 简化: 直接使用确认过的导入路径，这更清晰
            print("    - 正在从 indextts.infer 导入 IndexTTS 类...")
            from indextts.infer import IndexTTS
            print("      ✅ 成功导入 IndexTTS 模块")
            
            # 3. 初始化模型
            config_path = os.path.join(MODEL_DIR, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"关键配置文件 config.yaml 未在模型目录 {MODEL_DIR} 中找到！")

            print(f"    - 正在初始化 IndexTTS 模型，使用配置文件: {config_path}")
            self.tts_model = IndexTTS(
                cfg_path=config_path,
                model_dir=MODEL_DIR
            )
            print("✅ IndexTTS-2 模型加载成功!")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("\n" + "="*20 + " 调试信息 " + "="*20)
            print("Python 路径:")
            for path in sys.path: print(f"  {path}")
            print(f"\n模型目录内容 ({MODEL_DIR}):")
            if os.path.exists(MODEL_DIR):
                for item in os.listdir(MODEL_DIR): print(f"  - {item} ({os.path.getsize(os.path.join(MODEL_DIR, item))} bytes)")
            else: print("  目录不存在")
            print(f"\n仓库目录内容 ({repo_path}):")
            if os.path.exists(repo_path):
                for item in os.listdir(repo_path): print(f"  - {item}")
            else: print("  目录不存在")
            traceback.print_exc()
            raise e

    @modal.method()
    def generate_speech_internal(self, text: str, voice_file_bytes: bytes = None) -> bytes:
        """
        核心语音生成逻辑。
        """
        try:
            print(f"⏳ 正在生成语音，文本: '{text[:50]}...'")
            
            # 使用临时文件来处理参考语音和输出
            voice_path = None
            temp_voice_file = None
            if voice_file_bytes:
                temp_voice_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_voice_file.write(voice_file_bytes)
                temp_voice_file.close()
                voice_path = temp_voice_file.name
                print(f"    - 参考语音已保存到临时文件: {voice_path}")
            
            temp_output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_output_file.close()
            output_path = temp_output_file.name
            
            # 调用 TTS 推理
            # 简化: 直接调用 infer 方法，因为我们已经确认了 API
            print("    - 调用模型 infer() 方法...")
            self.tts_model.infer(
                text=text,
                voice=voice_path,
                output_path=output_path
            )
            
            # 读取生成的音频文件
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            print(f"    - 音频生成成功，大小: {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            print(f"❌ 推理过程中出错: {e}")
            traceback.print_exc()
            raise e
        finally:
            # 清理临时文件
            if 'temp_voice_file' in locals() and temp_voice_file and os.path.exists(temp_voice_file.name):
                os.unlink(temp_voice_file.name)
            if 'temp_output_file' in locals() and temp_output_file and os.path.exists(temp_output_file.name):
                os.unlink(temp_output_file.name)

    @modal.asgi_app()
    def fastapi_app(self):
        """
        提供 FastAPI 接口服务。
        """
        web_app = FastAPI(title="IndexTTS-2 Service")

        @web_app.get("/", summary="Health Check")
        async def root():
            return {"message": "IndexTTS-2 Service is running!"}

        @web_app.post("/generate-speech", summary="Generate Speech from Text and optional Voice")
        async def generate_speech_endpoint(
            text: str = Form(..., description="要转换的文字"),
            voice_file: UploadFile = File(None, description="WAV格式的参考语音文件（可选）")
        ):
            try:
                print(f"收到 /generate-speech 请求，文本: '{text[:100]}...'")
                
                voice_bytes = await voice_file.read() if voice_file else None
                if voice_bytes:
                    print(f"    - 收到参考语音文件: {voice_file.filename}, 大小: {len(voice_bytes)} bytes")
                
                audio_bytes = self.generate_speech_internal(text, voice_bytes)
                
                return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
                
            except Exception as e:
                print(f"❌ 处理请求时出错: {e}")
                return {"error": str(e), "type": "generation_error"}, 500

        return web_app
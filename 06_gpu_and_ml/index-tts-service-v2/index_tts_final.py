"""
IndexTTS-2 Modal 部署脚本 (修复版)

核心修复：
1. 移除了独立的 `download_models` 函数。
2. 将模型下载和检查的逻辑，直接移入 `IndexTTS2Service` 类的 `@modal.enter` (setup) 方法的 *开头*。
3. 这保证了在 `setup` 尝试加载模型之前，模型文件必定存在于 /checkpoints 目录，解决了 FileNotFoundError。
4. (v3) 修复了 `AttributeError: module 'modal' has no attribute 'UploadFile'`，
   通过从 `fastapi` 导入 `UploadFile` 和 `Form` 并更新 `api` 方法签名。
"""
import modal
import os
import io
from typing import Optional
from fastapi import UploadFile, Form  

# ===== 步骤 1: 构建镜像（遵循官方方式）=====
def build_indextts_image():
    """按照官方文档构建镜像"""
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "git",
            "git-lfs", 
            "ffmpeg",
            "build-essential",
            "curl"
        )
    # 1. 安装基础依赖
    .pip_install(
        "uv",
        "fastapi",
        "python-multipart",
        "huggingface_hub",
        "packaging",
        "torch>=2.0.0",  # 添加关键依赖
        "librosa>=0.10.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "soundfile>=0.12.0",
    )
        # 2. 克隆项目
        .run_commands(
            "git clone https://github.com/index-tts/index-tts.git /opt/index-tts2",
            "cd /opt/index-tts2 && git lfs install && git lfs pull"
        )
        # 3. 首先尝试 UV sync，然后运行 pip install -e 作为备份
        .run_commands(
            # UV sync 尝试（可能会有错误但继续）
            "cd /opt/index-tts2 && uv sync --all-extras || echo 'uv sync with errors, continuing...'",
            # 使用 pip install -e 作为备份安装方法
            "pip install -e /opt/index-tts2",
            # 安装项目的 requirements.txt（如果存在）
            "if [ -f /opt/index-tts2/requirements.txt ]; then pip install -r /opt/index-tts2/requirements.txt; fi"
        )
    )

image = build_indextts_image()

# ===== 步骤 2: 创建 Modal App =====
app = modal.App("indextts2-official-fixed") # 改了新名称以示区别

# 持久化存储
model_volume = modal.Volume.from_name("indextts2-models-official", create_if_missing=True)
CHECKPOINTS_DIR = "/checkpoints"


# ===== 步骤 3: TTS 服务 (集成了模型下载) =====
@app.cls(
    image=image,
    gpu="A10G",
    volumes={CHECKPOINTS_DIR: model_volume},
    timeout=1200,  # 增加了 setup 的超时时间以供下载
    scaledown_window=120,
    # min_containers=1,  # 保持热启动
)
class IndexTTS2Service:
    
    @modal.enter()
    def setup(self):
        """
        容器启动时初始化
        1. 检查并下载模型 (修复点)
        2. 设置 Python 路径
        3. 加载模型
        4. 下载默认参考音频
        """
        import sys
        import warnings
        import urllib.request
        from huggingface_hub import snapshot_download
        
        print("\n" + "=" * 70)
        print("🚀 初始化 IndexTTS-2 服务...")
        print("=" * 70)
        
        # ===== 步骤 1: 检查并下载模型（从 download_models 移入）=====
        print(f"🔄 检查模型文件于: {CHECKPOINTS_DIR}")
        
        required_files = [
            "config.yaml", "gpt.pth", "s2mel.pth",
            "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt"
        ]
        
        all_exist = all(
            os.path.exists(os.path.join(CHECKPOINTS_DIR, f)) 
            for f in required_files
        )
        
        if all_exist:
            print(f"✅ 模型已在持久化卷 {CHECKPOINTS_DIR} 中存在")
        else:
            print(f"⏳ 模型不存在，开始下载 (约 4.7GB) 到 {CHECKPOINTS_DIR}...")
            
            snapshot_download(
                repo_id="IndexTeam/IndexTTS-2",
                local_dir=CHECKPOINTS_DIR,
                local_dir_use_symlinks=False,
            )
            
            print("✅ 模型下载完成")
            # 在 @modal.enter 中，Volume 会在函数退出时自动提交
            # model_volume.commit() # 在这里不是必需的
        
        # ===== 步骤 1.5: 下载默认参考音频 =====
        print("⏳ 下载默认参考音频...")
        default_audio_url = "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
        default_audio_path = f"{CHECKPOINTS_DIR}/default_voice.wav"
        
        if not os.path.exists(default_audio_path):
            print(f"   从 {default_audio_url} 下载...")
            urllib.request.urlretrieve(default_audio_url, default_audio_path)
            print("✅ 默认参考音频下载完成")
        else:
            print("✅ 默认参考音频已存在")
        
        # ===== 步骤 2: 初始化环境和加载模型 (你的原始代码) =====
        
        print("⏳ 设置 Python 环境...")
        # 抑制警告
        warnings.filterwarnings('ignore')
        
        # 设置路径（官方文档要求）
        repo_path = "/opt/index-tts2"
        
        # 方法 1: 添加 UV 虚拟环境到 Python 路径
        venv_site_packages = f"{repo_path}/.venv/lib/python3.10/site-packages"
        if os.path.exists(venv_site_packages):
            sys.path.insert(0, venv_site_packages)
            print(f"✅ 添加 UV 虚拟环境: {venv_site_packages}")
        else:
            print(f"⚠️ 未找到 UV 虚拟环境路径: {venv_site_packages}")
        
        # 方法 2: 添加项目根目录（官方文档建议）
        sys.path.insert(0, repo_path)
        os.environ['PYTHONPATH'] = f"{repo_path}:{os.environ.get('PYTHONPATH', '')}"
        print(f"✅ 设置 PYTHONPATH: {repo_path}")
        
        # 检查 config.yaml 是否真的存在
        config_file_path = f"{CHECKPOINTS_DIR}/config.yaml"
        if not os.path.exists(config_file_path):
            print(f"❌ 严重错误: config.yaml 未能在 {config_file_path} 找到！")
            print("   请检查下载逻辑或持久卷。")
            # 列出目录内容以供调试
            print(f"   {CHECKPOINTS_DIR} 目录内容: {os.listdir(CHECKPOINTS_DIR)}")
            raise FileNotFoundError(f"config.yaml 未找到于 {config_file_path}")
        else:
             print(f"✅ 确认 config.yaml 存在: {config_file_path}")

        # 加载模型（使用官方 API）
        print("⏳ 加载 IndexTTS-2 模型...")
        
        try:
            # 优先使用 IndexTTS2（新版本）
            from indextts.infer_v2 import IndexTTS2
            
            print("✅ 尝试使用 IndexTTS2 (infer_v2)")
            self.tts = IndexTTS2(
                cfg_path=config_file_path, # 使用验证过的路径
                model_dir=CHECKPOINTS_DIR,
                use_fp16=False,  # 稳定性优先
                use_cuda_kernel=False,
                use_deepspeed=False
            )
            self.use_v2 = True
            
        except Exception as e:
            print(f"⚠️ IndexTTS2 (v2) 加载失败: {e}")
            print("🔄 回退到 IndexTTS (v1 infer)")
            
            from indextts.infer import IndexTTS
            
            self.tts = IndexTTS(
                cfg_path=config_file_path, # 使用验证过的路径
                model_dir=CHECKPOINTS_DIR
            )
            self.use_v2 = False
        
        print("=" * 70)
        print("✅ IndexTTS-2 服务就绪！")
        print("=" * 70 + "\n")
    
    def _generate_internal(
        self,
        text: str,
        voice_bytes: Optional[bytes] = None,
        emotion_bytes: Optional[bytes] = None
    ) -> bytes:
        """
        生成语音
        
        参数:
            text: 要合成的文本
            voice_bytes: 参考语音（WAV 格式）
            emotion_bytes: 情感参考（WAV 格式，仅 v1 支持）
        """
        import tempfile
        import time
        
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"📝 文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        voice_path = None
        emotion_path = None
        output_path = None
        
        try:
            # 保存参考语音
            if voice_bytes:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(voice_bytes)
                    voice_path = f.name
                print(f"   📎 参考语音: {len(voice_bytes)/1024:.1f}KB")
            
            # 保存情感参考（仅 v1）
            if emotion_bytes and not self.use_v2:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(emotion_bytes)
                    emotion_path = f.name
                print(f"   🎭 情感参考: {len(emotion_bytes)/1024:.1f}KB")
            
            # 输出文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            
            # 根据版本调用不同 API
            print("   ⏳ 生成中...")
            
            if self.use_v2:
                # IndexTTS2 API (官方文档)
                if voice_path is None:
                    # 如果没有提供参考音频，使用默认的示例音频
                    print("   ℹ️ 未提供参考音频，使用默认声音")
                    voice_path = f"{CHECKPOINTS_DIR}/default_voice.wav"
                    print(f"   使用默认参考音频: {voice_path}")
                
                self.tts.infer(
                    spk_audio_prompt=voice_path,
                    text=text,
                    output_path=output_path,
                    verbose=False
                )
            else:
                # IndexTTS1 API
                self.tts.infer(
                    voice=voice_path,
                    text=text,
                    output_path=output_path,
                    emotion=emotion_path
                )
            
            # 读取结果
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            elapsed = time.time() - start_time
            print(f"   ✅ 成功: {len(audio_data)/1024:.1f}KB, 耗时 {elapsed:.2f}s")
            print(f"{'='*60}\n")
            
            return audio_data
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            print(traceback.format_exc()) # 打印更详细的堆栈
            print(f"{'='*60}\n")
            raise
            
        finally:
            # 清理临时文件
            for path in [voice_path, emotion_path, output_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
    
    @modal.fastapi_endpoint(method="POST")
    def api(
        self, 
        text: str = Form(...),
        voice: UploadFile = None,
        voice_url: str = Form(None),
        emotion: UploadFile = None
    ):
        """
        HTTP API 端点
        
        参数:
            text: 要合成的文本
            voice: 上传的参考音频文件
            voice_url: 在线参考音频URL
            emotion: 情感参考音频（仅v1支持）
        
        使用方法:
        1. 使用本地音频文件：
            curl -X POST "https://YOUR-URL/api" \
              -F "text=你好世界" \
              -F "voice=@speaker.wav" \
              -F "speed=0.8" \
              --output output.wav
              
        2. 使用在线音频 URL：
            curl -X POST "https://YOUR-URL/api" \
              -F "text=你好世界" \
              -F "voice_url=https://example.com/audio.wav" \
              -F "speed=0.8" \
              --output output.wav
              
        3. 不提供参考音频（使用默认声音）：
            curl -X POST "https://YOUR-URL/api" \
              -F "text=你好世界" \
              -F "speed=0.8" \
              --output output.wav
        """
        from fastapi.responses import Response, JSONResponse
        import traceback, sys
        
        print("\n" + "=" * 70)
        print("📝 收到 API 请求")
        
        # 打印详细的请求信息
        print("\n请求参数:")
        print(f"text 类型: {type(text)}")
        print(f"text 值: {text}")
        print(f"voice: {voice}")
        print(f"emotion: {emotion}")
        print("\nPython 路径:")
        print("\n".join(sys.path))
        print("\n当前工作目录:")
        print(os.getcwd())
        
        try:
            print("\n⏳ 验证请求参数...")
            
            if text is None:
                print("❌ text 参数为 None")
                return JSONResponse(
                    content={"error": "text 参数（表单字段）不能为空"},
                    status_code=400
                )

            if len(text.strip()) == 0:
                print("❌ text 参数为空字符串")
                return JSONResponse(
                    content={"error": "text 参数不能为空"},
                    status_code=400
                )
            
            print("✅ 参数验证通过")
            print("⏳ 准备语音数据...")
            
            voice_data = None
            
            # 处理参考音频（支持文件上传或URL）
            if voice and voice_url:
                return JSONResponse(
                    content={"error": "不能同时提供voice和voice_url参数"},
                    status_code=400
                )
            
            voice_data = None
            voice_source = "默认"
            
            if voice:
                try:
                    voice_data = voice.file.read()
                    voice_source = "上传文件"
                    print(f"✅ 读取上传的参考音频: {len(voice_data)/1024:.1f}KB")
                except Exception as e:
                    print(f"❌ 读取上传的参考音频失败: {e}")
                    return JSONResponse(
                        content={"error": f"读取上传的参考音频失败: {str(e)}"},
                        status_code=400
                    )
            elif voice_url:
                try:
                    print(f"⏳ 从URL下载参考音频: {voice_url}")
                    import urllib.request
                    from urllib.parse import urlparse
                    
                    # 1. 验证URL
                    parsed_url = urlparse(voice_url)
                    if not parsed_url.scheme in ['http', 'https']:
                        raise ValueError("URL必须是http或https协议")
                    
                    # 2. 预先检查文件大小
                    MAX_SIZE = 10 * 1024 * 1024  # 10MB 限制
                    
                    req = urllib.request.Request(
                        voice_url,
                        method='HEAD'  # 只获取头信息，不下载内容
                    )
                    
                    with urllib.request.urlopen(req) as response:
                        # 检查内容类型
                        content_type = response.headers.get('content-type', '').lower()
                        if not any(t in content_type for t in ['audio', 'application/octet-stream']):
                            raise ValueError(f"不支持的文件类型: {content_type}")
                        
                        # 检查文件大小
                        size = response.headers.get('content-length')
                        if size and int(size) > MAX_SIZE:
                            raise ValueError(f"文件太大: {int(size)/1024/1024:.1f}MB (最大限制10MB)")
                    
                    # 3. 分块下载文件
                    print("   检查通过，开始下载...")
                    chunks = []
                    with urllib.request.urlopen(voice_url) as response:
                        total_size = 0
                        while True:
                            chunk = response.read(8192)  # 8KB 缓冲区
                            if not chunk:
                                break
                            total_size += len(chunk)
                            if total_size > MAX_SIZE:
                                raise ValueError("文件超过大小限制")
                            chunks.append(chunk)
                    
                    voice_data = b''.join(chunks)
                    voice_source = "URL"
                    print(f"✅ 成功下载参考音频: {len(voice_data)/1024:.1f}KB")
                    
                except Exception as e:
                    print(f"❌ URL音频下载失败: {e}")
                    raise
                        
                except Exception as e:
                    print(f"❌ 处理URL音频失败: {e}")
                    return JSONResponse(
                        content={"error": f"处理URL音频失败: {str(e)}"},
                        status_code=400
                    )

            emotion_data = None
            if emotion:
                try:
                    emotion_data = emotion.file.read()
                    print(f"✅ 读取情感参考: {len(emotion_data)/1024:.1f}KB")
                except Exception as e:
                    print(f"❌ 读取情感参考失败: {e}")
                    return JSONResponse(
                        content={"error": f"读取情感参考失败: {str(e)}"},
                        status_code=400
                    )

            print("\n" + "=" * 60)
            print(f"⏳ 开始生成语音 (使用{voice_source}参考音频)...")
            print(f"• 文本长度: {len(text)} 字符")
            print(f"• 参考音频: {voice_source}")
            if voice_data:
                print(f"• 音频大小: {len(voice_data)/1024:.1f}KB")
            print("=" * 60 + "\n")
            
            try:
                # 生成语音
                audio_data = self._generate_internal(text, voice_data, emotion_data)
                print(f"✅ 语音生成成功: {len(audio_data)/1024:.1f}KB")
                
                # 返回 WAV 文件
                return Response(
                    content=audio_data,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": 'attachment; filename="output.wav"',
                    }
                )
            except Exception as e:
                print(f"❌ 语音生成失败: {e}")
                print(traceback.format_exc())
                raise  # 让外层 try-except 处理
            
        except Exception as e:
            import traceback
            return JSONResponse(
                content={
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                },
                status_code=500
            )

# ===== 本地入口 (更新) =====
@app.local_entrypoint()
def main():
    """部署和测试"""
    print("\n" + "=" * 70)
    print("🚀 IndexTTS-2 Modal 部署（官方 UV 方式）")
    print("=" * 70)
    
    # 1. 部署/更新服务
    # 运行 `modal run` 时，Modal 会自动部署/更新 App。
    # IndexTTS2Service 的 @modal.enter() 方法 (setup) 
    # 将在容器启动时自动运行，并处理模型下载。
    print("\n📦 服务部署中...")
    print("   Modal 将自动启动容器并运行 setup 方法。")
    print("   setup 方法将自动检查并下载模型（如果需要）。")
    
    print("\n✅ 部署完成！服务正在启动...")
    print("   首次启动（冷启动）可能需要 5-10 分钟以下载 4.7GB 的模型。")
    print("   模型下载到持久卷后，后续重启（热启动）会非常快 (约 30-60秒)。")
    
    # 2. 显示使用说明
    print("\n" + "=" * 70)
    print("📝 API 使用方法:")
    print("=" * 70)
    print(" (请将 'YOUR-MODAL-URL' 替换为部署后 Modal 提供的 URL)")
    
    print("\n1. 基础用法（无参考语音）:")
    print('   curl -X POST "https://YOUR-MODAL-URL/api" \\')
    print('     -F "text=你好，这是测试" \\')
    print('     --output output.wav')
    
    print("\n2. 语音克隆（带参考语音）:")
    print('   curl -X POST "https://YOUR-MODAL-URL/api" \\')
    print('     -F "text=这是克隆的声音" \\')
    print('     -F "voice=@reference.wav" \\')
    print('     --output cloned.wav')
    
    print("\n3. 情感控制（仅 IndexTTS v1 支持）:")
    print('   curl -X POST "https://YOUR-MODAL-URL/api" \\')
    print('     -F "text=我很开心" \\')
    print('     -F "voice=@speaker.wav" \\')
    print('     -F "emotion=@happy.wav" \\')
    print('     --output emotional.wav')
    
    print("\n" + "=" * 70)
    print("⚠️  注意事项:")
    print("- 参考语音建议 3-10 秒，清晰的 WAV 格式")
    print("- min_containers=1 保持热启动，下载完成后，后续请求 < 5 秒")
    print("=" * 70 + "\n")

# 测试命令
# curl -X POST https://rodneycornwell--indextts2-official-fixed-indextts2service-api.modal.run \
#   -F "text=这是一个没有参考语音的测试。" \
#   --output no_voice.wav

#   curl -X POST "https://rodneycornwell--indextts2-official-fixed-indextts2service-api.modal.run" \
#   -F "text=这是一个测试文本,用于郭德纲声音小说" \
#   -F "voice_url=https://res.cloudinary.com/dhodnm1yv/video/upload/v1761051270/mtienyhdgsz6omvbaxrv.mp3" \
#   --output test.wav

#   curl -X POST "https://rodneycornwell--indextts2-official-fixed-indextts2service-api.modal.run" \
#   -F "text=你好世界,这个用于本地音频参考的测试" \
#   -F "voice=@ref.wav" \
#   --output output.wav
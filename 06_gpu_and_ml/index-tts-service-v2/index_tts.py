"""
IndexTTS-2 Modal 部署脚本（备用版本 - 使用更保守的依赖版本）
如果主版本仍有问题，可以尝试这个版本
"""
import modal
import os

# ===== 步骤 1: 构建基础镜像（使用更保守的版本）=====
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "ffmpeg", "build-essential", "wget")
    # 使用更保守的依赖版本
    .pip_install(
        "numpy<2.0",
        "torch==2.0.1",           # 使用更稳定的 torch 版本
        "torchaudio",
        "omegaconf",
        "hydra-core",
        "transformers==4.20.1",   # 更保守的 transformers 版本
        "accelerate==0.20.3",     # 匹配的 accelerate 版本
        "einops",
        "librosa",
        "soundfile",
        "phonemizer",
        "jieba",
        "pypinyin",
        "cn2an",
        "g2p_en",
        "inflect",
        "unidecode",
        "pydub",
        "matplotlib",
        "scipy",
        "huggingface_hub",
        "fastapi",
        "python-multipart",
    )
)

# ===== 步骤 2: 创建 Modal App =====
app = modal.App("indextts2-service-backup")

# 持久化存储
model_volume = modal.Volume.from_name("indextts2-models", create_if_missing=True)
CHECKPOINTS_DIR = "/checkpoints"

# ===== 步骤 3: 模型下载器 =====
@app.function(
    image=image,
    volumes={CHECKPOINTS_DIR: model_volume},
    timeout=3600,
)
def download_models():
    """从 Hugging Face 下载模型"""
    print("🔄 检查模型文件...")
    
    required_files = [
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt"
    ]
    
    all_exist = all(
        os.path.exists(os.path.join(CHECKPOINTS_DIR, f)) 
        for f in required_files
    )
    
    if all_exist:
        print("✅ 模型已存在")
        return
    
    print("⏳ 下载模型...")
    
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-2",
        local_dir=CHECKPOINTS_DIR,
        local_dir_use_symlinks=False,
    )
    
    print("✅ 下载完成")
    model_volume.commit()

# ===== 步骤 4: TTS 服务 =====
@app.cls(
    image=image,
    gpu="A10G",
    volumes={CHECKPOINTS_DIR: model_volume},
    timeout=600,
    scaledown_window=300,
)
class IndexTTS2Service:
    
    @modal.enter()
    def load_model(self):
        """容器启动时：克隆最新代码并加载模型"""
        import sys
        import subprocess
        
        repo_path = "/tmp/index-tts"
        
        print("⏳ 克隆 IndexTTS-2 最新代码...")
        
        # 每次启动都克隆最新代码（确保代码和模型匹配）
        if os.path.exists(repo_path):
            subprocess.run(["rm", "-rf", repo_path], check=True)
        
        subprocess.run([
            "git", "clone", 
            "--depth", "1",  # 只克隆最新提交
            "https://github.com/index-tts/index-tts.git",
            repo_path
        ], check=True)
        
        # 安装代码包（不安装依赖，因为基础镜像已有）
        subprocess.run([
            "pip", "install", "-e", repo_path, "--no-deps"
        ], check=True)
        
        sys.path.insert(0, repo_path)
        
        print("⏳ 初始化 IndexTTS-2 模型...")
        
        # 检查 transformers 版本兼容性
        import transformers
        print(f"📦 当前 transformers 版本: {transformers.__version__}")
        
        # 修复 IndexTTS 的导入问题
        self._fix_indextts_imports()
        
        # 直接修补 IndexTTS 源代码
        self._patch_indextts_source()
        
        try:
            from indextts.infer import IndexTTS
        except ImportError as e:
            print(f"❌ 导入 IndexTTS 失败: {e}")
            print("💡 这通常是由于 transformers 版本不兼容导致的")
            print("🔧 建议使用 transformers==4.20.1 或更早版本")
            raise
        
        self.tts = IndexTTS(
            cfg_path=f"{CHECKPOINTS_DIR}/config.yaml",
            model_dir=CHECKPOINTS_DIR
        )
        
        print("✅ 模型加载成功!")
    
    def _fix_indextts_imports(self):
        """修复 IndexTTS 的导入问题"""
        import sys
        import importlib
        
        print("🔧 修复 IndexTTS 导入问题...")
        
        # 尝试不同的 cache_utils 导入路径
        cache_utils_paths = [
            "transformers.cache_utils",
            "transformers.utils.cache_utils", 
            "transformers.generation.cache_utils",
            "transformers.generation.utils.cache_utils"
        ]
        
        cache_utils_module = None
        for path in cache_utils_paths:
            try:
                cache_utils_module = importlib.import_module(path)
                print(f"✅ 找到 cache_utils 模块: {path}")
                break
            except ImportError:
                continue
        
        if cache_utils_module is None:
            print("❌ 未找到 cache_utils 模块，尝试创建兼容性模块...")
            # 创建一个兼容性模块
            self._create_compatibility_module()
        else:
            # 将找到的模块注册到 transformers 命名空间
            import transformers
            transformers.cache_utils = cache_utils_module
            print("✅ 已注册 cache_utils 模块到 transformers 命名空间")
    
    def _create_compatibility_module(self):
        """创建兼容性模块以解决导入问题"""
        import sys
        import types
        
        print("🔧 创建兼容性模块...")
        
        # 创建一个空的 cache_utils 模块
        cache_utils_module = types.ModuleType('cache_utils')
        
        # 尝试从其他位置导入需要的类
        try:
            # 尝试从 transformers.generation 导入
            from transformers.generation import Cache
            cache_utils_module.Cache = Cache
            print("✅ 找到 Cache 类")
        except ImportError:
            try:
                # 尝试从 transformers.utils 导入
                from transformers.utils import Cache
                cache_utils_module.Cache = Cache
                print("✅ 找到 Cache 类")
            except ImportError:
                print("⚠️ 未找到 Cache 类，创建占位符")
                cache_utils_module.Cache = object
        
        # 创建 EncoderDecoderCache 占位符
        cache_utils_module.EncoderDecoderCache = cache_utils_module.Cache
        
        # 注册到 transformers 命名空间
        import transformers
        transformers.cache_utils = cache_utils_module
        print("✅ 已创建兼容性 cache_utils 模块")
    
    def _patch_indextts_source(self):
        """直接修补 IndexTTS 源代码文件"""
        import os
        import re
        
        print("🔧 修补 IndexTTS 源代码...")
        
        repo_path = "/tmp/index-tts"
        transformers_generation_utils_path = os.path.join(
            repo_path, "indextts", "gpt", "transformers_generation_utils.py"
        )
        
        if not os.path.exists(transformers_generation_utils_path):
            print("⚠️ 未找到 transformers_generation_utils.py 文件")
            return
        
        try:
            # 读取文件内容
            with open(transformers_generation_utils_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 备份原文件
            backup_path = transformers_generation_utils_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 修复导入语句
            old_import = "from transformers.cache_utils import ("
            new_import = "from transformers.utils.cache_utils import ("
            
            if old_import in content:
                content = content.replace(old_import, new_import)
                print("✅ 已修复 cache_utils 导入路径")
            else:
                # 尝试其他可能的修复
                patterns = [
                    (r"from transformers\.cache_utils import", "from transformers.utils.cache_utils import"),
                    (r"from transformers\.cache_utils", "from transformers.utils.cache_utils"),
                ]
                
                for pattern, replacement in patterns:
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        print(f"✅ 已修复导入模式: {pattern}")
                        break
            
            # 写回文件
            with open(transformers_generation_utils_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ 源代码修补完成")
            
        except Exception as e:
            print(f"❌ 修补源代码失败: {e}")
            # 恢复备份
            if os.path.exists(backup_path):
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(transformers_generation_utils_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
                print("🔄 已恢复原始文件")
    
    @modal.method()
    def generate(
        self, 
        text: str, 
        voice_bytes: bytes = None,
        emotion_bytes: bytes = None
    ) -> bytes:
        """生成语音"""
        import tempfile
        
        print(f"📝 生成: {text[:50]}...")
        
        voice_path = None
        emotion_path = None
        output_path = None
        
        try:
            # 保存参考语音
            if voice_bytes:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(voice_bytes)
                    voice_path = f.name
            
            # 保存情感参考
            if emotion_bytes:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(emotion_bytes)
                    emotion_path = f.name
            
            # 输出文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            
            # 生成
            self.tts.infer(
                voice=voice_path,
                text=text,
                output_path=output_path,
                emotion=emotion_path
            )
            
            # 读取结果
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            print(f"✅ 成功，{len(audio_data)} bytes")
            return audio_data
            
        finally:
            # 清理
            for path in [voice_path, emotion_path, output_path]:
                if path and os.path.exists(path):
                    os.unlink(path)
    
    @modal.fastapi_endpoint(method="POST")
    async def api(self, request):
        """HTTP API"""
        from fastapi.responses import Response
        import json
        
        form = await request.form()
        text = form.get("text")
        
        if not text:
            return Response(
                content=json.dumps({"error": "缺少 text 参数"}),
                status_code=400,
                media_type="application/json"
            )
        
        voice_file = form.get("voice")
        emotion_file = form.get("emotion")
        
        voice_bytes = await voice_file.read() if voice_file else None
        emotion_bytes = await emotion_file.read() if emotion_file else None
        
        try:
            audio_data = self.generate(text, voice_bytes, emotion_bytes)
            return Response(content=audio_data, media_type="audio/wav")
        except Exception as e:
            import traceback
            return Response(
                content=json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }),
                status_code=500,
                media_type="application/json"
            )

@app.local_entrypoint()
def main():
    """本地入口"""
    print("🚀 下载模型...")
    download_models.remote()
    
    print("\n✅ 部署完成!")
    print("\n📝 使用示例:")
    print('curl -X POST "https://YOUR-URL.modal.run" \\')
    print('  -F "text=你好世界" \\')
    print('  --output output.wav')

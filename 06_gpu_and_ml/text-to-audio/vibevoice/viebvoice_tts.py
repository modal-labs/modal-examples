import modal
import os
import re
from typing import List, Optional, Tuple, Dict

# --- 配置 ---
MODEL_REPO_ID = "vibevoice/VibeVoice-1.5B"
MODEL_CACHE_DIR = "/cache/models"
GPU_CONFIG = "A100-40GB"

# --- Modal App 定义 ---
app = modal.App("vibevoice-tts-final-fixed")
model_cache = modal.Volume.from_name("vibevoice-cache", create_if_missing=True)

# --- 环境镜像定义 ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1-dev")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "gradio",
        "soundfile",
        "accelerate",
        "huggingface_hub",
        "packaging",
        "uv",
    )
    .run_commands(
        "git clone https://github.com/vibevoice-community/VibeVoice.git /app || true",
        "cd /app && uv pip install -e . --system",
    )
    .workdir("/app")
)


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    scaledown_window=300,
    timeout=3600,
    volumes={MODEL_CACHE_DIR: model_cache}
)
class VibeVoiceModel:
    """
    一个封装了 VibeVoice 模型和处理器的类，用于生成语音。
    """
    model: object = None
    processor: object = None
    model_path: str = None

    @modal.enter()
    def load_model_and_processor(self):
        """
        在容器启动时加载模型和处理器，并将它们缓存到实例变量中。
        """
        import torch
        import sys
        import os

        try:
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference as VVModel
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        except ImportError as e:
            print(f"!!! 导入模块失败: {e}")
            print(f"    当前的 Python 路径 (sys.path): {sys.path}")
            print(f"    /app 目录下的内容: {os.listdir('/app') if os.path.exists('/app') else '不存在'}")
            if os.path.exists('/app/vibevoice'):
                print(f"    /app/vibevoice 目录下的内容: {os.listdir('/app/vibevoice')}")
            raise

        model_dir = os.path.join(MODEL_CACHE_DIR, MODEL_REPO_ID.split("/")[-1])
        if not os.path.exists(model_dir):
            print(f"本地缓存 {model_dir} 不存在，开始从 Hugging Face 下载...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=MODEL_REPO_ID,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            model_cache.commit()
            print("模型下载并缓存成功。")
        
        self.model_path = model_dir
        print(f"从路径 {self.model_path} 加载 VibeVoice 模型和处理器...")

        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        print("处理器加载成功。")

        self.model = VVModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True 
        )
        self.model.eval()
        print("模型加载成功！")

    @modal.method()
    def generate_speech(
        self,
        text: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> bytes:
        """
        根据输入文本生成语音的核心函数 (最终修复版)。
        此版本严格遵循社区仓库 Gradio 示例的逻辑，以确保稳定性。
        """
        import torch
        import torchaudio
        import tempfile
        import soundfile as sf
        import os

        if not self.model or not self.processor:
            raise RuntimeError("模型或处理器未初始化，请检查 load_model_and_processor 方法。")

        try:
            # ========== 核心修正：严格遵循官方 Gradio 示例的调用方式 ==========
            
            lines = text.strip().split('\n')
            if not any(':' in line for line in lines):
                 raise ValueError("输入文本格式不正确。请使用 '说话人: 内容' 的格式。")

            # Step 1: 为所有唯一说话人加载一次音频样本，存入字典方便查找
            speaker_waveforms = {}
            sample_rate = 24000
            
            # 先遍历一次，找出所有唯一的说话人
            unique_speakers = sorted(list(set(line.split(':', 1)[0].strip() for line in lines if ':' in line)))
            
            for speaker_name in unique_speakers:
                audio_path = f"demo/voices/{speaker_name.lower()}.wav"
                if os.path.exists(audio_path):
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != sample_rate:
                         from torchaudio.transforms import Resample
                         resampler = Resample(sr, sample_rate)
                         waveform = resampler(waveform)
                    speaker_waveforms[speaker_name] = waveform
                else:
                    print(f"Warning: 未找到 {speaker_name} 的音频文件。正在内存中创建默认静音张量。")
                    speaker_waveforms[speaker_name] = torch.zeros((1, sample_rate), dtype=torch.float32)

            # Step 2: 构建两个长度完全相同的列表: texts 和 voice_samples_list
            texts_for_processor = []
            voice_samples_for_processor = []

            for line in lines:
                if ':' in line:
                    try:
                        speaker_name, content = line.split(':', 1)
                        speaker_name = speaker_name.strip()
                        content = content.strip()
                        
                        if content: # 确保内容不为空
                            texts_for_processor.append(content)
                            voice_samples_for_processor.append(speaker_waveforms[speaker_name])
                    except (ValueError, KeyError) as e:
                        print(f"跳过格式不正确的行: '{line}' - 错误: {e}")
                        continue
            
            if not texts_for_processor:
                raise ValueError("无法从输入中解析出任何有效的对话行。")

            print(f"准备好的文本行数: {len(texts_for_processor)}")
            print(f"准备好的音频片段数: {len(voice_samples_for_processor)}")

            # Step 3: 使用 "texts" 和 "voice_samples_list" 参数调用 processor
            inputs = self.processor(
                texts=texts_for_processor, 
                voice_samples_list=voice_samples_for_processor,
                return_tensors="pt"
            )
            # ======================= 修正结束 =======================
            
            inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
            print("文本已成功处理为输入张量。")

            with torch.no_grad():
                print("调用 model.generate 开始生成音频...")
                audio_output = self.model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=temperature, top_p=top_p)
            
            print("音频生成完毕。")
            
            waveform_np = audio_output[0].cpu().to(torch.float32).numpy()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, waveform_np, samplerate=sample_rate)
                tmp_file.seek(0)
                audio_bytes = tmp_file.read()
            
            os.unlink(tmp_file.name)
            return audio_bytes

        except Exception as e:
            print(f"生成语音时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            raise

# --- API 和 Gradio 的代码保持不变 ---

@app.function(image=image, min_containers=1)
@modal.fastapi_endpoint(method="POST")
def api(text_data: dict):
    """一个简单的 POST API 入口点，用于程序化调用。"""
    from fastapi import status
    from fastapi.responses import JSONResponse, Response
    
    text = text_data.get("text")
    if not text:
        return JSONResponse(
            content={"error": "缺少 'text' 字段"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    
    model = VibeVoiceModel()
    try:
        audio_bytes = model.generate_speech.remote(text)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        # --- 核心修复 ---
        # 1. 将复杂的异常对象 e 转换为一个简单的字符串。
        error_message = f"在生成语音时发生错误: {e}"
        print(error_message) # 在服务器日志中打印详细错误，方便调试

        # 2. 将这个纯净的字符串放入响应体中。
        #    这样可以确保 jsonable_encoder 不会接触到原始的 e 对象。
        return JSONResponse(
            content={"error": error_message},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.function(image=image, timeout=1800)
def run_gradio_app():
    """
    启动 Gradio Web UI 的函数。
    """
    import gradio as gr
    
    model_instance = VibeVoiceModel()

    def generate_speech_ui(text, temperature, top_p):
        if not text or not text.strip():
            gr.Warning("请输入文本！")
            return None
            
        print("收到 Gradio 请求...")
        try:
            audio_bytes = model_instance.generate_speech.remote(
                text=text,
                temperature=temperature,
                top_p=top_p,
            )
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                return tmp_file.name
        except Exception as e:
            error_message = f"生成失败: {e}"
            print(error_message)
            gr.Error(error_message)
            return None

    with gr.Blocks(title="VibeVoice TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ VibeVoice - 长对话语音合成")
        gr.Markdown("一个高质量的文本转语音模型，支持多说话人、长时间对话和情感表达。")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="输入文本 (支持对话格式)",
                    placeholder="Alice: Hello, how are you?\nBob: I'm doing great, thanks for asking!",
                    lines=15,
                    value="Alice: Welcome to our podcast!\nBob: Thanks for having me, Alice.\nAlice: So tell us about your research."
                )
                with gr.Accordion("高级设置", open=False):
                     with gr.Row():
                        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature (随机性)")
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (核心采样)")
                
                generate_btn = gr.Button("🎵 生成语音", variant="primary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="生成的语音", type="filepath")
                with gr.Accordion("使用说明和示例", open=True):
                    gr.Markdown("""
                    ### 使用说明:
                    1. **格式**: 使用 `说话人姓名: 对话内容` 的格式，每句话占一行。
                    2. **语音文件**: 您无需准备任何文件。如果 `demo/voices/` 目录下有对应说话人名称的小写 `.wav` 文件 (例如 `alice.wav`)，程序会使用它。如果文件不存在，程序会自动创建一段静音作为替代。
                    3. **耐心**: 生成可能需要一些时间，特别是对于长文本。
                    
                    ### 示例:
                    ```
                    Alice: Welcome to our podcast!
                    Bob: Thanks for having me, Alice.
                    ```
                    """)
        
        generate_btn.click(
            fn=generate_speech_ui,
            inputs=[text_input, temperature, top_p],
            outputs=audio_output
        )

    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)


@app.local_entrypoint()
def main():
    print("正在启动 Gradio Web 界面... 这可能需要几分钟来构建镜像和下载模型。")
    run_gradio_app.remote()


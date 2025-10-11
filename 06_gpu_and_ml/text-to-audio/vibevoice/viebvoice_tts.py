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
        根据输入文本生成语音的核心函数。
        基于 VibeVoice 官方 Gradio demo 的实现方式。
        """
        import torch
        import tempfile
        import soundfile as sf
        import os
        import numpy as np

        if not self.model or not self.processor:
            raise RuntimeError("模型或处理器未初始化，请检查 load_model_and_processor 方法。")

        try:
            # 确保输入文本不为空
            if not text or not text.strip():
                raise ValueError("输入文本不能为空。")
            
            print(f"收到输入文本: {repr(text)}")
            
            # 解析对话文本并转换为 VibeVoice 格式
            lines = text.strip().split('\n')
            print(f"解析到的行数: {len(lines)}")
            
            # 检查是否有有效的对话格式
            valid_lines = [line for line in lines if ':' in line and line.strip()]
            if not valid_lines:
                raise ValueError("输入文本格式不正确。请使用 '说话人: 内容' 的格式，每句话占一行。")

            # 转换为 VibeVoice 格式：Speaker 0:, Speaker 1: 等
            formatted_script_lines = []
            speaker_mapping = {}  # 记录说话人名称到数字的映射
            speaker_counter = 0
            
            for line in valid_lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    speaker_name, content = line.split(':', 1)
                    speaker_name = speaker_name.strip()
                    content = content.strip()
                    
                    if content:  # 确保内容不为空
                        # 为每个新的说话人分配数字ID
                        if speaker_name not in speaker_mapping:
                            speaker_mapping[speaker_name] = speaker_counter
                            speaker_counter += 1
                        
                        speaker_id = speaker_mapping[speaker_name]
                        formatted_line = f"Speaker {speaker_id}: {content}"
                        formatted_script_lines.append(formatted_line)
                        print(f"转换: {line} -> {formatted_line}")
                except ValueError as e:
                    print(f"跳过格式不正确的行: '{line}' - 错误: {e}")
                    continue
            
            if not formatted_script_lines:
                raise ValueError("无法从输入中解析出任何有效的对话行。")

            formatted_script = '\n'.join(formatted_script_lines)
            print(f"格式化后的脚本: {formatted_script}")
            
            # 准备音频样本 - 完全按照官方代码的 read_audio 函数
            sample_rate = 24000
            voice_samples = []
            
            def read_audio(audio_path: str, target_sr: int = 24000):
                """按照官方代码的 read_audio 函数实现"""
                try:
                    import soundfile as sf
                    import librosa
                    wav, sr = sf.read(audio_path)
                    if len(wav.shape) > 1:
                        wav = np.mean(wav, axis=1)  # 转换为单声道
                    if sr != target_sr:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                    return wav
                except Exception as e:
                    print(f"Error reading audio {audio_path}: {e}")
                    return np.array([])
            
            # 定义可用的音频文件映射
            available_voices = {
                'Alice': 'en-Alice_woman.wav',
                'alice': 'en-Alice_woman.wav',
                'Carter': 'en-Carter_man.wav',
                'carter': 'en-Carter_man.wav',
                'Frank': 'en-Frank_man.wav',
                'frank': 'en-Frank_man.wav',
                'Maya': 'en-Maya_woman.wav',
                'maya': 'en-Maya_woman.wav',
                'Samuel': 'in-Samuel_man.wav',
                'samuel': 'in-Samuel_man.wav',
                'Anchen': 'zh-Anchen_man_bgm.wav',
                'anchen': 'zh-Anchen_man_bgm.wav',
                'Bowen': 'zh-Bowen_man.wav',
                'bowen': 'zh-Bowen_man.wav',
                'Xinran': 'zh-Xinran_woman.wav',
                'xinran': 'zh-Xinran_woman.wav',
            }
            
            for speaker_name in speaker_mapping.keys():
                # 首先尝试直接映射到可用的音频文件
                mapped_filename = available_voices.get(speaker_name)
                
                if mapped_filename:
                    # 使用映射的文件名
                    possible_paths = [
                        f"/app/demo/voices/{mapped_filename}",
                        f"demo/voices/{mapped_filename}",
                    ]
                    print(f"为说话人 '{speaker_name}' 使用映射文件: {mapped_filename}")
                else:
                    # 如果没有映射，尝试多种可能的音频文件路径格式
                    possible_paths = [
                        f"/app/demo/voices/{speaker_name}.wav",  # 原始格式
                        f"/app/demo/voices/{speaker_name.lower()}.wav",  # 小写格式
                        f"/app/demo/voices/{speaker_name.replace('_', '-')}.wav",  # 下划线转横线
                        f"/app/demo/voices/{speaker_name.lower().replace('_', '-')}.wav",  # 小写+下划线转横线
                        f"demo/voices/{speaker_name}.wav",  # 相对路径格式
                        f"demo/voices/{speaker_name.lower()}.wav",  # 相对路径小写格式
                    ]
                    print(f"为说话人 '{speaker_name}' 尝试通用路径匹配")
                
                audio_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        audio_path = path
                        break
                
                if audio_path:
                    audio_data = read_audio(audio_path)
                    if len(audio_data) == 0:
                        print(f"Warning: 无法读取 {speaker_name} 的音频文件。使用默认静音。")
                        # 创建1秒的静音
                        audio_data = np.zeros(sample_rate, dtype=np.float32)
                    voice_samples.append(audio_data)
                    print(f"加载了 {speaker_name} 的音频文件 ({audio_path})，长度: {len(audio_data)}")
                else:
                    print(f"Warning: 未找到 {speaker_name} 的音频文件。尝试的路径: {possible_paths}")
                    # 创建1秒的静音
                    audio_data = np.zeros(sample_rate, dtype=np.float32)
                    voice_samples.append(audio_data)
            
            print(f"准备了 {len(voice_samples)} 个音频样本")
            
            # 添加调试信息：检查音频样本是否为空
            for i, sample in enumerate(voice_samples):
                if len(sample) == 0:
                    print(f"警告: 音频样本 {i} 为空")
                elif np.all(sample == 0):
                    print(f"警告: 音频样本 {i} 全为零（静音）")
                else:
                    print(f"音频样本 {i} 正常，长度: {len(sample)}, 非零值数量: {np.count_nonzero(sample)}")
            
            # 使用官方格式调用 processor
            print("使用官方格式调用 processor...")
            inputs = self.processor(
                text=[formatted_script],  # 注意：这里是列表
                voice_samples=[voice_samples],  # 注意：这里也是列表
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            print("文本已成功处理为输入张量。")
            print(f"Processor 输出键: {list(inputs.keys())}")
            
            # 将输入移动到模型设备 - 不修改形状，让 processor 自己处理
            processed_inputs = {}
            for key, val in inputs.items():
                if torch.is_tensor(val):
                    processed_inputs[key] = val.to(self.model.device)
                    print(f"移动张量到设备: {key} -> {val.shape}")
                else:
                    processed_inputs[key] = val  # 保持非张量值不变
                    print(f"保持非张量值: {key} = {type(val)}")
            
            # 检查是否有 input_ids
            if 'input_ids' not in processed_inputs:
                print("警告: 没有找到 input_ids，检查 processor 输出...")
                for key, val in processed_inputs.items():
                    print(f"  {key}: {type(val)} - {val if not torch.is_tensor(val) else val.shape}")
                # 尝试使用其他可能的键
                if 'input_token_ids' in processed_inputs:
                    processed_inputs['input_ids'] = processed_inputs['input_token_ids']
                    print("使用 input_token_ids 作为 input_ids")
                elif 'token_ids' in processed_inputs:
                    processed_inputs['input_ids'] = processed_inputs['token_ids']
                    print("使用 token_ids 作为 input_ids")
                else:
                    raise ValueError("无法找到 input_ids 或等效的输入键")

            # 生成音频 - 使用官方参数并添加更多控制
            with torch.no_grad():
                print("调用 model.generate 开始生成音频...")
                print(f"传递给模型的键: {list(processed_inputs.keys())}")
                
                # 计算合适的 max_new_tokens
                input_length = processed_inputs['input_ids'].shape[1]
                max_length = min(2048, input_length + 1000)  # 限制最大长度
                print(f"输入长度: {input_length}, 最大生成长度: {max_length}")
                
                audio_output = self.model.generate(
                    **processed_inputs,
                    max_new_tokens=max_length - input_length,  # 明确设置生成长度
                    cfg_scale=1.3,  # 使用官方默认值
                    tokenizer=self.processor.tokenizer,
                    generation_config={
                        'do_sample': False,  # 使用官方设置
                        'max_length': max_length,
                        'min_length': input_length + 100,  # 确保生成足够的长度
                    },
                    verbose=True,  # 开启详细输出
                    refresh_negative=True,
                )
            
            print("音频生成完毕。")
            
            # 处理 VibeVoiceGenerationOutput 对象
            if hasattr(audio_output, 'audio_values'):
                # 如果输出有 audio_values 属性
                waveform_tensor = audio_output.audio_values
            elif hasattr(audio_output, 'sequences'):
                # 如果输出有 sequences 属性
                waveform_tensor = audio_output.sequences
            elif isinstance(audio_output, tuple):
                # 如果是元组，取第一个元素
                waveform_tensor = audio_output[0]
            else:
                # 直接使用输出
                waveform_tensor = audio_output
            
            # 确保是张量并转换为 numpy
            if torch.is_tensor(waveform_tensor):
                waveform_np = waveform_tensor.cpu().to(torch.float32).numpy()
            else:
                # 如果不是张量，尝试直接转换
                waveform_np = np.array(waveform_tensor, dtype=np.float32)
            
            print(f"音频数据形状: {waveform_np.shape}")
            
            # 确保是2D数组
            if len(waveform_np.shape) == 1:
                waveform_np = waveform_np.reshape(1, -1)
            elif len(waveform_np.shape) > 2:
                waveform_np = waveform_np.reshape(1, -1)

            # 保存为临时文件并返回字节
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
    text = text_data.get("text")
    if not text:
        from fastapi import status
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={"error": "缺少 'text' 字段"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    
    model = VibeVoiceModel()
    try:
        audio_bytes = model.generate_speech.remote(text)
        from fastapi.responses import Response
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        from fastapi import status
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={"error": str(e)},
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
        print(f"输入文本: {repr(text)}")
        print(f"Temperature: {temperature}, Top-p: {top_p}")
        
        try:
            audio_bytes = model_instance.generate_speech.remote(
                text=text,
                temperature=temperature,
                top_p=top_p,
            )
            
            if audio_bytes is None:
                gr.Error("生成的音频为空")
                return None
                
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
                    placeholder="Alice: Hello, how are you?\nCarter: I'm doing great, thanks for asking!",
                    lines=15,
                    value="Alice: Welcome to our podcast!\nCarter: Thanks for having me, Alice.\nAlice: So tell us about your research."
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
                    2. **可用说话人**: Alice, Carter, Frank, Maya, Samuel, Anchen, Bowen, Xinran
                    3. **语音文件**: 程序会自动使用预置的音频文件，无需额外准备。
                    4. **耐心**: 生成可能需要一些时间，特别是对于长文本。
                    
                    ### 示例:
                    ```
                    Alice: Welcome to our podcast!
                    Carter: Thanks for having me, Alice.
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


# 测试函数 - 可以在本地运行来验证文本解析逻辑
def test_text_parsing():
    """测试文本解析逻辑"""
    test_text = """Alice: Welcome to our podcast!
Bob: Thanks for having me, Alice.
Alice: So tell us about your research."""
    
    lines = test_text.strip().split('\n')
    print(f"解析到的行数: {len(lines)}")
    
    valid_lines = [line for line in lines if ':' in line and line.strip()]
    print(f"有效行数: {len(valid_lines)}")
    
    # 提取说话人和文本
    speaker_names = []
    texts = []
    
    for line in valid_lines:
        try:
            speaker_name, content = line.split(':', 1)
            speaker_name = speaker_name.strip()
            content = content.strip()
            
            if content:  # 确保内容不为空
                speaker_names.append(speaker_name)
                texts.append(content)
                print(f"添加对话: {speaker_name} -> {content}")
        except ValueError as e:
            print(f"跳过格式不正确的行: '{line}' - 错误: {e}")
            continue
    
    print(f"说话人列表: {speaker_names}")
    print(f"最终文本列表: {texts}")
    print(f"文本列表长度: {len(texts)}")
    print(f"说话人列表长度: {len(speaker_names)}")
    print("测试完成！")


if __name__ == "__main__":
    # 如果直接运行此文件，先测试文本解析
    test_text_parsing()


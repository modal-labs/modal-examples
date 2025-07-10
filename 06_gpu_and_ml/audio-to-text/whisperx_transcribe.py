"""
WhisperX transcription with word-level timestamps.

Usage:
    modal run whisperx_transcribe.py --audio-file "audio.wav"
    modal run whisperx_transcribe.py --audio-file "audio.wav" --stream
"""

import os
from typing import Dict, List
from dataclasses import dataclass

import modal

MODEL_CACHE_DIR = "/whisperx-cache"

image = (
     modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ffmpeg")
    .apt_install("libcudnn8")
    .apt_install("libcudnn8-dev")
    .pip_install(
        "whisperx",
        "numpy",
        "scipy",
    )
    .env({"HF_HUB_CACHE": MODEL_CACHE_DIR})
)


app = modal.App("whisperx-example", image=image)
models_volume = modal.Volume.from_name("whisperx-models", create_if_missing=True)


@dataclass
class ChunkInput:
    chunk_bytes: bytes
    chunk_index: int
    start_time: float
    end_time: float
    sample_rate: int


@dataclass
class ChunkResult:
    chunk_index: int
    segments: List[Dict]
    language: str
    start_time: float
    end_time: float


@app.cls(
    gpu="A100",
    image=image,
    volumes={MODEL_CACHE_DIR: models_volume},
    timeout=1800,
    min_containers=0,
    max_containers=60,
    scaledown_window=30,
)
class WhisperX:
    device: str = modal.parameter(default="cuda")
        
    @modal.enter()
    def setup(self):
        import os
        os.environ["TORCH_HOME"] = MODEL_CACHE_DIR
        
        import whisperx
        import torch
        
        self.model = None
        
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
            self.device = "cpu"
            
        print("Loading WhisperX model...")
        self.model = whisperx.load_model(
            "large-v2", 
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
            download_root=MODEL_CACHE_DIR
        )
        print("Model loaded!")
    
    @modal.method()
    def transcribe(self, audio_data: bytes) -> Dict:
        import whisperx
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            audio = whisperx.load_audio(temp_audio_path)
            result = self.model.transcribe(audio, batch_size=16, language="en")
            
            language = result.get("language", "en")
            if len(result["segments"]) > 0:
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=language, 
                        device=self.device,
                        model_dir=MODEL_CACHE_DIR
                    )
                    result = whisperx.align(result["segments"], align_model, metadata, audio, self.device)
                except:
                    print("Alignment failed, using segment-level timestamps")
            
            return {
                "language": language,
                "segments": result["segments"],
                "duration": len(audio) / 16000,
            }
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    @modal.method()
    def process_chunk(self, chunk_input: ChunkInput) -> ChunkResult:
        import whisperx
        import tempfile
        import scipy.io.wavfile as wav
        import io
        
        with io.BytesIO(chunk_input.chunk_bytes) as bio:
            _, chunk_audio = wav.read(bio)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            wav.write(temp_audio.name, chunk_input.sample_rate, chunk_audio)
            temp_audio_path = temp_audio.name
        
        try:
            audio = whisperx.load_audio(temp_audio_path)
            result = self.model.transcribe(audio, batch_size=16, language="en")
            
            language = result.get("language", "en")
            if len(result.get("segments", [])) > 0:
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self.device,
                        model_dir=MODEL_CACHE_DIR
                    )
                    result = whisperx.align(result["segments"], align_model, metadata, audio, self.device)
                except:
                    pass
            
            adjusted_segments = []
            for segment in result.get("segments", []):
                adjusted_segment = segment.copy()
                adjusted_segment["start"] = segment.get("start", 0) + chunk_input.start_time
                adjusted_segment["end"] = segment.get("end", 0) + chunk_input.start_time
                adjusted_segments.append(adjusted_segment)
            
            return ChunkResult(
                chunk_index=chunk_input.chunk_index,
                segments=adjusted_segments,
                language=language,
                start_time=chunk_input.start_time,
                end_time=chunk_input.end_time
            )
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


def chunk_generator(audio_file: str, chunk_seconds: float):
    import scipy.io.wavfile as wav
    import io
    import warnings
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sample_rate, audio_data = wav.read(audio_file)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
    
    total_samples = len(audio_data)
    chunk_samples = int(chunk_seconds * sample_rate)
    
    chunk_index = 0
    position = 0
    
    while position < total_samples:
        chunk_end = min(position + chunk_samples, total_samples)
        chunk_audio = audio_data[position:chunk_end]
        
        bio = io.BytesIO()
        wav.write(bio, sample_rate, chunk_audio)
        chunk_bytes = bio.getvalue()
        
        start_time = position / sample_rate
        end_time = chunk_end / sample_rate
        
        yield ChunkInput(
            chunk_bytes=chunk_bytes,
            chunk_index=chunk_index,
            start_time=start_time,
            end_time=end_time,
            sample_rate=sample_rate
        )
        
        position = chunk_end
        chunk_index += 1


def aggregate_and_display_stats(transcription_data: Dict, base_filename: str = "transcript"):
    segments = transcription_data.get('segments', [])
    
    if not segments:
        print("No segments found in transcription")
        return
    
    full_text = []
    for segment in segments:
        text = segment.get('text', '').strip()
        if text:
            full_text.append(text)
    
    aggregated_text = ' '.join(full_text)
    
    plain_file = f"{base_filename}.txt"
    with open(plain_file, 'w', encoding='utf-8') as f:
        f.write(aggregated_text)
    
    formatted_file = f"{base_filename}_with_timestamps.txt"
    with open(formatted_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription\n")
        f.write(f"Language: {transcription_data.get('language', 'unknown')}\n")
        if 'duration' in transcription_data:
            f.write(f"Duration: {transcription_data['duration']:.2f} seconds\n")
        f.write("=" * 50 + "\n\n")
        
        for segment in segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            if text:
                f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
    
    word_count = len(aggregated_text.split())
    char_count = len(aggregated_text)
    
    total_speaking_time = 0
    for segment in segments:
        if segment.get('text', '').strip():
            duration = segment.get('end', 0) - segment.get('start', 0)
            total_speaking_time += duration
    
    print("\n" + "=" * 50)
    print("📊 TRANSCRIPTION STATISTICS")
    print("=" * 50)
    print(f"📝 Files created:")
    print(f"   - Plain text: {plain_file}")
    print(f"   - With timestamps: {formatted_file}")
    print(f"\n📈 Content stats:")
    print(f"   - Total segments: {len(segments)}")
    print(f"   - Total words: {word_count:,}")
    print(f"   - Total characters: {char_count:,}")
    print(f"   - Speaking time: {total_speaking_time:.1f}s ({total_speaking_time/60:.1f} minutes)")
    
    if total_speaking_time > 0:
        words_per_minute = (word_count / total_speaking_time) * 60
        print(f"   - Speaking rate: {words_per_minute:.0f} words/minute")
    
    print("=" * 50 + "\n")


"""
Usage:
    modal run example.py --audio-file "audio.wav" # if you want to transcribe a single file
    modal run example.py --audio-file "audio.wav" --stream # if you want to stream the audio in chunks and process them in parallel

"""
@app.local_entrypoint()
def main(
    audio_file: str = None,
    stream: bool = False,
):
    import time
    
    if not audio_file:
        print("Error: Provide --audio-file")
        return
    
    transcriber = WhisperX()
    time_taken = 0
    
    if stream:
        start_time = time.time()    
        print("Streaming mode: Processing audio in parallel chunks...")
        
        chunk_seconds = 120.0
        
        chunk_gen = chunk_generator(audio_file, chunk_seconds)
        results = list(transcriber.process_chunk.map(chunk_gen))
        
        results.sort(key=lambda r: r.chunk_index)
        
        time_taken = time.time() - start_time
        
        all_segments = []
        language = "en"
        
        for result in results:
            all_segments.extend(result.segments)
            if result.language:
                language = result.language
        
        print(f"\nLanguage: {language}")
        print(f"Duration: {len(all_segments) / 16000:.2f} seconds")
        print(f"Time taken: {time_taken:.2f} seconds")
        
        transcription_data = {
            "language": language,
            "segments": all_segments
        }
        
        import json
        with open("transcription.json", "w") as f:
            json.dump(transcription_data, f, indent=2)
        print(f"\nSaved to: transcription.json")
        
        aggregate_and_display_stats(transcription_data)
        
    else:
        print(f"Full-file mode: Reading {audio_file}")
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        print("Transcribing...")
        result = transcriber.transcribe.remote(audio_data)
        
        print(f"\nLanguage: {result['language']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Time taken: {time_taken:.2f} seconds")
        
        import json
        with open("transcription.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: transcription.json")
        
        aggregate_and_display_stats(result)
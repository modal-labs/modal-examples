"""
Qwen2.5-VL Video Analysis API with Modal

This example demonstrates deploying Qwen2.5-VL-7B-Instruct on Modal for:
- Video understanding (up to 1 hour+ videos)
- Temporal event localization with timestamps
- Structured data extraction from video frames
- Multi-modal analysis (video + text queries)

Qwen2.5-VL is SOTA for vision-language tasks, matching GPT-4o in video understanding
while being fully open-source and deployable on Modal's serverless infrastructure.
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("qwen25vl-video-analysis")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.45.0",
        "accelerate==0.34.0",
        "qwen-vl-utils==0.0.8",
        "Pillow==10.4.0",
        "av==13.1.0",  # For video processing
        "requests==2.32.3",
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev",
    )
)

# Download model weights at build time for faster cold starts
with image.imports():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import av
    from PIL import Image
    import io


@app.cls(
    image=image,
    gpu="A10G",  # Qwen2.5-VL-7B works well on A10G
    timeout=600,
    container_idle_timeout=300,
)
class Qwen25VLModel:
    """
    Qwen2.5-VL model for video and image analysis.
    
    Features:
    - Video understanding with temporal localization
    - Structured output generation (JSON, bounding boxes)
    - Multi-frame analysis for long videos
    - OCR and document parsing from video frames
    """

    @modal.build()
    def download_model(self):
        """Download model weights during container build."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Download model and processor
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        AutoProcessor.from_pretrained(model_id)
        
        print(f"✓ Downloaded {model_id}")

    @modal.enter()
    def load_model(self):
        """Load model into GPU memory when container starts."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Load with flash attention 2 for better performance
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.model.device}")

    def _extract_video_frames(
        self, 
        video_bytes: bytes, 
        max_frames: int = 32,
        fps: float = None
    ) -> list[Image.Image]:
        """
        Extract frames from video bytes.
        
        Args:
            video_bytes: Raw video file bytes
            max_frames: Maximum number of frames to extract
            fps: Target FPS for frame extraction (None = auto)
            
        Returns:
            List of PIL Images
        """
        import av
        from PIL import Image
        import io
        
        container = av.open(io.BytesIO(video_bytes))
        frames = []
        
        # Get video stream
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        # Calculate frame sampling rate
        if fps is None:
            # Sample uniformly across video
            step = max(1, total_frames // max_frames)
        else:
            # Sample at specific FPS
            original_fps = float(stream.average_rate)
            step = max(1, int(original_fps / fps))
        
        # Extract frames
        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0 and len(frames) < max_frames:
                img = frame.to_image()
                frames.append(img)
        
        container.close()
        return frames

    @modal.method()
    def analyze_video(
        self,
        video_bytes: bytes,
        query: str,
        max_frames: int = 32,
        output_format: str = "text",  # "text" or "json"
    ) -> dict:
        """
        Analyze video with natural language query.
        
        Args:
            video_bytes: Raw video file bytes
            query: Natural language question about the video
            max_frames: Max frames to analyze (higher = more detail, slower)
            output_format: "text" for narrative, "json" for structured data
            
        Returns:
            Dict with analysis results and metadata
        """
        from qwen_vl_utils import process_vision_info
        
        # Extract frames from video
        frames = self._extract_video_frames(video_bytes, max_frames=max_frames)
        
        # Prepare prompt based on output format
        if output_format == "json":
            query_prompt = f"{query}\n\nProvide your response in JSON format."
        else:
            query_prompt = query
        
        # Build messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "video", "video": frames}],
                    {"type": "text", "text": query_prompt},
                ],
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Deterministic for structured outputs
            )
        
        # Trim and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return {
            "query": query,
            "response": response,
            "frames_analyzed": len(frames),
            "output_format": output_format,
        }

    @modal.method()
    def detect_events(
        self,
        video_bytes: bytes,
        event_query: str,
        max_frames: int = 64,
    ) -> dict:
        """
        Detect and localize specific events in video with timestamps.
        
        Args:
            video_bytes: Raw video file bytes
            event_query: Description of event to detect (e.g., "person entering room")
            max_frames: Frames to analyze for temporal localization
            
        Returns:
            Dict with detected events and approximate timestamps
        """
        frames = self._extract_video_frames(video_bytes, max_frames=max_frames)
        
        query = f"""Analyze this video and identify when the following event occurs: '{event_query}'

For each occurrence, provide:
1. Approximate timestamp (based on frame sequence)
2. Brief description of what's happening
3. Confidence level (high/medium/low)

Output in JSON format with structure:
{{
    "events": [
        {{"timestamp": "00:00:15", "description": "...", "confidence": "high"}},
        ...
    ]
}}"""
        
        result = self.analyze_video(
            video_bytes=video_bytes,
            query=query,
            max_frames=max_frames,
            output_format="json",
        )
        
        return result

    @modal.method()
    def extract_ocr_from_video(
        self,
        video_bytes: bytes,
        max_frames: int = 16,
    ) -> dict:
        """
        Extract all text visible in video frames.
        
        Args:
            video_bytes: Raw video file bytes
            max_frames: Frames to analyze for OCR
            
        Returns:
            Dict with extracted text and frame references
        """
        query = """Extract all visible text from this video.

Output in JSON format:
{
    "text_segments": [
        {"frame_approx": 1, "text": "...", "location": "top-left/center/etc"},
        ...
    ],
    "summary": "Brief summary of text content"
}"""
        
        result = self.analyze_video(
            video_bytes=video_bytes,
            query=query,
            max_frames=max_frames,
            output_format="json",
        )
        
        return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_sample_video() -> bytes:
    """Download a sample video for testing."""
    import requests
    
    # Sample video URL (replace with actual test video)
    url = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
    
    response = requests.get(url)
    response.raise_for_status()
    
    return response.content


@app.local_entrypoint()
def main():
    """Demo: Analyze a sample video."""
    
    # Download sample video
    print("📥 Downloading sample video...")
    video_bytes = download_sample_video.remote()
    print(f"✓ Video downloaded: {len(video_bytes)} bytes")
    
    # Initialize model
    model = Qwen25VLModel()
    
    # Example 1: Basic video analysis
    print("\n🎬 Analyzing video content...")
    result = model.analyze_video.remote(
        video_bytes=video_bytes,
        query="Describe what happens in this video in detail.",
        max_frames=16,
    )
    print(f"Response: {result['response']}")
    
    # Example 2: Event detection
    print("\n🔍 Detecting specific events...")
    events = model.detect_events.remote(
        video_bytes=video_bytes,
        event_query="any scene changes or transitions",
        max_frames=32,
    )
    print(f"Events detected: {events['response']}")
    
    # Example 3: OCR extraction
    print("\n📝 Extracting text from video...")
    ocr_result = model.extract_ocr_from_video.remote(
        video_bytes=video_bytes,
        max_frames=8,
    )
    print(f"OCR results: {ocr_result['response']}")


# Web endpoint for API access
@app.function(image=image)
@modal.web_endpoint(method="POST")
def api_analyze_video(video_file: modal.web_endpoint.File, query: str):
    """
    Web API endpoint for video analysis.
    
    Usage:
        curl -X POST https://your-modal-url.modal.run/api_analyze_video \
             -F "video_file=@video.mp4" \
             -F "query=What happens in this video?"
    """
    model = Qwen25VLModel()
    
    video_bytes = video_file.content
    
    result = model.analyze_video.remote(
        video_bytes=video_bytes,
        query=query,
        max_frames=32,
    )
    
    return result

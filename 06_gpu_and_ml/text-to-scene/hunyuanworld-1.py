from pathlib import Path

import modal

app = modal.App("example-hunyuanworld-1")

flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .entrypoint([])  # removes chatty prints on entry
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1",
        "libopenmpi-dev",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
        "wget",
    )
    .uv_pip_install(
        "torchaudio==2.5.0",
        "torch==2.5.0",
        "torchvision==0.20.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .uv_pip_install(
        "av==14.3.0",
        "cmake==4.1.0",
        "einops==0.4.1",
        "hydra-core==1.1.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.4.9",
        "kornia==0.8.0",
        "matplotlib==3.10.1",
        "numpy==1.24.1",
        "omegaconf==2.1.2",
        "open3d>=0.18.0",
        "opencv-python==4.11.0.86",
        "opencv-python-headless==4.11.0.86",
        "pillow==11.1.0",
        "plyfile==1.1",
        "py360convert==1.0.3",
        "pyyaml==6.0.2",
        "scikit-image==0.24.0",
        "scikit-learn==1.6.1",
        "scipy==1.15.2",
        "timm==1.0.13",
        "tqdm==4.67.1",
        "trimesh>=4.6.1",
        "wheel==0.45.1",
        "albumentations==0.5.2",
        "accelerate==1.6.0",
        "diffusers==0.34.0",
        "huggingface-hub==0.30.2",
        "loguru==0.7.3",
        "onnx==1.17.0",
        "onnxruntime-gpu==1.21.1",
        "open-clip-torch==2.30.0",
        "peft==0.15.0",
        "rich==14.0.0",
        "sageattention==1.0.6",
        "safetensors==0.5.3",
        "segment-anything==1.0",
        "sentencepiece==0.2.0",
        "submitit==1.4.2",
        "tokenizers==0.21.1",
        "transformers==4.51.0",
        "ultralytics==8.3.74",
        "webdataset==0.2.100",
        "xformers==0.0.28.post2",
        flash_attn_wheel,
        "basicsr>=1.4.2",
        "facexlib>=0.2.5",
        "gfpgan>=1.3.5",
    )
    .env({"CC": "gcc", "CXX": "g++"})  # force use of g++ instead of clang
    .uv_pip_install(
        "moge @ git+https://github.com/microsoft/MoGe.git",
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        "cd /tmp && git clone https://github.com/xinntao/Real-ESRGAN.git",
        "cd /tmp/Real-ESRGAN && python setup.py develop",
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py",  # fix basicsr import for torchvision 0.20.0 compatibility
    )
    .run_commands(
        "cd /tmp && git clone https://github.com/naver-ai/ZIM.git",
        "cd /tmp/ZIM && uv pip install --system --compile-bytecode easydict",
        "cd /tmp/ZIM && uv pip install --system --compile-bytecode -e .",
    )
    .run_commands(
        "cd /root && git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git",
    )
    .run_commands(
        "mkdir -p /root/HunyuanWorld-1.0/ZIM/zim_vit_l_2092",
        "cd /root/HunyuanWorld-1.0/ZIM/zim_vit_l_2092 && wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx",
        "cd /root/HunyuanWorld-1.0/ZIM/zim_vit_l_2092 && wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx",
    )
    .run_commands(
        "cd /tmp && git clone https://github.com/google/draco.git",
        "cd /tmp/draco && mkdir build && cd build && cmake .. && make -j$(nproc) && make install",
        "ldconfig",
    )
)

hf_cache = modal.Volume.from_name("hunyuanworld-1-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("hunyuanworld-1-output", create_if_missing=True)

hf_cache_path = Path("/root/.cache/huggingface")
output_dir = Path("/output")

minutes = 60


@app.cls(
    image=image,
    gpu="h200",
    volumes={hf_cache_path: hf_cache, output_dir: output_volume},
    timeout=60 * minutes,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class HunyuanWorld:
    @modal.enter()
    def setup(self):
        import torch
        from hy3dworld import (
            LayerDecomposition,
            Text2PanoramaPipelines,
            WorldComposer,
        )
        from hy3dworld.AngelSlim.attention_quantization_processor import (
            FluxFp8AttnProcessor2_0,
        )
        from hy3dworld.AngelSlim.gemm_quantization_processor import FluxFp8GeMMProcessor

        self.seed = 42
        self.width = 1920

        # panorama

        self.height = 960

        self.guidance_scale = 30
        self.shifting_extend = 0
        self.num_inference_steps = 50
        self.true_cfg_scale = 0.0
        self.blend_extend = 6

        self.lora_path = "tencent/HunyuanWorld-1"
        self.model_path = "black-forest-labs/FLUX.1-dev"

        self.pipe = Text2PanoramaPipelines.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        )
        self.pipe.load_lora_weights(
            self.lora_path,
            subfolder="HunyuanWorld-PanoDiT-Text",
            weight_name="lora.safetensors",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()
        self.pipe.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
        FluxFp8GeMMProcessor(self.pipe.transformer)

        # scene generation

        self.export_drc = False

        target_size = 3840
        kernel_scale = max(1, int(target_size / self.width))

        class DummyObj:
            def __init__(self):
                self.cache = True

        self.LayerDecomposer = LayerDecomposition(DummyObj())
        self.hy3d_world = WorldComposer(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            resolution=(target_size, target_size // 2),
            seed=self.seed,
            filter_mask=True,
            kernel_scale=kernel_scale,
        )
        self.LayerDecomposer.inpaint_fg_model.transformer.set_attn_processor(
            FluxFp8AttnProcessor2_0()
        )
        self.LayerDecomposer.inpaint_sky_model.transformer.set_attn_processor(
            FluxFp8AttnProcessor2_0()
        )
        FluxFp8GeMMProcessor(self.LayerDecomposer.inpaint_fg_model.transformer)
        FluxFp8GeMMProcessor(self.LayerDecomposer.inpaint_sky_model.transformer)

        print("✓ Models loaded!")

    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        labels_fg1: str = "",
        labels_fg2: str = "",
        classes: str = "outdoor",
    ):
        from datetime import datetime

        import open3d as o3d
        import torch
        from hy3dworld import process_file
        from hy3dworld.AngelSlim.cache_helper import DeepCacheHelper
        from PIL import Image

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generation_dir = output_dir / timestamp
        generation_dir.mkdir(parents=True, exist_ok=True)

        # generate panorama

        helper = DeepCacheHelper(
            pipe_model=self.pipe.transformer,
            no_cache_steps=list(range(0, 10))
            + list(range(10, 40, 3))
            + list(range(40, 50)),
            no_cache_block_id={"single": [38]},
        )
        helper.start_timestep = 0
        helper.enable()

        image = self.pipe(
            prompt,
            height=self.height,
            width=self.width,
            negative_prompt=negative_prompt,
            generator=torch.Generator("cpu").manual_seed(self.seed),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            blend_extend=self.blend_extend,
            true_cfg_scale=self.true_cfg_scale,
            helper=helper,
        ).images[0]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        panorama_path = generation_dir / "panorama.png"
        image.save(panorama_path)
        print(f"Panorama saved to: {panorama_path}")

        # generate scene

        # foreground layer information
        fg1_infos = [
            {
                "image_path": str(panorama_path),
                "output_path": str(generation_dir),
                "labels": labels_fg1.split(),
                "class": classes,
            }
        ]
        fg2_infos = [
            {
                "image_path": str(generation_dir / "remove_fg1_image.png"),
                "output_path": str(generation_dir),
                "labels": labels_fg2.split(),
                "class": classes,
            }
        ]

        # layer decompose
        self.LayerDecomposer(fg1_infos, layer=0)
        self.LayerDecomposer(fg2_infos, layer=1)
        self.LayerDecomposer(fg2_infos, layer=2)
        separate_pano, fg_bboxes = self.hy3d_world._load_separate_pano_from_dir(
            str(generation_dir), sr=True
        )

        # layer-wise reconstruction
        layered_world_mesh = self.hy3d_world.generate_world(
            separate_pano=separate_pano, fg_bboxes=fg_bboxes, world_type="mesh"
        )

        # export scenes
        scenes_dir = generation_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)

        scene_files = []
        for layer_idx, layer_info in enumerate(layered_world_mesh):
            output_path = scenes_dir / f"mesh_layer{layer_idx}.ply"
            o3d.io.write_triangle_mesh(str(output_path), layer_info["mesh"])
            scene_files.append(str(output_path))
            print(f"Saved layer {layer_idx} to: {output_path}")

            if self.export_drc:
                output_path_drc = scenes_dir / f"mesh_layer{layer_idx}.drc"
                process_file(str(output_path), str(output_path_drc))
                print(f"Saved layer {layer_idx} to: {output_path_drc}")


# web

max_inputs = 1000


@app.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi")
    .add_local_file(Path(__file__).parent / "index.html", "/index.html")
    .add_local_file(Path(__file__).parent / "scene.js", "/scene.js"),
    volumes={output_dir: output_volume},
    timeout=24 * 60 * minutes,
)
@modal.concurrent(max_inputs=max_inputs)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse

    web_app = FastAPI()

    @web_app.get("/")
    async def index():
        return FileResponse("/index.html")

    @web_app.get("/scene.js")
    async def scene_js():
        return FileResponse("/scene.js", media_type="application/javascript")

    @web_app.post("/generate")
    async def generate(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        labels_fg1 = data.get("labels_fg1", "stones")
        labels_fg2 = data.get("labels_fg2", "trees mountains")
        classes = data.get("classes", "outdoor")

        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        hunyuan = HunyuanWorld()

        hunyuan.generate.remote(
            prompt,
            negative_prompt,
            labels_fg1,
            labels_fg2,
            classes,
        )
        return JSONResponse({}, status_code=200)

    @web_app.get("/scenes/{timestamp}/{filename}")
    async def get_scene(timestamp: str, filename: str):
        file_path = output_dir / timestamp / "scenes" / filename
        if file_path.exists():
            return FileResponse(file_path, media_type="model/ply")
        return JSONResponse({"error": "File not found"}, status_code=404)

    @web_app.get("/list-scenes")
    async def list_scenes():
        scenes = []

        for timestamp_dir in sorted(output_dir.iterdir(), reverse=True):
            if timestamp_dir.is_dir():
                scenes_dir = timestamp_dir / "scenes"
                if scenes_dir.exists():
                    ply_files = list(scenes_dir.glob("*.ply"))
                    if ply_files:
                        for ply in ply_files:
                            scenes.append(
                                {
                                    "name": ply.name,
                                    "url": f"/scenes/{timestamp_dir.name}/{ply.name}",
                                    "timestamp": timestamp_dir.name,
                                    "created": ply.stat().st_mtime,
                                }
                            )

        return JSONResponse(sorted(scenes, key=lambda x: x["created"], reverse=True))

    return web_app


@app.local_entrypoint()
def main(
    prompt: str = "A serene mountain landscape with a clear lake",
    negative_prompt: str = "",
    labels_fg1: str = "stones",
    labels_fg2: str = "trees mountains",
    classes: str = "outdoor",
):
    hunyuan = HunyuanWorld()

    print("Generating...")
    hunyuan.generate.remote(
        prompt,
        negative_prompt,
        labels_fg1,
        labels_fg2,
        classes,
    )
    print("✓ Generation complete!")

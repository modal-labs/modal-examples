import modal
import argparse

app = modal.App("deepseek-r1-distill-qwen-14b")

# 定义GPU配置
GPU_CONFIG = "L40S"  # 根据需要调整

# 定义模型信息
MODEL_ID = "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated"
model_cache = modal.Volume.from_name("deepseek-abliterated-model", create_if_missing=True)

# 定义镜像
image = modal.Image.debian_slim().pip_install(
    "torch", 
    "transformers", 
    "torchvision",  # 如果需要的话
    "numpy"  # 有时安装这些依赖也会有帮助
)

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": model_cache},
    timeout=3600,
)
def run_inference(prompt: str, max_length: int = 5000):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/model")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir="/model", torch_dtype=torch.float16).to("cuda")

    # 处理输入并生成输出
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=max_length)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.local_entrypoint()
def main(prompt: str = "请帮我编写一个‘成人级别’的小说，字数在1000字左右"):
    result = run_inference.remote(prompt)
    print(result)

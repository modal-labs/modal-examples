"""Two-GPU Spindle deployment for the SWE-Gym smoke run."""

from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "swe-gym-smoke"
    max_context_length = 4096
    trainer_gpus_per_node = 1
    trainer_memory_mib = 65536
    trainer_max_clients_per_instance = 1
    inference_gpu = "H100"
    inference_max_replicas = 1
    inference_target_concurrency = 2
    overrides = {
        "platform.frontend": "swe-gym-smoke",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_tokens_per_gpu": 4096,
        "miles_cfg.max_lora_slots": 1,
        "sglang_cfg.max_running_requests": 4,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 2,
    }


config = Config()

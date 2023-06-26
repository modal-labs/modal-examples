from modal import Image, Stub, gpu, method, web_endpoint


# Spec for an image where model is cached locally
def download_model():
    from huggingface_hub import snapshot_download

    model_name = "mosaicml/mpt-30b-instruct"
    snapshot_download(model_name)


image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.39.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "peft @ git+https://github.com/huggingface/peft.git",
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "accelerate @ git+https://github.com/huggingface/accelerate.git",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "huggingface_hub==0.14.1",
        "einops==0.6.1",
    )
    .run_function(download_model)
)

stub = Stub(image=image, name="mpt")


@stub.cls(
    gpu=gpu.A100(),  # Use A100s
    timeout=60 * 10,  # 10 minute timeout on inputs
    container_idle_timeout=60 * 5,  # Keep runner alive for 5 minutes
)
class MPT30B:
    def __enter__(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_name = "mosaicml/mpt-30b-instruct"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.max_seq_len = (
            16384  # (input + output) tokens can now be up to 16384
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,  # Model is downloaded to cache dir
            device_map="auto",
            load_in_8bit=True,
            config=config,
        )
        model.tie_weights()
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        tokenizer.bos_token_id = 1

        self.model = torch.compile(model)
        self.tokenizer = tokenizer

    @method()
    def generate(self, prompt: str):
        from threading import Thread

        import torch
        from transformers import GenerationConfig, TextIteratorStreamer

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=512,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized.input_ids
            input_ids = input_ids.to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_special_tokens=True
            )
            generate_kwargs = dict(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                attention_mask=tokenized.attention_mask,
                output_scores=True,
                streamer=streamer,
            )

            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()
            for new_text in streamer:
                yield new_text

            thread.join()


prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{}\n\n### Response\n"


@stub.local_entrypoint()
def cli():
    question = "Can you describe the main differences between Python and JavaScript programming languages."
    model = MPT30B()
    for text in model.generate.call(prompt_template.format(question)):
        print(text, end="", flush=True)


@stub.function(timeout=60 * 10)
@web_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = MPT30B()
    return StreamingResponse(
        chain(
            ("Loading model (60GB). This usually takes around 100s ...\n\n"),
            model.generate.call(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )

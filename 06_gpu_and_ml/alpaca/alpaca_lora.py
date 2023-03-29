import sys
import modal

# Alpaca-LoRA is distributed as a public Github repository and the repository is not
# installable by `pip`, so instead we install the repository by cloning it into our Modal
# image.

repo_url = "https://github.com/tloen/alpaca-lora"
image = (
    modal.Image.debian_slim().apt_install("git")
    # Here we place the latest repository code into /root.
    # Because /root is almost empty, but not entirely empty, `git clone` won't work,
    # so this `init` then `checkout` workaround is used.
    .run_commands(
        "cd /root && git init .",
        f"cd /root && git remote add --fetch origin {repo_url}",
        "cd /root && git checkout main",
    )
    # The alpaca-lora repository's dependencies list is in the repository,
    # but it's currently missing a dependency and not specifying dependency versions,
    # which leads to issues: https://github.com/tloen/alpaca-lora/issues/200.
    # So we install a strictly versioned dependency list. This list excludes one or two
    # dependencies listed by `tloen/alpaca-lora` but that are irrelevant within Modal,
    # e.g. `black` code formatting library.
    .pip_install(
        "accelerate==0.18.0",
        "appdirs==1.4.4",
        "bitsandbytes==0.37.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "datasets==2.10.1",
        "fire==0.5.0",
        "gradio==3.23.0",
        "peft @ git+https://github.com/huggingface/peft.git@d8c3b6bca49e4aa6e0498b416ed9adc50cc1a5fd",
        "transformers @ git+https://github.com/huggingface/transformers.git@a92e0ad2e20ef4ce28410b5e05c5d63a5a304e65",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
    )
)
stub = modal.Stub(name="example-alpaca-lora", image=image)

# The Alpaca-LoRA model is integrated into model as a Python class with an __enter__
# method to take advantage of Modal's container lifecycle functionality.
#
# https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta
#
# On each container startup the model is loaded once and then subsequent model
# text generations are run 'warm' with the model already initialized in memory.


class AlpacaLoRAModel:
    def __enter__(self):
        """
        Container-lifeycle method for model setup. Code is taken from
        https://github.com/tloen/alpaca-lora/blob/main/generate.py and minor
        modifications are made to support usage in a Python class.
        """
        import torch
        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer

        load_8bit = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = "decapoda-research/llama-7b-hf"
        lora_weights = "tloen/alpaca-lora-7b"

        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model
        self.device = device

    @stub.function(gpu="A10G")
    def generate(self, instructions: list[str]):
        for instrctn in instructions:
            print(f"\033[96mInstruction: {instrctn}\033[0m")
            print("Response:", self.evaluate(instrctn))
            print()

    def evaluate(
        self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        import torch
        from generate import generate_prompt
        from transformers import GenerationConfig

        prompt = generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return output.split("### Response:")[1].strip()


# This Modal app local entrypoint just runs the example instructions shown in the
# repositorie's README: https://github.com/tloen/alpaca-lora#example-outputs.
#
# Run this app with `modal run alpaca_lora.py` and you can cross-reference against the
# repository to see how well the outputs match.


@stub.local_entrypoint
def main():
    instructions = [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]
    model = AlpacaLoRAModel()
    model.generate.call(instructions)

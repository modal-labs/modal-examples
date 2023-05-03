from modal import Image, Stub, method

base_model = "openlm-research/open_llama_7b_preview_200bt"
subfolder = "open_llama_7b_preview_200bt_transformers_weights"


def download_models():
    from transformers import LlamaForCausalLM, LlamaTokenizer

    LlamaForCausalLM.from_pretrained(base_model, subfolder=subfolder)
    LlamaTokenizer.from_pretrained(base_model, subfolder=subfolder)


image = (
    Image.debian_slim()
    .pip_install(
        "accelerate~=0.18.0",
        "transformers~=4.28.1",
        "torch~=2.0.0",
        "sentencepiece~=0.1.97",
    )
    .run_function(download_models)
)
stub = Stub(name="example-open-llama", image=image)


@stub.cls(gpu="A10G")
class OpenLlamaModel:
    def __enter__(self):
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained(
            base_model, subfolder=subfolder
        )
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            subfolder=subfolder,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer.bos_token_id = 1

        model.eval()
        self.model = torch.compile(model)
        self.device = "cuda"

    @method()
    def generate(
        self,
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):
        import torch
        from transformers import GenerationConfig

        inputs = self.tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=temperature > 0,
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
        print(f"\033[96m{input}\033[0m")
        print(output.split(input)[1].strip())


@stub.local_entrypoint()
def main():
    inputs = [
        "Building a website can be done in 10 simple steps:",
    ]
    model = OpenLlamaModel()
    for input in inputs:
        model.generate.call(input)

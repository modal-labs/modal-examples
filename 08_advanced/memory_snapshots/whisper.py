import modal

image = (
    modal.Image.debian_slim()
        .pip_install("transformers", "datasets", "hf_transfer", "torch", "librosa", "soundfile")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
stub = modal.Stub("whisper", image=image)


@stub.cls(
    gpu=modal.gpu.A10G(),
    timeout=60 * 10,
    container_idle_timeout=2,
    retries=0, 
    checkpointing_enabled=True,
)
class Whisper():

    @modal.build()
    def build(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from datasets import load_dataset

        load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        WhisperProcessor.from_pretrained("openai/whisper-tiny")
        WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    
    @modal.enter(checkpoint=True)
    def load(self):

        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from datasets import load_dataset

        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.model.config.forced_decoder_ids = None

        # load dummy dataset and read audio files
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        self.sample = ds[0]["audio"]
        

    @modal.enter(checkpoint=False)
    def setup(self):
        self.model.to("cuda")
        

    @modal.method()
    def run(self):
        input_features = self.processor(
            self.sample["array"],
            sampling_rate=self.sample["sampling_rate"], return_tensors="pt").input_features
        
        input_features.to("cuda")
        predicted_ids = self.model.generate(input_features.to("cuda"))
        self.processor.batch_decode(predicted_ids, skip_special_tokens=True)



if __name__ == "__main__":
    cls = modal.Cls.lookup("whisper", "Whisper_no_checkpoint")
    cls().run.remote()

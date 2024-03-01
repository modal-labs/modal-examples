# # Memory snapshots for Whisper Tiny
#
# This program runs inference using OpenAI's [`whisper-tiny`](https://huggingface.co/openai/whisper-tiny).
# We improve cold boot times by loading the memory into CPU and creating
# a memory snapshot, then moving weights to a GPU during startup.

import modal

image = (
    modal.Image.debian_slim()
        .pip_install("transformers", "datasets", "hf_transfer", "torch", "librosa", "soundfile")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
stub = modal.Stub("whisper", image=image)


## Create a modal class with memory snapshots enabled
# 
# `checkpointing_enabled=True` creates a Modal class with memory snapshots enabled.
# When this is set to `True` only imports are snapshotted. Use it in combination with
# `@enter(checkpoint=True)` to add load model weights in CPU memory and the snapshot.
# You can transfer weights to a GPU in `@enter(checkpoint=False)`. All methods decorated
# with `@enter(checkpoint=True)` are only executed during the snapshotting stage.

@stub.cls(
    gpu=modal.gpu.A10G(),
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

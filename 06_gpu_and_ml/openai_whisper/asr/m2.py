import base64
import os
import re


from modal import App, Image, gpu, enter, asgi_app
from utils.languages import LANGUAGES
import logging

tensorrt_image = Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)


tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.12.0.dev2024071600",
    "tiktoken",
    "datasets",
    "kaldialign",
    "openai-whisper",
    "librosa",
    "soundfile",
    "safetensors",
    "transformers",
    "janus",
    "kaldialign",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)


tensorrt_image = tensorrt_image.run_commands([
    "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
    "wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav",
    # take large-v3 model as an example
    "wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
])
CHECKPOINT_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/convert_checkpoint.py"


INFERENCE_PRECISION = "float16"
WEIGHT_ONLY_PRECISION = "int8"
MAX_BEAM_WIDTH = 4
MAX_BATCH_SIZE = 8
CHECKPOINT_DIR = f"/root/whisper/checkpoints/whisper_large_v3_weights_{WEIGHT_ONLY_PRECISION}"
OUTPUT_DIR= f"/root/whisper/output/whisper_large_v3_{WEIGHT_ONLY_PRECISION}"
N_GPUS = 1 
GPU_CONFIG = gpu.A100(count=N_GPUS)

tensorrt_image = tensorrt_image.run_commands([f"wget {CHECKPOINT_SCRIPT_URL} -O /root/convert_checkpoint.py",
    f"python /root/convert_checkpoint.py \
                --use_weight_only \
                --weight_only_precision {WEIGHT_ONLY_PRECISION} \
                --output_dir {CHECKPOINT_DIR}"
], gpu = GPU_CONFIG)


tensorrt_image = tensorrt_image.run_commands([
f"trtllm-build  --checkpoint_dir {CHECKPOINT_DIR}/encoder \
              --output_dir {OUTPUT_DIR}/encoder \
              --paged_kv_cache disable \
              --moe_plugin disable \
              --enable_xqa disable \
              --use_custom_all_reduce disable \
              --max_batch_size {MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin {INFERENCE_PRECISION} \
              --remove_input_padding disable \
              --max_input_len 1500",
], gpu=GPU_CONFIG)

image = tensorrt_image.run_commands([
f"trtllm-build  --checkpoint_dir {CHECKPOINT_DIR}/decoder \
              --output_dir {OUTPUT_DIR}/decoder \
              --paged_kv_cache disable \
              --moe_plugin disable \
              --enable_xqa disable \
              --use_custom_all_reduce disable \
              --max_beam_width {MAX_BEAM_WIDTH} \
              --max_batch_size {MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 1500 \
              --gemm_plugin {INFERENCE_PRECISION} \
              --bert_attention_plugin {INFERENCE_PRECISION} \
              --gpt_attention_plugin {INFERENCE_PRECISION} \
              --remove_input_padding disable"
], gpu=GPU_CONFIG)

image = image.pip_install("pydantic==1.10.11")


app = App("low-latency-transcription", image=image)

with image.imports():
    from typing import Annotated
    from pydantic import Json
    import tensorrt_llm
    import tensorrt_llm.logger as logger
    import logging
    from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                    trt_dtype_to_torch)
    from tensorrt_llm.runtime import ModelConfig, SamplingConfig
    from tensorrt_llm.runtime.session import Session, TensorInfo    
    import torch
    import json
    from collections import OrderedDict
    from pathlib import Path
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import PlainTextResponse
    from utils.whisper_utils import log_mel_spectrogram

    import tiktoken
    import base64
    import os
    import re


def get_tokenizer(name: str = "multilingual",
                  num_languages: int = 99,
                  tokenizer_dir: str = None):
    if tokenizer_dir is None:
        vocab_path = os.path.join(os.path.dirname(__file__),
                                  f"assets/{name}.tiktoken")
    else:
        vocab_path = os.path.join(tokenizer_dir, f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    import tiktoken
    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )

class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config_path = engine_dir / 'encoder' / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.n_mels = config['pretrained_config']['n_mels']
        self.dtype = config['pretrained_config']['dtype']
        self.num_languages = config['pretrained_config']['num_languages']

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, mel):

        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)

        inputs = OrderedDict()
        inputs['x'] = mel
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('x', str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                    input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                            outputs=outputs,
                            stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['output']
        return audio_features
    
class WhisperDecoding:
    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):
        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_config(self, engine_dir):
        config_path = engine_dir / 'decoder' / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config['pretrained_config'])
        decoder_config.update(config['build_config'])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                decoder_input_ids,
                encoder_outputs,
                eot_id,
                max_new_tokens=40,
                num_beams=1):
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device='cuda')
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                            dtype=torch.int32,
                                            device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [encoder_outputs.shape[0], 1,
            encoder_outputs.shape[1]]).int().cuda()

        # generation config
        sampling_config = SamplingConfig(end_id=eot_id,
                                        pad_id=eot_id,
                                        num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1])

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids
    
class WhisperTRTLLM(object):
    def __init__(self, engine_dir, debug_mode=False, assets_dir=None):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir,
                                       runtime_mapping,
                                       debug_mode=False)
        is_multilingual = (self.decoder.decoder_config['vocab_size'] >= 51865)
        if is_multilingual:
            tokenizer_name = "multilingual"
            assert (Path(assets_dir) / "multilingual.tiktoken").exists(
            ), "multilingual.tiktoken file is not existed in assets_dir"
        else:
            tokenizer_name == "gpt2"
            assert (Path(assets_dir) / "gpt2.tiktoken").exists(
            ), "gpt2.tiktoken file is not existed in assets_dir"
        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=self.encoder.num_languages,
                                       tokenizer_dir=assets_dir)
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]

    def process_batch(
            self,
            mel,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            num_beams=1):
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        encoder_output = self.encoder.get_audio_features(mel)
        output_ids = self.decoder.generate(decoder_input_ids,
                                           encoder_output,
                                           self.eot_id,
                                           max_new_tokens=96,
                                           num_beams=num_beams)
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts


def setup_logger():
    server_logger = logging.getLogger(__name__)
    server_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    server_logger.addHandler(console_handler)

    return server_logger

server_logger = setup_logger()

async def decode_wav_file(
        input_file_path,
        model,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype='float16',
        batch_size=1,
        num_beams=1,
        normalizer=None,
        mel_filters_dir=None):
    mel, total_duration = await log_mel_spectrogram(input_file_path,
                                              model.encoder.n_mels,
                                              device='cuda',
                                              return_duration=True,
                                              mel_filters_dir=mel_filters_dir)
    mel = mel.type(str_dtype_to_torch(dtype))
    mel = mel.unsqueeze(0)
    # repeat the mel spectrogram to match the batch size
    mel = mel.repeat(batch_size, 1, 1)
    predictions = model.process_batch(mel, text_prefix, num_beams)
    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r'<\|.*?\|>', '', prediction)
    if normalizer:
        prediction = normalizer(prediction)
    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, total_duration

@app.cls(keep_warm=1, allow_concurrent_inputs=1, concurrency_limit=1, gpu=GPU_CONFIG)
class Model:
    @enter()
    def enter(self):
        self.model = WhisperTRTLLM(OUTPUT_DIR, assets_dir="/assets")

    @asgi_app()
    def web(self):
        from fastapi import FastAPI
        webapp = FastAPI()

        @webapp.get("/", response_class=PlainTextResponse)
        @webapp.get("/health", response_class=PlainTextResponse)
        async def health():
            server_logger.info("health check")
            return "OK"

        @webapp.post("/")
        @webapp.post("/predict")
        async def predict(
            file: Annotated[UploadFile, File()],
        ):
            results, _ = await decode_wav_file(file, self.model)
            return results

        return webapp

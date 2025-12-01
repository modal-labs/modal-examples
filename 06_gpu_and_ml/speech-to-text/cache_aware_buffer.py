# ---
# lambda-test: false
# pytest: false
# ---

import copy

import torch
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.preprocessing.segment import get_samples
from omegaconf import OmegaConf


class CacheAwareStreamingAudioBuffer:
    """
    A buffer to be used for cache-aware streaming. It can load a single or multiple audio
    files/processed signals, split them in chunks and return one on one. It can be used to
    simulate streaming audio or audios.
    """

    def __init__(self, model, online_normalization=None, pad_and_drop_preencoded=False):
        """
        Args:
            model: An ASR model.
            online_normalization (bool): whether to perform online normalization per chunk or
            normalize the whole audio before chunking
            pad_and_drop_preencoded (bool): if true pad first audio chunk and always drop preencoded
        """
        self.model = model
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0
        self.pad_and_drop_preencoded = pad_and_drop_preencoded

        self.online_normalization = online_normalization
        if not isinstance(model.encoder, StreamingEncoder):
            raise ValueError(
                "The model's encoder is not inherited from StreamingEncoder, and likely not to support streaming!"
            )
        if model.encoder.streaming_cfg is None:
            model.encoder.setup_streaming_params()
        self.streaming_cfg = model.encoder.streaming_cfg

        self.input_features = model.encoder._feat_in

        self.preprocessor = self.extract_preprocessor()

        if hasattr(model.encoder, "pre_encode") and hasattr(
            model.encoder.pre_encode, "get_sampling_frames"
        ):
            self.sampling_frames = model.encoder.pre_encode.get_sampling_frames()
        else:
            self.sampling_frames = None

    def get_next_chunk(self):
        """
        Get the next audio chunk for streaming processing.

        This method can be called repeatedly after appending audio via append_audio()
        to process the audio in a streaming fashion.

        Returns:
            tuple: (audio_chunk, chunk_lengths) if there's data to process, None if buffer is exhausted
                - audio_chunk: tensor containing the audio chunk with pre-encode cache prepended
                - chunk_lengths: tensor containing the valid lengths for each stream in the batch
        """
        if self.buffer is None or self.buffer_idx >= self.buffer.size(-1):
            return None

        # Determine chunk size based on position (first chunk may be different)
        if self.buffer_idx == 0 and isinstance(self.streaming_cfg.chunk_size, list):
            if self.pad_and_drop_preencoded:
                chunk_size = self.streaming_cfg.chunk_size[1]
            else:
                chunk_size = self.streaming_cfg.chunk_size[0]
        else:
            chunk_size = (
                self.streaming_cfg.chunk_size[1]
                if isinstance(self.streaming_cfg.chunk_size, list)
                else self.streaming_cfg.chunk_size
            )

        # Determine shift size based on position (first chunk may be different)
        if self.buffer_idx == 0 and isinstance(self.streaming_cfg.shift_size, list):
            if self.pad_and_drop_preencoded:
                shift_size = self.streaming_cfg.shift_size[1]
            else:
                shift_size = self.streaming_cfg.shift_size[0]
        else:
            shift_size = (
                self.streaming_cfg.shift_size[1]
                if isinstance(self.streaming_cfg.shift_size, list)
                else self.streaming_cfg.shift_size
            )

        # Check if we have enough valid data available for a full chunk
        # We need at least chunk_size frames of valid data from buffer_idx onwards
        available_valid_frames = self.streams_length - self.buffer_idx
        if available_valid_frames.min() < chunk_size:
            # Not enough data accumulated yet, wait for more audio
            return None

        # Extract the current audio chunk
        audio_chunk = self.buffer[:, :, self.buffer_idx : self.buffer_idx + chunk_size]

        # Check if we have enough frames for downsampling (if applicable)
        if self.sampling_frames is not None:
            if self.buffer_idx == 0 and isinstance(self.sampling_frames, list):
                cur_sampling_frames = self.sampling_frames[0]
            else:
                cur_sampling_frames = (
                    self.sampling_frames[1]
                    if isinstance(self.sampling_frames, list)
                    else self.sampling_frames
                )
            if audio_chunk.size(-1) < cur_sampling_frames:
                return None

        # Add the pre-encode cache to the chunk
        zeros_pads = None
        if self.buffer_idx == 0 and isinstance(
            self.streaming_cfg.pre_encode_cache_size, list
        ):
            if self.pad_and_drop_preencoded:
                cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[
                    1
                ]
            else:
                cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[
                    0
                ]
            cache_pre_encode = torch.zeros(
                (audio_chunk.size(0), self.input_features, cache_pre_encode_num_frames),
                device=audio_chunk.device,
                dtype=audio_chunk.dtype,
            )
        else:
            if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
            else:
                pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size

            start_pre_encode_cache = self.buffer_idx - pre_encode_cache_size
            if start_pre_encode_cache < 0:
                start_pre_encode_cache = 0
            cache_pre_encode = self.buffer[
                :, :, start_pre_encode_cache : self.buffer_idx
            ]
            if cache_pre_encode.size(-1) < pre_encode_cache_size:
                zeros_pads = torch.zeros(
                    (
                        audio_chunk.size(0),
                        audio_chunk.size(-2),
                        pre_encode_cache_size - cache_pre_encode.size(-1),
                    ),
                    device=audio_chunk.device,
                    dtype=audio_chunk.dtype,
                )

        added_len = cache_pre_encode.size(-1)
        audio_chunk = torch.cat((cache_pre_encode, audio_chunk), dim=-1)

        # Apply online normalization if enabled
        if self.online_normalization:
            audio_chunk, x_mean, x_std = normalize_batch(
                x=audio_chunk,
                seq_len=torch.tensor([audio_chunk.size(-1)] * audio_chunk.size(0)),
                normalize_type=self.model_normalize_type,
            )

        # Add zero padding if needed
        if zeros_pads is not None:
            audio_chunk = torch.cat((zeros_pads, audio_chunk), dim=-1)
            added_len += zeros_pads.size(-1)

        # Calculate valid chunk lengths for each stream
        max_chunk_lengths = self.streams_length - self.buffer_idx
        max_chunk_lengths = max_chunk_lengths + added_len
        chunk_lengths = torch.clamp(max_chunk_lengths, min=0, max=audio_chunk.size(-1))

        # Update buffer position and step counter
        print(
            f"[get_next_chunk] BEFORE: buffer_idx={self.buffer_idx}, shift_size={shift_size}, streams_length={self.streams_length}, chunk_lengths={chunk_lengths}"
        )
        self.buffer_idx += shift_size
        self.step += 1
        print(
            f"[get_next_chunk] AFTER: buffer_idx={self.buffer_idx}, buffer.size(-1)={self.buffer.size(-1)}"
        )

        return audio_chunk, chunk_lengths

    def __iter__(self):
        """
        Iterator interface for batch processing.
        Yields chunks by repeatedly calling get_next_chunk().
        """
        while True:
            result = self.get_next_chunk()
            if result is None:
                return
            yield result

    def is_buffer_empty(self):
        if self.buffer_idx >= self.buffer.size(-1):
            return True
        else:
            return False

    def has_next_chunk(self):
        """
        Check if there are more chunks available to process.

        Returns:
            bool: True if get_next_chunk() will return data, False otherwise
        """
        if self.buffer is None or self.streams_length is None:
            return False

        # Determine the required chunk size for the next chunk
        if self.buffer_idx == 0 and isinstance(self.streaming_cfg.chunk_size, list):
            if self.pad_and_drop_preencoded:
                chunk_size = self.streaming_cfg.chunk_size[1]
            else:
                chunk_size = self.streaming_cfg.chunk_size[0]
        else:
            chunk_size = (
                self.streaming_cfg.chunk_size[1]
                if isinstance(self.streaming_cfg.chunk_size, list)
                else self.streaming_cfg.chunk_size
            )

        # Check if we have enough valid data available
        available_valid_frames = self.streams_length - self.buffer_idx
        return available_valid_frames.min() >= chunk_size

    def __len__(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0

    def reset_buffer_pointer(self):
        self.buffer_idx = 0
        self.step = 0

    def extract_preprocessor(self):
        cfg = copy.deepcopy(self.model._cfg)
        self.model_normalize_type = cfg.preprocessor.normalize
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        if self.online_normalization:
            cfg.preprocessor.normalize = "None"

        preprocessor = self.model.from_config_dict(cfg.preprocessor)
        return preprocessor.to(self.get_model_device())

    def append_audio_file(self, audio_filepath, stream_id=-1):
        audio = get_samples(audio_filepath)
        processed_signal, processed_signal_length, stream_id = self.append_audio(
            audio, stream_id
        )
        return processed_signal, processed_signal_length, stream_id

    def append_audio(self, audio, stream_id=-1):
        processed_signal, processed_signal_length = self.preprocess_audio(audio)
        print(
            f"[append_audio] Audio samples: {len(audio)}, Preprocessed to {processed_signal_length} frames"
        )
        processed_signal, processed_signal_length, stream_id = (
            self.append_processed_signal(processed_signal, stream_id)
        )
        print(
            f"[append_audio] After append: stream_id={stream_id}, streams_length={self.streams_length}, buffer_idx={self.buffer_idx}"
        )
        return processed_signal, processed_signal_length, stream_id

    def append_processed_signal(self, processed_signal, stream_id=-1):
        processed_signal_length = torch.tensor(
            processed_signal.size(-1), device=processed_signal.device
        )
        if stream_id >= 0 and (
            self.streams_length is not None and stream_id >= len(self.streams_length)
        ):
            raise ValueError("Not valid stream_id!")
        if self.buffer is None:
            if stream_id >= 0:
                raise ValueError(
                    "stream_id can not be specified when there is no stream."
                )
            self.buffer = processed_signal
            self.streams_length = torch.tensor(
                [processed_signal_length], device=processed_signal.device
            )
        else:
            if self.buffer.size(1) != processed_signal.size(1):
                raise ValueError(
                    "Buffer and the processed signal have different dimensions!"
                )
            if stream_id < 0:
                self.buffer = torch.nn.functional.pad(
                    self.buffer, pad=(0, 0, 0, 0, 0, 1)
                )
                self.streams_length = torch.cat(
                    (
                        self.streams_length,
                        torch.tensor([0], device=self.streams_length.device),
                    ),
                    dim=-1,
                )
                stream_id = len(self.streams_length) - 1
            needed_len = self.streams_length[stream_id] + processed_signal_length
            if needed_len > self.buffer.size(-1):
                self.buffer = torch.nn.functional.pad(
                    self.buffer, pad=(0, needed_len - self.buffer.size(-1))
                )

            self.buffer[
                stream_id,
                :,
                self.streams_length[stream_id] : self.streams_length[stream_id]
                + processed_signal_length,
            ] = processed_signal
            self.streams_length[stream_id] = self.streams_length[
                stream_id
            ] + processed_signal.size(-1)

        if self.online_normalization:
            processed_signal, x_mean, x_std = normalize_batch(
                x=processed_signal,
                seq_len=torch.tensor([processed_signal_length]),
                normalize_type=self.model_normalize_type,
            )
        return processed_signal, processed_signal_length, stream_id

    def get_model_device(self):
        return self.model.device

    def preprocess_audio(self, audio, device=None):
        if device is None:
            device = self.get_model_device()
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def get_all_audios(self):
        processed_signal = self.buffer
        if self.online_normalization:
            processed_signal, x_mean, x_std = normalize_batch(
                x=processed_signal,
                seq_len=torch.tensor(self.streams_length),
                normalize_type=self.model_normalize_type,
            )
        return processed_signal, self.streams_length

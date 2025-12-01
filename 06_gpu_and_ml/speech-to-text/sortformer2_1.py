# ---
# lambda-test: false
# pytest: false
# ---


# modified from:
# https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/agents/voice_agent/pipecat/services/nemo/streaming_diar.py

import math
from dataclasses import dataclass
from typing import Tuple

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState
from omegaconf import DictConfig
from torch import Tensor

LOG_MEL_ZERO = -16.635


class AudioBufferer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.zero_()

    def update(self, audio: np.ndarray) -> None:
        """
        Update the buffer with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)

        audio_size = audio.shape[0]
        if audio_size > self.buffer_size:
            raise ValueError(
                f"Frame size ({audio_size}) exceeds buffer size ({self.buffer_size})"
            )

        shift = audio_size
        self.sample_buffer[:-shift] = self.sample_buffer[shift:].clone()
        self.sample_buffer[-shift:] = audio.clone()

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.sample_buffer.clone()

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty
        Returns:
            bool: True if the buffer is empty, False otherwise
        """
        return self.sample_buffer.sum() == 0


class CacheFeatureBufferer:
    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg: DictConfig,
        device: torch.device,
        fill_value: float = LOG_MEL_ZERO,
    ):
        if buffer_size_in_secs < chunk_size_in_secs:
            raise ValueError(
                f"Buffer size ({buffer_size_in_secs}s) should be no less than chunk size ({chunk_size_in_secs}s)"
            )

        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.device = device

        if hasattr(preprocessor_cfg, "log") and preprocessor_cfg.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = (
                LOG_MEL_ZERO  # Log-Mel spectrogram value for zero signals
            )
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value

        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.sample_buffer = AudioBufferer(sample_rate, buffer_size_in_secs)

        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.float32,
            device=self.device,
        )

        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty
        Returns:
            bool: True if the buffer is empty, False otherwise
        """
        return self.sample_buffer.is_buffer_empty()

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)

    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        """
        Add an extracted feature to `feature_buffer`
        """
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[
            :, self.feature_chunk_len :
        ].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the audio signal using the preprocessor
        Args:
            audio_signal (torch.Tensor): audio signal
        Returns:
            torch.Tensor: preprocessed features
        """
        audio_signal = audio_signal.unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len,
        )
        features = features.squeeze()
        return features

    def update(self, audio: np.ndarray) -> None:
        """
        Update the sample anf feature buffers with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """

        # Update the sample buffer with the new frame
        self.sample_buffer.update(audio)

        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            # If the buffer size is equal to the chunk size, just take the whole buffer
            samples = self.sample_buffer.sample_buffer.clone()
        else:
            # Add look_back to have context for the first feature
            samples = self.sample_buffer.sample_buffer[
                -(self.n_chunk_look_back + self.chunk_size) :
            ]

        # Get the mel spectrogram
        features = self.preprocess(samples)

        # If the features are longer than supposed to be, drop the last frames
        # Drop the last diff frames because they might be incomplete
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]

        # Update the feature buffer with the new features
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current sample buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.sample_buffer.get_buffer()

    def get_feature_buffer(self) -> torch.Tensor:
        """
        Get the current feature buffer
        Returns:
            torch.Tensor: current state of the feature buffer
        """
        return self.feature_buffer.clone()


@dataclass
class DiarizationConfig:
    """Diarization configuration parameters for inference."""

    model_path: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    device: str = "cuda"

    log: bool = False  # If True, log will be printed
    max_num_speakers: int = 4
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 144
    fifo_len: int = 188
    chunk_len: int = 6
    chunk_left_context: int = 1
    chunk_right_context: int = 7


class NeMoStreamingDiarizer:
    def __init__(
        self,
        cfg: DiarizationConfig,
        model: str,
        frame_len_in_secs: float = 0.08,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
        use_amp: bool = False,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.cfg = cfg
        self.cfg.model_path = model
        self.diarizer = self.build_diarizer()
        self.device = cfg.device
        self.use_amp = use_amp
        self.compute_dtype = compute_dtype
        self.frame_len_in_secs = frame_len_in_secs
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.chunk_size = self.cfg.chunk_len
        self.buffer_size_in_secs = (
            self.cfg.chunk_len * self.frame_len_in_secs
            + (self.left_offset + self.right_offset) * 0.01
        )
        self.max_num_speakers = self.cfg.max_num_speakers

        self.feature_bufferer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.cfg.chunk_len * self.frame_len_in_secs,
            preprocessor_cfg=self.diarizer.cfg.preprocessor,
            device=self.device,
        )
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros(
            (1, 0, self.max_num_speakers), device=self.diarizer.device
        )

        print(
            f"NeMoStreamingDiarService initialized with model `{model}` on device `{self.device}`"
        )

    def build_diarizer(self):
        if self.cfg.model_path.endswith(".nemo"):
            diar_model = SortformerEncLabelModel.restore_from(
                self.cfg.model_path, map_location=self.cfg.device
            )
        else:
            diar_model = SortformerEncLabelModel.from_pretrained(
                self.cfg.model_path, map_location=self.cfg.device
            )

        # Steaming mode setup
        diar_model.sortformer_modules.chunk_len = self.cfg.chunk_len
        diar_model.sortformer_modules.spkcache_len = self.cfg.spkcache_len
        diar_model.sortformer_modules.chunk_left_context = self.cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = self.cfg.chunk_right_context
        diar_model.sortformer_modules.fifo_len = self.cfg.fifo_len
        diar_model.sortformer_modules.log = self.cfg.log
        diar_model.sortformer_modules.spkcache_refresh_rate = (
            self.cfg.spkcache_refresh_rate
        )
        diar_model.eval()

        return diar_model

    def print_diar_result(self, diar_result: np.ndarray):
        full_result = []
        for t in range(diar_result.shape[0]):
            spk_probs = ""
            for s in range(diar_result.shape[1]):
                spk_probs += f"{diar_result[t, s]:.2f} "
            full_result.append(f"Time {t}: {spk_probs}")
        print(full_result)
        return full_result

    def diarize(self, audio: bytes, stream_id: str = "default"):
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        self.feature_bufferer.update(audio_array)

        features = self.feature_bufferer.get_feature_buffer()
        feature_buffers = features.unsqueeze(0)  # add batch dimension
        feature_buffers = feature_buffers.transpose(
            1, 2
        )  # [batch, feature, time] -> [batch, time, feature]
        feature_buffer_lens = torch.tensor(
            [feature_buffers.shape[1]], device=self.device
        )
        self.streaming_state, chunk_preds = self.stream_step(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            streaming_state=self.streaming_state,
            total_preds=self.total_preds,
            left_offset=self.left_offset,
            right_offset=self.right_offset,
        )
        self.total_preds = chunk_preds
        diar_result = chunk_preds[:, -self.chunk_size :, :].clone().cpu().numpy()
        return diar_result[0]  # tensor of shape [6, 4]

    def reset_state(self, stream_id: str = "default"):
        self.feature_bufferer.reset()
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros(
            (1, 0, self.max_num_speakers), device=self.diarizer.device
        )

    def init_streaming_state(self, batch_size: int = 1) -> StreamingSortformerState:
        """
        Initialize the streaming state for the diarization model.

        Args:
            batch_size: The batch size to use.

        Returns:
            SortformerStreamingState: The initialized streaming state.
        """
        # Use the model's init_streaming_state method but convert to SortformerStreamingState format
        nemo_state = self.diarizer.sortformer_modules.init_streaming_state(
            batch_size=batch_size,
            async_streaming=self.diarizer.async_streaming,
            device=self.device,
        )

        return nemo_state

    def stream_step(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        streaming_state: StreamingSortformerState,
        total_preds: Tensor,
        left_offset: int = 0,
        right_offset: int = 0,
    ) -> Tuple[StreamingSortformerState, Tensor]:
        """
        Execute a single streaming step for diarization.

        Args:
            processed_signal: The processed audio signal.
            processed_signal_length: The length of the processed signal.
            streaming_state: The current streaming state.
            total_preds: The total predictions so far.
            left_offset: The left offset for the current chunk.
            right_offset: The right offset for the current chunk.

        Returns:
            Tuple[SortformerStreamingState, Tensor]: The updated streaming state and predictions.
        """
        # Move tensors to correct device
        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)

        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)

        if total_preds is not None and total_preds.device != self.device:
            total_preds = total_preds.to(self.device)

        with (
            torch.amp.autocast(
                device_type=self.device, dtype=self.compute_dtype, enabled=self.use_amp
            ),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            try:
                # Call the model's forward_streaming_step method
                streaming_state, diar_pred_out_stream = (
                    self.diarizer.forward_streaming_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        streaming_state=streaming_state,
                        total_preds=total_preds,
                        left_offset=left_offset,
                        right_offset=right_offset,
                    )
                )
            except Exception as e:
                print(f"Error in diarizer streaming step: {e}")
                # print the stack trace
                import traceback

                traceback.print_exc()
                # Return the existing state and preds if there's an error
                return streaming_state, total_preds

        return streaming_state, diar_pred_out_stream

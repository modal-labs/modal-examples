# ---
# lambda-test: false
# pytest: false
# ---

import math
import os
import sys
import tempfile
import wave
from typing import Callable, Sequence
from urllib.request import urlopen

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from omegaconf import DictConfig


def preprocess_audio(audio: bytes | str, target_sample_rate: int = 16000) -> bytes:
    import array
    import io
    import wave

    if isinstance(audio, str):
        audio = get_bytes_from_wav(audio)

    with wave.open(io.BytesIO(audio), "rb") as wav_in:
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    # Convert frames to array based on sample width
    if sample_width == 1:
        audio_data = array.array("B", frames)  # unsigned char
    elif sample_width == 2:
        audio_data = array.array("h", frames)  # signed short
    elif sample_width == 3:
        audio_data = array.array("b", frames)  # signed byte
    elif sample_width == 4:
        audio_data = array.array("i", frames)  # signed int
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Downmix to mono if needed
    if n_channels > 1:
        mono_data = array.array(audio_data.typecode)
        for i in range(0, len(audio_data), n_channels):
            chunk = audio_data[i : i + n_channels]
            mono_data.append(sum(chunk) // n_channels)
        audio_data = mono_data

    # Resample to 16kHz if needed
    if frame_rate != target_sample_rate:
        ratio = target_sample_rate / frame_rate
        new_length = int(len(audio_data) * ratio)
        resampled_data = array.array(audio_data.typecode)

        for i in range(new_length):
            # Linear interpolation
            pos = i / ratio
            pos_int = int(pos)
            pos_frac = pos - pos_int

            if pos_int >= len(audio_data) - 1:
                sample = audio_data[-1]
            else:
                sample1 = audio_data[pos_int]
                sample2 = audio_data[pos_int + 1]
                sample = int(sample1 + (sample2 - sample1) * pos_frac)

            resampled_data.append(sample)

        audio_data = resampled_data

    return audio_data.tobytes()


def get_bytes_from_wav(location: str) -> bytes:
    if location.startswith("http"):
        bytes = urlopen(location).read()
    else:
        bytes = open(location, "rb").read()

    return bytes


def identity(data):
    return data


def batch_seq(
    data: Sequence, chunk_size: int, transform: Callable = None
) -> list[bytes]:
    if transform is None:
        transform = identity
    return [
        transform(data[i : i + chunk_size]) for i in range(0, len(data), chunk_size)
    ]


SHUTDOWN_SIGNAL = (
    b"END_OF_STREAM_8f13d09"  # byte sequence indicating a stream is finished
)


def int2float(audio_data):
    import numpy as np

    abs_max = np.abs(audio_data).max()
    audio_data = audio_data.astype("float32")
    if abs_max > 0:
        audio_data *= 1 / 32768
    audio_data = audio_data.squeeze()  # depends on the use case
    return audio_data


def bytes_to_torch(data, device="cuda"):
    import numpy as np
    import torch

    data = np.frombuffer(data, dtype=np.int16)
    data = torch.from_numpy(int2float(data)).to(device)
    return data


class NoStdStreams(object):
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush(), self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


def write_wav_file(args):
    idx, data = args
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)  # 16-bit
        wav_out.setframerate(16000)
        wav_out.writeframes(data)
    temp_file.close()
    return temp_file.name


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

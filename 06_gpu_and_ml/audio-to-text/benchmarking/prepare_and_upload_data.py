# ---
# cmd: ["modal", "run", "06_gpu_and_ml/audio-to-text/benchmarking/prepare_and_upload_data.py::process_wav_files"]
# ---

import modal
from common import DATASET_VOLUME_NAME, app, dataset_volume

data_prep_image = (
    modal.Image.debian_slim()
    .pip_install("numpy==2.2.6")
    .add_local_python_source("common")
)


@app.function(
    volumes={"/data": dataset_volume},
    image=data_prep_image,
)
def convert_to_mono_16khz(input_path: str, output_dir: str):
    """Converts an input WAV file to 16khz mono and stores output in `output_path` WAV file."""
    import wave
    from pathlib import Path

    import numpy as np

    # Open the input WAV file
    filename = Path(input_path).name
    input_path = str(Path("/data") / input_path)
    with wave.open(input_path, "rb") as wav_in:
        # Get the input parameters
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()

        # Read all frames
        frames = wav_in.readframes(n_frames)

    # Convert frames to numpy array
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Reshape the array based on number of channels
    audio_data = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels)
        # Convert to mono by averaging all channels
        audio_data = audio_data.mean(axis=1).astype(dtype)

    # Resample to 16 kHz if needed
    if frame_rate != 16000:
        # Calculate resampling ratio
        ratio = 16000 / frame_rate
        new_length = int(len(audio_data) * ratio)

        # Resample using linear interpolation
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(
            dtype
        )

    # Create a new WAV file
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_path = str(Path(output_dir) / filename)
    with wave.open(output_path, "wb") as wav_out:
        wav_out.setnchannels(1)  # Mono
        wav_out.setsampwidth(sample_width)
        wav_out.setframerate(16000)  # 16 kHz
        wav_out.writeframes(audio_data.tobytes())


@app.function(
    volumes={"/data": dataset_volume},
    image=data_prep_image,
)
def process_wav_files():
    import itertools

    print("Processing files to 16kHz mono...")
    output_dir = "/data/processed"
    files = [
        f.path for f in dataset_volume.listdir("/raw/wavs") if f.path.endswith(".wav")
    ]
    inputs = list(zip(files, itertools.cycle([output_dir])))

    # Process all files in parallel
    list(convert_to_mono_16khz.starmap(inputs))
    print(
        f"✨ All done! Data processed and available in the {output_dir} in the Modal Volume {DATASET_VOLUME_NAME}"
    )

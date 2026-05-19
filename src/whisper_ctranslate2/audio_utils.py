"""
Audio format utilities for whisper-ctranslate2.

Provides automatic format detection and conversion for audio files that are not
directly supported by faster-whisper's built-in decoder.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Supported audio formats
# - native: supported by faster-whisper (via av/soundfile)
# - ffmpeg: requires conversion via ffmpeg

SUPPORTED_NATIVE_FORMATS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mwv",
}

SUPPORTED_FFMPEG_FORMATS = {
    ".wma",
    ".aac",
    ".ape",
    ".alac",
    ".opus",
    ".m4b",
    ".mpc",
    ".ofr",
    ".ofs",
    ".tta",
    ".wv",
    ".3gp",
    ".amr",
    ".awb",
    ".ac3",
    ".dts",
    ".eac3",
    ".mka",
    ".mlp",
    ".flac",
    ".tak",
    ".thd",
}

# All formats that can be processed
ALL_SUPPORTED_FORMATS = SUPPORTED_NATIVE_FORMATS | SUPPORTED_FFMPEG_FORMATS

# Standard audio parameters for Whisper
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=False,
        )
        return True
    except FileNotFoundError:
        return False


def get_audio_format(file_path: str) -> str:
    """
    Get the audio format category for a file.

    Returns:
        'native' if the format is natively supported,
        'ffmpeg' if conversion via ffmpeg is needed,
        'unsupported' if the format cannot be processed.
    """
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_NATIVE_FORMATS:
        return "native"
    elif ext in SUPPORTED_FFMPEG_FORMATS:
        return "ffmpeg"
    else:
        return "unsupported"


def convert_audio_with_ffmpeg(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> str:
    """
    Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to the input audio file
        output_path: Path to the output WAV file. If None, a temporary file is created.
        sample_rate: Target sample rate (default: 16000 Hz)
        channels: Target number of channels (default: 1 = mono)

    Returns:
        Path to the converted WAV file

    Raises:
        RuntimeError: If ffmpeg conversion fails
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_path,
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-y",  # Overwrite output file if exists
                "-loglevel",
                "error",
                output_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg conversion failed for '{input_path}': {e.stderr}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg not found. Please install ffmpeg to process '{input_path}'"
        )

    return output_path


def ensure_audio_file(
    file_path: Union[str, np.ndarray],
) -> tuple[Union[str, np.ndarray], Optional[str]]:
    """
    Ensure the audio file is in a format that faster-whisper can process.

    For native formats, returns the original path.
    For ffmpeg-supported formats, converts to a temporary WAV file.
    For numpy arrays, returns as-is.

    Args:
        file_path: Path to the audio file or numpy array

    Returns:
        Tuple of (audio_path, temp_file_to_cleanup)
        temp_file_to_cleanup will be None if no conversion was needed
    """
    if isinstance(file_path, np.ndarray):
        # Already a numpy array, no conversion needed
        return file_path, None

    audio_format = get_audio_format(file_path)

    if audio_format == "native":
        return file_path, None
    elif audio_format == "ffmpeg":
        if not is_ffmpeg_available():
            raise RuntimeError(
                f"Unsupported audio format '{Path(file_path).suffix}'. "
                f"Please install ffmpeg to process this file type."
            )
        temp_wav = convert_audio_with_ffmpeg(file_path)
        return temp_wav, temp_wav  # Return path and cleanup path
    else:
        raise ValueError(
            f"Unsupported audio format '{Path(file_path).suffix}'. "
            f"Supported formats: {', '.join(sorted(ALL_SUPPORTED_FORMATS))}"
        )


def get_supported_formats_list() -> list[str]:
    """Get a sorted list of all supported audio format extensions."""
    return sorted(ALL_SUPPORTED_FORMATS)


def is_supported_format(file_path: str) -> bool:
    """Check if a file path has a supported audio format."""
    ext = Path(file_path).suffix.lower()
    return ext in ALL_SUPPORTED_FORMATS

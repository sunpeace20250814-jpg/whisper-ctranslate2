# Based on code from https://github.com/Nikorasu/LiveWhisper/blob/main/livewhisper.py

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Union

import numpy as np

from .transcribe import Transcribe, TranscriptionOptions

# Configuration constants with sensible defaults
DEFAULT_BLOCK_SIZE_MS = 30  # Block size in milliseconds
DEFAULT_VOCAL_FREQ_RANGE = (50, 1000)  # Frequency range to detect sounds that could be speech
DEFAULT_END_BLOCKS_MULTIPLIER = 2  # Multiplier for end blocks wait time
DEFAULT_FLUSH_BLOCKS_MULTIPLIER = 10  # Multiplier for flush blocks wait time

sounddevice_available = False
sounddevice_exception: Optional[Exception] = None
sd = None

try:
    import sounddevice as _sd
    sd = _sd
    sounddevice_available = True
except Exception as e:
    sounddevice_available = False
    sounddevice_exception = e


@dataclass
class LiveConfig:
    block_size_ms: int = DEFAULT_BLOCK_SIZE_MS
    vocal_freq_range: tuple = field(default_factory=lambda: DEFAULT_VOCAL_FREQ_RANGE)
    end_blocks_multiplier: int = DEFAULT_END_BLOCKS_MULTIPLIER
    flush_blocks_multiplier: int = DEFAULT_FLUSH_BLOCKS_MULTIPLIER

    def __post_init__(self):
        if not isinstance(self.vocal_freq_range, tuple) or len(self.vocal_freq_range) != 2:
            raise ValueError("vocal_freq_range must be a tuple of (min_freq, max_freq)")
        if self.vocal_freq_range[0] >= self.vocal_freq_range[1]:
            raise ValueError("vocal_freq_range[0] must be less than vocal_freq_range[1]")
        if self.block_size_ms <= 0:
            raise ValueError("block_size_ms must be positive")
        if self.end_blocks_multiplier <= 0 or self.flush_blocks_multiplier <= 0:
            raise ValueError("multipliers must be positive")

    @property
    def end_blocks(self) -> int:
        return 33 * self.end_blocks_multiplier

    @property
    def flush_blocks(self) -> int:
        return 33 * self.flush_blocks_multiplier


class Live:
    def __init__(
        self,
        model_path: str,
        cache_directory: str,
        local_files_only: bool,
        task: str,
        language: str,
        threads: int,
        device: str,
        device_index: Union[int, List[int]],
        compute_type: str,
        verbose: bool,
        threshold: float,
        input_device: int,
        input_device_sample_rate: int,
        options: TranscriptionOptions,
        config: Optional[LiveConfig] = None,
    ):
        self.model_path = model_path
        self.cache_directory = cache_directory
        self.local_files_only = local_files_only
        self.task = task
        self.language = language
        self.threads = threads
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.verbose = verbose
        self.threshold = threshold
        self.input_device = input_device
        self.input_device_sample_rate = input_device_sample_rate
        self.options = options
        self.config = config or LiveConfig()

        # Thread-safe state management
        self._lock = threading.Lock()
        self._running = True
        self._waiting = 0
        self._prevblock = self._buffer = np.zeros((0, 1))
        self._speaking = False
        self._blocks_speaking = 0
        self._buffers_to_process: Deque[np.ndarray] = deque()
        self.transcribe = None

    @staticmethod
    def is_available():
        return sounddevice_available

    @staticmethod
    def force_not_available_exception():
        if sounddevice_exception is not None:
            raise sounddevice_exception
        raise RuntimeError(
            "sounddevice library is not available. "
            "Install it with: pip install sounddevice"
        )

    def _is_there_voice(self, indata, frames):
        freq = (
            np.argmax(np.abs(np.fft.rfft(indata[:, 0])))
            * self.input_device_sample_rate
            / frames
        )
        volume = np.sqrt(np.mean(indata**2))
        min_freq, max_freq = self.config.vocal_freq_range

        return volume > self.threshold and min_freq <= freq <= max_freq

    def _save_to_process(self):
        with self._lock:
            self._buffers_to_process.append(self._buffer.copy())
            self._buffer = np.zeros((0, 1))
            self._speaking = False

    def callback(self, indata, frames, _time, status):
        if not any(indata):
            return

        voice = self._is_there_voice(indata, frames)

        if not voice and not self._speaking:
            return

        if voice:
            if self.verbose:
                print(".", end="", flush=True)
            with self._lock:
                if self._waiting < 1:
                    self._buffer = self._prevblock.copy()

                self._buffer = np.concatenate((self._buffer, indata))
                self._waiting = self.config.end_blocks

                if not self._speaking:
                    self._blocks_speaking = self.config.flush_blocks

                self._speaking = True
        else:
            with self._lock:
                self._waiting -= 1
                if self._waiting < 1:
                    self._save_to_process()
                    return
                else:
                    self._buffer = np.concatenate((self._buffer, indata))

            with self._lock:
                self._blocks_speaking -= 1
                if self._blocks_speaking < 1:
                    self._save_to_process()

    def process(self):
        with self._lock:
            if len(self._buffers_to_process) > 0:
                _buffer = self._buffers_to_process.popleft()
            else:
                return

        if self.verbose:
            print("\n\033[90mTranscribing..\033[0m")

        if not self.transcribe:
            self.transcribe = Transcribe(
                self.model_path,
                self.device,
                self.device_index,
                self.compute_type,
                self.threads,
                self.cache_directory,
                self.local_files_only,
                False,
            )

        result = self.transcribe.inference(
            audio=_buffer.flatten().astype("float32"),
            task=self.task,
            language=self.language,
            verbose=self.verbose,
            live=True,
            options=self.options,
        )
        print(f"\033[1A\033[2K\033[0G{result['text']}")
        if not self.verbose:
            print("")

    def listen(self):
        show_device = (
            self.input_device if self.input_device is not None else sd.default.device[0]
        )
        print(
            f"\033[32mLive stream device: \033[37m{sd.query_devices(device=show_device)['name']}\033[0m"
        )
        print("\033[32mListening.. \033[37m(Ctrl+C to Quit)\033[0m")

        self._prevblock = np.zeros((0, 1))

        with sd.InputStream(
            channels=1,
            callback=self.callback,
            blocksize=int(self.input_device_sample_rate * self.config.block_size_ms / 1000),
            samplerate=self.input_device_sample_rate,
            device=self.input_device,
        ):
            while self._running:
                self.process()

    def inference(self):
        try:
            self.listen()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self._running = False
            print("\n\033[93mQuitting..\033[0m")

    def stop(self):
        with self._lock:
            self._running = False

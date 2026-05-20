import numpy as np
from collections import OrderedDict
from typing import Optional

from faster_whisper.audio import decode_audio

class DiarizationError(Exception):
    '''Custom exception for diarization errors.'''
    pass

class Diarization:
    def __init__(
        self,
        token: Optional[str] = None,
        device: str = "cpu",
        num_speakers: int = 2,
    ):
        self.device = device
        self.token = token
        self.num_speakers = num_speakers
        self.model = None
        self._torch_available = False
        self._pyannote_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import torch
            self._torch_available = True
        except ImportError as e:
            raise DiarizationError(
                f"Unable to import torch library: {e}. Make sure PyTorch is installed."
            ) from e

        try:
            from pyannote.audio import Pipeline
            self._pyannote_available = True
        except ImportError as e:
            raise DiarizationError(
                f"Unable to import pyannote.audio library: {e}. Make sure pyannote.audio is installed."
            ) from e

    def set_threads(self, threads: int) -> None:
        if self._torch_available:
            import torch
            torch.set_num_threads(threads)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            import torch
            torch.cuda.empty_cache()
        self.model = None

    def _load_model(self):
        if not self._pyannote_available:
            raise DiarizationError("pyannote.audio is not available")
        if not self.token:
            raise DiarizationError(
                "HuggingFace token is required for diarization. "
                "Get one at https://huggingface.co/settings/tokens"
            )

        from pyannote.audio import Pipeline
        import torch

        model_name = "pyannote/speaker-diarization-community-1"
        device = torch.device(self.device)
        model_handle = Pipeline.from_pretrained(model_name, token=self.token)
        if model_handle is None:
            raise DiarizationError(
                f"The HuggingFace token is not valid or you did not accept the EULAs for the necessary models. "
                f"See https://github.com/Softcatala/whisper-ctranslate2#diarization-speaker-identification"
            )

        self.model = model_handle.to(device)

    def run_model(self, audio: str):
        if not self._torch_available:
            raise DiarizationError("torch is required but not available")

        import torch

        if self.model is None:
            self._load_model()
        audio = decode_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": 16000,
        }
        segments = self.model(audio_data, num_speakers=self.num_speakers)
        return segments

    def assign_speakers_to_segments(self, segments, transcript_result, speaker_name: Optional[str] = None):
        diarize_data = []
        for turn, speaker in segments.speaker_diarization:
            diarize_data.append((turn, None, speaker))

        return self._do_assign_speakers_to_segments(
            diarize_data, transcript_result, speaker_name
        )

    def _do_assign_speakers_to_segments(
        self, diarize_data, transcript_result, speaker_name: Optional[str]
    ):
        diarize_df = np.array(
            diarize_data,
            dtype=[("segment", object), ("label", object), ("speaker", object)],
        )

        diarize_df = np.core.records.fromarrays(
            [
                diarize_df["segment"],
                diarize_df["label"],
                diarize_df["speaker"],
                np.array([seg.start for seg in diarize_df["segment"]]),
                np.array([seg.end for seg in diarize_df["segment"]]),
                np.zeros(len(diarize_df)),
            ],
            names="segment, label, speaker, start, end, intersection",
        )

        for seg in transcript_result["segments"]:
            intersection = np.minimum(diarize_df["end"], seg["end"]) - np.maximum(
                diarize_df["start"], seg["start"]
            )
            diarize_df["intersection"] = intersection
            dia_segment = diarize_df[diarize_df["intersection"] > 0]
            if len(dia_segment) > 0:
                speakers = {}
                for item in dia_segment:
                    speaker = item["speaker"]
                    old_i = speakers.get(speaker, 0)
                    speakers[speaker] = old_i + item["intersection"]

                sorted_dict = OrderedDict(
                    sorted(speakers.items(), key=lambda x: x[1], reverse=True)
                )
                first_item = next(iter(sorted_dict.items()))
                if first_item:
                    speaker = first_item[0]
                    if speaker_name:
                        speaker = speaker.replace("SPEAKER", speaker_name)
                    seg["speaker"] = speaker

        return transcript_result

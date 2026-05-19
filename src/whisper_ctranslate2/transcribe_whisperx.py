import os
import sys
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import tqdm

try:
    import whisperx
except ImportError:
    print("WhisperX not installed. Install with: pip install whisperx")
    sys.exit(1)

from .audio_utils import ensure_audio_file
from .languages import LANGUAGES


class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperature: List[float]
    initial_prompt: Optional[str]
    prefix: Optional[str]
    hotwords: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    word_timestamps: bool
    print_colors: bool
    prepend_punctuations: str
    append_punctuations: str
    hallucination_silence_threshold: Optional[float]
    vad_filter: bool
    vad_threshold: Optional[float]
    vad_min_speech_duration_ms: Optional[int]
    vad_max_speech_duration_s: Optional[int]
    vad_min_silence_duration_ms: Optional[int]
    multilingual: bool
    max_new_tokens: Optional[int]
    return_scores: bool


class TranscribeWhisperX:
    """WhisperX backend for enhanced transcription with word-level timestamps."""

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        batch_size: int = None,
        language: str = None,
    ):
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size or 8

        # Load WhisperX model
        # compute_type mapping: float16->fp16, int8_float16->int8
        if compute_type == "int8_float16":
            torch_dtype = "int8"
        elif compute_type == "float16":
            torch_dtype = "float16"
        else:
            torch_dtype = "float32"

        self.model = whisperx.load_model(
            model_name,
            device=device,
            download_root=None,
            language=language,
        )
        self.alignment_model = None
        self.metadata = None

    def _load_alignment_model(self, language: str):
        """Lazy load alignment model for word timestamps."""
        if self.alignment_model is None:
            self.alignment_model, self.metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
        return self.alignment_model, self.metadata

    def inference(
        self,
        audio: Union[str, np.ndarray],
        task: str,
        language: str,
        verbose: bool,
        options: TranscriptionOptions,
    ):
        """Run WhisperX transcription."""
        temp_audio_path = None
        try:
            # Convert audio if needed (WhisperX handles many formats natively)
            if isinstance(audio, str):
                audio, temp_audio_path = ensure_audio_file(audio)

            # Transcription
            result = self.model.transcribe(
                audio,
                language=language,
                batch_size=self.batch_size,
                task=task,
                verbose=verbose,
            )

            # Word-level timestamps via alignment
            if options.word_timestamps or options.print_colors:
                align_model, metadata = self._load_alignment_model(language)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    device=self.device,
                    return_char_alignments=False,
                )

            # Build output
            language_name = LANGUAGES.get(result.get("language", "en"), {}).get("name", "English")
            if verbose:
                print(
                    f"Detected language '{language_name}' with probability {result.get('language_probability', 0):.2f}"
                )

            list_segments = []
            all_text = ""
            total_duration = result.get("duration", 0)

            with tqdm.tqdm(
                total=total_duration, unit="seconds", disable=not verbose
            ) as pbar:
                for segment in result.get("segments", []):
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "")

                    if isinstance(text, dict):
                        text = text.get("text", "")

                    all_text += text

                    if verbose:
                        from .writers import format_timestamp
                        line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                        print(line)

                    # Build word-level data
                    words_data = []
                    if "words" in segment:
                        for word in segment["words"]:
                            words_data.append({
                                "word": word.get("word", ""),
                                "start": word.get("start", 0),
                                "end": word.get("end", 0),
                                "probability": word.get("probability", 1.0),
                            })

                    list_segments.append({
                        "start": start,
                        "end": end,
                        "text": text,
                        "words": words_data if words_data else None,
                    })

                    pbar.update(end - start)

            return {
                "text": all_text,
                "segments": list_segments,
                "language": result.get("language", "en"),
            }

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

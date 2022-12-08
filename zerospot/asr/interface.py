from pathlib import Path
from typing import List, Union

import torch
from huggingsound import SpeechRecognitionModel

from zerospot.asr.constants import (END_TIMESTAMPS_FIELD,
                                    START_TIMESTAMPS_FIELD,
                                    TRANSCRIPTION_FIELD)
from zerospot.asr.data_types import ASRTranscription


class ASRModelInterface:
    """A class representing an interface to an automatic speech recognition (ASR) model.

    Attributes:
        _asr_model_name: The name of the ASR model to use.
        _timestamp_to_sec_ratio: The ratio used to convert timestamps to seconds.
        _model: The ASR model.
    """
    _asr_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    _timestamp_to_sec_ratio = 0.001

    def __init__(self, device: Union[torch.device, str]) -> None:
        """Create a new `ASRModelInterface`.

        Args:
            device: The device to use for running the ASR model.
        """
        self._model = SpeechRecognitionModel(self._asr_model_name, device=device)

    def get_transcription(self, audio_paths: List[Path]) -> List[ASRTranscription]:
        """Transcribe the audio at the specified paths.

        Args:
            audio_paths: A list of paths to the audio files to transcribe.

        Returns:
            A list of transcriptions for the audio.
        """
        raw_transcriptions = self._model.transcribe([str(i) for i in audio_paths])
        transcriptions = []
        for transcription in raw_transcriptions:
            text = transcription[TRANSCRIPTION_FIELD]

            timestamps = []
            for i in range(len(text)):
                start, end = transcription[START_TIMESTAMPS_FIELD][i], transcription[END_TIMESTAMPS_FIELD][i]
                start, end = start * self._timestamp_to_sec_ratio, end * self._timestamp_to_sec_ratio
                timestamps.append((start, end))

            transcription = ASRTranscription(text=text, timestamps=timestamps)
            transcriptions.append(transcription)

        return transcriptions

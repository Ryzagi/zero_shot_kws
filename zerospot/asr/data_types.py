import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class ASRTranscription:
    """A class representing an ASR transcription.

    Attributes:
        text: The transcribed text.
        timestamps: A list of tuples representing the start and end times of each char in the transcription.
    """

    text: str
    timestamps: List[Tuple[float, float]]

    def to_json(self, path: Path) -> None:
        """Save the transcription to a JSON file at the specified path.

        Args:
            path: The path to the JSON file to save to.
        """
        path.write_text(json.dumps(self.__dict__))

    @classmethod
    def from_json(cls, path: Path) -> "ASRTranscription":
        """Load a transcription from a JSON file at the specified path.

        Args:
            path: The path to the JSON file to load from.

        Returns:
            The loaded transcription.
        """
        return cls(**json.loads(path.read_text()))

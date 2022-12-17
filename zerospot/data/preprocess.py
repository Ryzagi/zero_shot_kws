import re
import pandas as pd
import pydub
from zerospot.asr.data_types import ASRTranscription
from pathlib import Path


def zero_shot_create_dataset(path):
    path = Path(path).rglob('*.wav')

    wav_paths = []

    words = []

    start_secs = []

    end_secs = []

    for wav_path in path:

        asr_data = ASRTranscription.from_json(wav_path.with_suffix('.asr.json'))

        word_indexes = [(m.start(0), m.end(0)) for m in re.finditer('[A-z]+', asr_data.text)]

        for start, end in word_indexes:
            start_sec, end_sec = asr_data.timestamps[start][0], asr_data.timestamps[end - 1][1]

            words.append(asr_data.text[start:end])

            start_secs.append(start_sec)

            end_secs.append(end_sec)

            wav_paths.append(str(wav_path.absolute()))

    dataframe = pd.DataFrame(zip(wav_paths, words, start_secs, end_secs),
                             columns=['wav_paths', 'words', 'start_secs', 'end_secs'])
    dataframe.to_csv('zero_shot_full.csv')

    return dataframe


def convert_mp3_to_wav(path_to_mp3, path_to_wav):

    for mp3 in Path(path_to_mp3).rglob('*.mp3'):
        sound = pydub.AudioSegment.from_mp3(mp3)
        sound.export(path_to_wav + str(mp3.stem) + ".wav", format="wav")

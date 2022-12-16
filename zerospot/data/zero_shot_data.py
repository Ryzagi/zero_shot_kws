from pathlib import Path

import librosa

from zerospot.asr.data_types import ASRTranscription
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from zerospot.data.tokenizer import Tokenizer


class ZeroShotDataClass(Dataset):
    def __init__(self, path_to_csv):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.tokenizer = Tokenizer()

    def __getitem__(self, x):

        wav_path = self.data.loc[x, 'wav_paths']
        start_sec = self.data.loc[x, 'start_secs']
        end_sec = self.data.loc[x, 'end_secs']
        word = self.data.loc[x, 'words']
        wav_data, sr = librosa.load(wav_path, offset=start_sec, duration=end_sec - start_sec)

        path_to_json = Path(wav_path).with_suffix('.asr.json')

        asr_data = ASRTranscription.from_json(path_to_json)

        mel_spectrogram = librosa.feature.melspectrogram(y=wav_data, sr=16000, n_mels=40)
        mel_spectrogram = torch.Tensor(mel_spectrogram)

        word_label = 1

        if random.random() > 0.5:
            word = ''.join(random.sample(word, k=len(word)))
            word_label = 0

        return mel_spectrogram, word, word_label

    def collate_fn(self, batch):

        max_spec_length = max([spec.shape[1] for spec, word, word_label in batch])

        word_labels = []
        # Create an empty list to store the padded tensors.
        specs = []
        tokens_id, tokens_lentghts = self.tokenizer(list(zip(*batch))[1])

        #words = []
        for spec, word, word_label in batch:
            # Calculate the padding size needed for this spectrogram.
            pad_size = max_spec_length - spec.shape[1]

            # Pad the spectrogram to the maximum length.
            spec = torch.nn.functional.pad(spec, (0, pad_size, 0, 0), "constant", 0)

            # Append the padded tensor to the list of spectrograms.
            specs.append(spec)

            word_labels.append(word_label)

            #words.append(word)

        #for token_id in tokens_lentghts:
        #    if token_id == 0:
        #        print(tokens_id, tokens_lentghts, words)

        targets_tensor = torch.LongTensor(word_labels)
        tokens_ids_tensor = torch.LongTensor(tokens_id)
        tokens_lentghts_tensor = torch.LongTensor(tokens_lentghts)

        # Stack the padded tensors.
        spec_tensors = torch.stack(specs)

        return spec_tensors, tokens_ids_tensor, tokens_lentghts_tensor, targets_tensor

    def __len__(self):
        return len(self.data)
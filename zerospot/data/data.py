from pathlib import Path

import librosa
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class WavDataClass(Dataset):
    def __init__(self, path_to_data):
        super().__init__()
        self.data = pd.read_csv(path_to_data)

    def __getitem__(self, x):
        idx, label, wav_path = self.data.iloc[x]

        # Load the audio data using librosa
        audio_data, sample_rate = librosa.load(wav_path, sr=16000)

        # Create a melspectrogram  with the desired parameters
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=16000, n_mels=40
        )

        # Convert a melspectrogram  with the desired parameters into torch.Tensor
        mel_spectrogram = torch.Tensor(mel_spectrogram)

        return mel_spectrogram, idx

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    # Get the maximum length of the spectrograms in the batch.
    max_length = max([spec.shape[1] for spec, target in batch])

    # Create an empty list to store the padded tensors.
    specs = []

    # Create an empty list to store the targets.
    targets = []

    # Loop over the spectrograms in the batch.
    for spec, target in batch:

        # Calculate the padding size needed for this spectrogram.
        pad_size = max_length - spec.shape[1]

        # Pad the spectrogram to the maximum length.
        spec = torch.nn.functional.pad(spec, (0, pad_size, 0, 0), "constant", 0)

        # Append the padded tensor to the list of spectrograms.
        specs.append(spec)

        # Append the target to the list of targets.
        targets.append(target)

    # Stack the padded tensors.
    tensors = torch.stack(specs)

    # Convert the targets to a tensor.
    targets = torch.LongTensor(targets)

    return tensors, targets


def create_datasets(path, test_size):
    # create an empty list to store the data
    data = []

    # get a list of all the .wav files in the given path
    dataset_from_path = Path(path).rglob("*.wav")

    # loop through each .wav file in the list
    for wav in dataset_from_path:

        # get the label of the .wav file by getting the parent folder name
        label = wav.parent.name

        # append the label and file path to the data list
        data.append([0, label, wav])

    # create a pandas DataFrame from the data list
    dataset = pd.DataFrame(data, columns=["indexes", "labels", "wav_path"])

    # create a dictionary mapping the labels to index values
    mapped_labels = {v: k for k, v in enumerate(set(dataset.labels))}

    # use the dictionary to convert the labels to index values
    dataset["indexes"] = dataset.labels.apply(lambda x: mapped_labels[x])

    # split the dataset into training and testing sets
    train, test = train_test_split(
        dataset, test_size=test_size, random_state=42
    )

    # save the training and testing sets as csv files
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

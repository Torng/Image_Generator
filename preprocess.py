import torchaudio
from pathlib import Path
import torch
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.whole_training_data = self.load_data()
        self.training_data = []

    def load_data(self):
        training_data = []
        max_size = float('-inf')
        for file_path in self.folder_path.glob("*.wav"):
            data, sample_rate = torchaudio.load(file_path)
            # new_data = data
            # new_data = (data[0] + data[1]) / 2
            # new_data = new_data.view(1, -1)
            if data.shape[1] > 65924:  # 132300 = 44100*3 3seconds music
                data = data[:, :65924]
            elif data.shape[1] < 65924:
                data = torch.cat((data, torch.zeros(1, 65924 - data.shape[1])), 1)
            new_data = self.data_process(data)
            training_data.append(new_data)
        return training_data

    def data_process(self, data):
        transform = torchaudio.transforms.Spectrogram(n_fft=1022, normalized=True,win_length=700)
        spectrogram_data = transform(data)
        return spectrogram_data

    def plot_specgram(self, waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)

    def set_batch_size(self, batch_size=4):
        for batch in range(0, len(self.whole_training_data), batch_size):
            batch_data = self.whole_training_data[batch]
            for i in range(batch, batch + batch_size - 1):
                i = i % len(self.whole_training_data)
                # if i >= len(self.whole_training_data):
                #     break
                batch_data = torch.cat((batch_data, self.whole_training_data[i]), 0)
            self.training_data.append(batch_data)

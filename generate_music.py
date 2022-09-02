import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
model = torch.load("model_1", map_location=device)
model.eval()

z = torch.randn(1, 1, 1, 128, device=device)
new_music = model.forward(z)
new_music = new_music[0]
new_music = new_music.to(torch.cdouble)
transform = torchaudio.transforms.InverseSpectrogram(n_fft=1022, normalized=True, win_length=700)
new_music = transform(new_music)
# play_audio(waveform,sample_rate)float32
new_music = new_music.to(torch.float32)
torchaudio.save("test.wav", new_music, 44100)

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the data from the provided .mat file
mat = scipy.io.loadmat('ex3M2.mat')
signal = mat['we_be_10k'].flatten()
fs = 10000  # Sampling rate is 10,000 Hz

# Function to plot the spectrogram with given window and hop lengths
def plot_spectrogram(y, sr, win_ms, hop_ms, title):
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    D = librosa.stft(y, n_fft=1024, hop_length=hop_length, win_length=win_length, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title} (win={win_ms}ms, hop={hop_ms}ms)")
    plt.tight_layout()
    plt.show()

# 1. 20ms window, 1ms frame interval (Figure 3.33b)
plot_spectrogram(signal, fs, win_ms=20, hop_ms=1, title="Spectrogram - 20ms Window (Wideband)")

# 2. 5ms window, 1ms frame interval
plot_spectrogram(signal, fs, win_ms=5, hop_ms=1, title="Spectrogram - 5ms Window (Narrowband)")

# 3. Additional experiments for time-frequency tradeoff
plot_spectrogram(signal, fs, win_ms=3, hop_ms=1, title="Spectrogram - 3ms Window")
plot_spectrogram(signal, fs, win_ms=30, hop_ms=1, title="Spectrogram - 30ms Window")
plot_spectrogram(signal, fs, win_ms=50, hop_ms=2, title="Spectrogram - 50ms Window, 2ms Hop")
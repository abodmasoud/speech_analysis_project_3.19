import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd

# Load data
mat = scipy.io.loadmat('ex3M2.mat')
signal = mat['we_be_10k'].flatten()
fs = 10000  # Sampling rate

# Step 1: Extract the word "be"
# Approximate time range (adjust if needed)
be_start = int(0.45 * fs)
be_end = int(0.65 * fs)
be = signal[be_start:be_end]

# Display waveform and spectrogram
plt.figure(figsize=(10, 3))
plt.plot(np.linspace(0, len(be)/fs, len(be)), be)
plt.title("Waveform of 'be'")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()

# Spectrogram of "be"
D = librosa.stft(be, n_fft=1024, hop_length=10, win_length=200, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=fs, hop_length=10, x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of 'be'")
plt.tight_layout()
plt.show()

# Step 2: Splice out "b" and increase VOT
# Approximate: first 100 samples is the burst ("b"), rest is vowel ("ee")
b_burst = be[:100]
vowel_part = be[100:]

# Create silent gap to simulate increased VOT
vot_gap = np.zeros(int(0.05 * fs))  # 50 ms gap

# Synthesize modified "be"
modified_be = np.concatenate([b_burst, vot_gap, vowel_part])

# Play the result
print("== Playing 'be' with increased VOT ==")
sd.play(modified_be, fs)
sd.wait()
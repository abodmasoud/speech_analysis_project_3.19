import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
import librosa.display

# Load the .mat file
mat = scipy.io.loadmat('ex3M2.mat')
signal = mat['we_be_10k'].flatten()
fs = 10000  # Sampling rate = 10kHz

# Extract the word "be" (approximate region)
be_start = int(0.45 * fs)
be_end = int(0.65 * fs)
be = signal[be_start:be_end]

# Display original waveform
plt.figure(figsize=(10, 3))
plt.plot(np.linspace(0, len(be)/fs, len(be)), be)
plt.title("Waveform of Original 'be'")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()

# Display original spectrogram
D = librosa.stft(be, n_fft=1024, hop_length=10, win_length=200, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=fs, hop_length=10, x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Original 'be'")
plt.tight_layout()
plt.show()

# Split "be" into b-burst and vowel (approximate)
b_burst = be[:100]          # Initial burst ("b")
vowel_part = be[100:]       # Rest of the vowel sound ("ee")

# Try different VOT values
vot_values = [0.02, 0.05, 0.1]  # 20ms, 50ms, 100ms

for vot in vot_values:
    vot_gap = np.zeros(int(vot * fs))  # Create silent gap
    modified_be = np.concatenate([b_burst, vot_gap, vowel_part])

    # Plot modified waveform
    plt.figure(figsize=(8, 2))
    plt.plot(modified_be)
    plt.title(f"Waveform of 'be' with VOT = {int(vot*1000)} ms")
    plt.tight_layout()
    plt.show()

    # Play modified sound
    print(f"== Playing 'be' with VOT = {int(vot*1000)} ms ==")
    sd.play(modified_be, fs)
    sd.wait()
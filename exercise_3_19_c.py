import scipy.io
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Load the signal
mat = scipy.io.loadmat('ex3M2.mat')
signal = mat['we_be_10k'].flatten()
fs = 10000  # 10 kHz sampling rate

# Approximate positions of "we" and "be"
we_start, we_end = int(0.15 * fs), int(0.35 * fs)
be_start, be_end = int(0.45 * fs), int(0.65 * fs)

# Extract the two words
we = signal[we_start:we_end]
be = signal[be_start:be_end]

# Slice initial phones (adjust if needed based on waveform)
w_phone = we[:80]          # "w" from "we"
e_from_we = we[80:]        # rest of "we"

b_phone = be[:80]          # "b" from "be"
e_from_be = be[80:]        # rest of "be"

# Create modified versions
we_with_b = np.concatenate([b_phone, e_from_we])
be_with_w = np.concatenate([w_phone, e_from_be])

# Add padding silence
silence = np.zeros(int(0.2 * fs))  # 200ms of silence

# Repeat each word twice with silence before and after
def prepare_playback(signal_piece):
    padded = np.concatenate([silence, signal_piece, silence])
    repeated = np.tile(padded, 2)  # repeat the word twice
    return repeated

# Prepare and play: "we" with "b"
we_with_b_ready = prepare_playback(we_with_b)
print("== Playing 'we' with 'b' instead of 'w' ==")
sd.play(we_with_b_ready, fs)
sd.wait()

# Prepare and play: "be" with "w"
be_with_w_ready = prepare_playback(be_with_w)
print("== Playing 'be' with 'w' instead of 'b' ==")
sd.play(be_with_w_ready, fs)
sd.wait()

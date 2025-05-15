import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Load the waveform
mat = scipy.io.loadmat('ex3M2.mat')
signal = mat['we_be_10k'].flatten()
fs = 10000  # Sampling rate

# Step 1: Extract the two words
# Approximate time ranges (can be adjusted based on listening or plotting)
we_start, we_end = int(0.15 * fs), int(0.35 * fs)
be_start, be_end = int(0.45 * fs), int(0.65 * fs)

we = signal[we_start:we_end]
be = signal[be_start:be_end]

# Step 2: Slice phones from each word
# Approximate: first 80 samples for the initial consonants
w_phone = we[:80]         # initial "w" from "we"
e_from_we = we[80:]       # rest of the word "we"

b_phone = be[:80]         # initial "b" from "be"
e_from_be = be[80:]       # rest of the word "be"

# Step 3: Construct new words by swapping the initial phones
we_with_b = np.concatenate([b_phone, e_from_we])
be_with_w = np.concatenate([w_phone, e_from_be])

# Step 4: Playback the results
print("== Playing original 'we' with 'b' instead of 'w' ==")
sd.play(we_with_b, fs)
sd.wait()

print("== Playing original 'be' with 'w' instead of 'b' ==")
sd.play(be_with_w, fs)
sd.wait()

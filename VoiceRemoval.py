import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa

import librosa.display

y, sr = librosa.load("/content/Pop2.wav", duration = 120)

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

# Convert the spectrogram back to audio
audio_background = librosa.istft(S_background * phase)

# Play the audio
ipd.Audio(audio_background, rate=sr)

audio_foreground = librosa.istft(S_foreground * phase)

ipd.Audio(audio_foreground, rate=sr)

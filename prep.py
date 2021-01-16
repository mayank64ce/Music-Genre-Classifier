import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)  # number of elements in signal array = sr * T =
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()
# fft -> spectrum
# fft = np.fft.fft(signal)
#
# magnitudes = np.abs(fft)
# frequencies = np.linspace(0, sr, len(magnitudes))
# plt.plot(frequencies, magnitudes)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()
# stft -> spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# MFCCs

MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()

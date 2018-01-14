import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from mediaio.audio_io import AudioSignal
from dsp.spectrogram import MelConverter

# signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/test.wav')
signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/source.wav')
# signal = AudioSignal.from_wav_file('/cs/labs/peleg/testing/mix.wav')

stft = lb.stft(signal.get_data(), n_fft=640, hop_length=160)
spec, phase = lb.core.magphase(stft)

real = stft.real
imag = stft.imag

log_real = lb.amplitude_to_db(np.abs(real))
log_imag = lb.amplitude_to_db(np.abs(imag))
log_spec = lb.amplitude_to_db(spec)

plt.figure()
plt.pcolormesh(log_spec)

plt.figure()
plt.pcolormesh(log_real)

plt.figure()
plt.pcolormesh(log_imag)

plt.figure()
plt.pcolormesh(np.sign(real) * log_real)

plt.figure()
plt.pcolormesh(np.sign(imag) * log_imag)

real[real>0] = 1
real[real<0] = 0

plt.figure()
plt.pcolormesh(real)



plt.show()
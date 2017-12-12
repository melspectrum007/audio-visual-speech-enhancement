import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from mediaio.audio_io import AudioSignal
from dsp.spectrogram import MelConverter

# signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/test.wav')
signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/mix.wav')
signal = AudioSignal.from_wav_file('/cs/labs/peleg/testing/mix.wav')

spec, phase = lb.core.magphase(lb.stft(signal.get_data(), n_fft=640, hop_length=160))

log_spec = lb.amplitude_to_db(spec)
dct_spec = dct(log_spec)
# dct_spec = dct(spec)
# plt.pcolormesh(log_spec)
# plt.figure()
# plt.pcolormesh(dct_spec)
index = 95
plt.plot(dct_spec[1:, index])
# plt.plot(log_spec[index,:])
print '12_norm: ', np.linalg.norm(dct_spec[1:13, index], ord=1), 'all_norm: ', np.linalg.norm(dct_spec[1:, index], ord=1)

plt.show()
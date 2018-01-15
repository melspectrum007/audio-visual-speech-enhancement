import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from mediaio.audio_io import AudioSignal
from dsp.spectrogram import MelConverter

# signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/test.wav')
signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/source.wav')
# signal = AudioSignal.from_wav_file('/cs/labs/peleg/testing/mix.wav')
mean, std = signal.normalize()

stft = lb.stft(signal.get_data(), n_fft=640, hop_length=160)
# spec, phase = lb.core.magphase(stft)

real = stft.real
imag = stft.imag

log_real = np.log(1 + np.abs(real))
log_imag = np.log(1 + np.abs(imag))

signed_log_real = np.sign(real) * log_real
signed_log_imag = np.sign(imag) * log_imag

# plt.figure()
# plt.pcolormesh(log_real)
# plt.colorbar()
#
# plt.figure()
# plt.pcolormesh(log_imag)
# plt.colorbar()
#
# plt.figure()
# plt.pcolormesh(signed_log_real)
# plt.colorbar()
#
# plt.figure()
# plt.pcolormesh(signed_log_imag)
# plt.colorbar()

recon_real = np.sign(signed_log_real) * (np.exp(np.abs(signed_log_real)) - 1)
recon_imag = np.sign(signed_log_imag) * (np.exp(np.abs(signed_log_imag)) - 1)

stft = recon_real + 1j * recon_imag
data = lb.istft(stft, hop_length=160)
data *= std
data += mean
data = data.astype('int16')

AudioSignal(data, 16000).save_to_wav_file('/cs/grad/asaph/testing/bbb.wav')
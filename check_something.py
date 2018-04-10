
import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import *
from mediaio.audio_io import AudioSignal
import librosa as lb

s = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/pbai6a_s2.wav')
g = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/geese.wav')

while g.get_number_of_samples() < s.get_number_of_samples():
	g = AudioSignal.concat([g, g])

g.truncate(s.get_number_of_samples())


s = s.get_data().astype('f')
g = g.get_data().astype('f')


mag_s, phase_s = lb.magphase(lb.stft(s, 640, 160))
mag_g, phase_g = lb.magphase(lb.stft(g, 640, 160))

s_plus_g = lb.istft(mag_s * phase_g).astype(np.int16)
g_plus_s = lb.istft(mag_g * phase_s).astype(np.int16)
s = lb.istft(mag_s * phase_s).astype(np.int16)
g = lb.istft(mag_g * phase_g).astype(np.int16)

AudioSignal(s_plus_g, 16000).save_to_wav_file('/cs/grad/asaph/testing/s_g.wav')
AudioSignal(g_plus_s, 16000).save_to_wav_file('/cs/grad/asaph/testing/g_s.wav')
AudioSignal(s, 16000).save_to_wav_file('/cs/grad/asaph/testing/s_s.wav')
AudioSignal(g, 16000).save_to_wav_file('/cs/grad/asaph/testing/g_g.wav')




# raw_data = s.get_data().astype('f')
#
# pl.plot(raw_data)
#
# data = np.abs(raw_data)
# # pl.figure()
# # pl.plot(data)
#
# k = 25.
#
# med = medfilt(data, kernel_size=int(k))
# mean = np.convolve(data, np.arange(k) / k, mode='same')
#
#
# pl.figure()
# pl.plot(mean)
# # pl.figure()
# # pl.plot(med)
#
# thresh = 10000
#
# line = (mean > thresh) * 10000
# pl.figure(1)
# pl.plot(line, 'r')
#
# pl.show()



